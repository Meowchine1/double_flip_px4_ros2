import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3
from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import (VehicleAttitude, VehicleImu, ActuatorOutputs, 
                          VehicleLocalPosition,SensorCombined,VehicleAngularVelocity,
                          VehicleAngularAccelerationSetpoint, VehicleMagnetometer, SensorBaro) # TODO CONNECT VehicleMagnetometer SensorBaro
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from scipy.spatial.transform import Rotation as R
 
from rclpy.time import Time


from nav_msgs.msg import Odometry  # inner ROS2 SEKF
from filterpy.kalman import ExtendedKalmanFilter

 

# ===== MATRIX OPERTIONS =====
# ===== QUATERNION UTILS (SCIPY-based) =====


SEA_LEVEL_PRESSURE = 101325.0

def quat_multiply(q1, q2):
    """Multiply two quaternions [x, y, z, w] using scipy."""
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    return (r1 * r2).as_quat()

def normalize_quat(q):
    """Normalize a quaternion."""
    return R.from_quat(q).as_quat()

def quat_derivative(q, omega):
    """Compute quaternion derivative from angular velocity."""
    r = R.from_quat(q)
    Omega = R.from_rotvec(omega * 0.5)
    dq = (Omega * r).as_quat()  # Small-angle rotation
    return dq

def quat_to_matrix(q):
    """Convert quaternion [x, y, z, w] to rotation matrix."""
    return R.from_quat(q).as_matrix()

# ==========
class DynamicModelNode(Node):
    def __init__(self):
        super().__init__('dynamic_model_node')
        
        self.get_logger().info(f"DynamicModelNode")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        # ======= PUBLISHERS =======
        self.pose_pub = self.create_publisher(PoseStamped, '/quad/pose_pred', qos_profile)
        self.motor_pub = self.create_publisher(Float32MultiArray, '/fmu/motor_inputs', qos_profile)
        self.position_pub = self.create_publisher(PoseStamped, '/drone/imu_position', qos_profile)
        self.imu_pos_err_pub = self.create_publisher(Vector3, '/drone/imu_pos_err', qos_profile)

        # ======= SUBSCRIBERS =======
        self.create_subscription(SensorCombined, '/fmu/out/sensor_combined', self.sensor_combined_callback, qos_profile)
        self.create_subscription(VehicleAngularVelocity, '/fmu/out/vehicle_angular_velocity', self.angular_velocity_callback, qos_profile)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.vehicle_attitude_callback, qos_profile)
        self.create_subscription(VehicleAngularAccelerationSetpoint,
        '/fmu/out/vehicle_angular_acceleration_setpoint', self.vehicle_angular_acceleration_setpoint_callback, qos_profile)
        self.create_subscription(VehicleImu,'/fmu/out/vehicle_imu',self.vehicle_imu_callback, qos_profile)
        self.create_subscription(ActuatorOutputs, '/fmu/out/actuator_outputs', self.actuator_outputs_callback, qos_profile)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        # FOR EKF
        self.create_subscription(SensorBaro, '/fmu/out/sensor_baro', self.sensor_baro_callback, qos_profile)
        self.create_subscription(VehicleMagnetometer, '/fmu/out/vehicle_magnetometer', self.vehicle_magnetometer_callback, qos_profile)


        # Подписываемся на топик с результатами фильтрации
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)

        # ======= DRONE CONSTRUCT PARAMETERS =======
        self.mass = 0.82
        self.inertia = np.diag([0.045, 0.045, 0.045])
        self.arm_length = 0.15
        self.k_thrust = 1.48e-6
        self.k_torque = 9.4e-8
        self.motor_tau = 0.02
        self.max_speed = 2100.0
        self.drag = 0.1
        self.max_rate = 25.0  # рад/с, ограничение на угловую скорость (roll/pitch)

        self.angularVelocity = np.zeros(3, dtype=np.float32)
        self.vehicleImu_velocity = np.zeros(3, dtype=np.float32) # в мировых координатах
        self.vehicleImu_linear_acceleration = np.zeros(3, dtype=np.float32)
        self.position = [0.0, 0.0, 0.0] # drone position estimates with IMU localization
        self.motor_inputs = np.zeros(4)  # нормализованные входы [0..1] или RPM
        self.vehicleAttitude_q = [0.0, 0.0, 0.0, 1.0] # momental quaternion from topic
        self.magnetometer_data = np.zeros(3, dtype=np.float32)
        self.baro_attitude = 0.0
        # FOR TESTING IN SITL
        self.vehicleLocalPosition_position = np.zeros(3, dtype=np.float32)  

        # ======= TOPIC DATA ======= 
        # :[TOPIC NAME]_[PARAM NAME] OR [TOPIC NAME] IF PARAM = TOPIC NAME
        self.vehicleImu_velocity = np.zeros(3, dtype=np.float32) # Текущая линейная скорость (в мировых координатах)
        self.sensorCombined_linear_acceleration = np.zeros(3, dtype=np.float32)
        self.sensorCombined_q = [0.0, 0.0, 0.0, 1.0] #  float32  custom calculation
        self.sensorCombined_angular_velocity = np.zeros(3, dtype=np.float32)
        self.vehicleImu_angular_velocity = np.zeros(3, dtype=np.float32)
        self.angularVelocity_angular_acceleration = np.zeros(3, dtype=np.float32)
        self.vehiclAngularAcceleration_angular_acceleration = np.zeros(3, dtype=np.float32)
        self.baro_temperature = 0.0 # temperature in degrees Celsius

        # ======= EKF =======
        self.ekf = ExtendedKalmanFilter(dim_x=13, dim_z=4)  # z: baro (1) + magnetometer (3)
        # Подключаем функцию перехода состояния
        self.ekf.fx = self.fx
        # Подключаем функцию измерения
        self.ekf.hx = self.hx
        # Состояние: позиция, скорость, кватернион
        self.ekf.x = np.zeros(13)
        self.ekf.x[6] = 1.0  # qw = 1 (единичный кватернион)

        # Начальные ковариации
        self.ekf.P *= 0.1
        self.ekf.Q *= 0.01
        self.ekf.R = np.diag([0.5, 0.05, 0.05, 0.05])  # baro + mag
        
        # ======= TIMERS =======
        self.timer = self.create_timer(0.01, self.step_dynamics)
        self.position_IMU_timer = self.create_timer(0.01, self.position_IMU)
        self.last_time = Time.time()


    def odom_callback(self, msg: Odometry):
        # Обработка данных, например, извлечение позиции
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.get_logger().info(f'Position: x={position.x}, y={position.y}, z={position.z}')
        self.get_logger().info(f'Orientation: x={orientation.x}, y={orientation.y}, z={orientation.z}, w={orientation.w}')

    def sensor_baro_callback(self, msg):
        self.baro_temperature = msg.baro_temperature
        self.baro_attitude = 44330.0 * (1.0 - (msg.baro_pressure / SEA_LEVEL_PRESSURE) ** 0.1903)

    def vehicle_magnetometer_callback(self, msg: VehicleMagnetometer):
        # Измерения магнитометра
        self.magnetometer_data = np.array([msg.x, msg.y, msg.z], dtype=np.float32)

    # ПОЗИЦИЯ ДЛЯ ОЦЕНКИ ИНЕРЦИАЛНОЙ ЛОКАЛИЗАЦИИ
    def vehicle_local_position_callback(self, msg: VehicleLocalPosition):
        #self.get_logger().info(f"vehicle_local_position_callback {msg.x} {msg.y} {msg.z}")
        self.vehicleLocalPosition_position[0] = msg.x
        self.vehicleLocalPosition_position[1] = msg.y
        self.vehicleLocalPosition_position[2] = msg.z

    def actuator_outputs_callback(self, msg: ActuatorOutputs):
        pwm_outputs = msg.output[:4]  # предполагаем, что 0-3 — это моторы
        # Преобразование PWM в радианы в секунду (линейное приближение)
        self.motor_inputs = np.clip((np.array(pwm_outputs) - 1000.0) / 1000.0 * self.max_speed, 0.0, self.max_speed)

    # ЛИНЕЙНОЕ УСКОРЕНИЕ, УГЛОВОЕ УСКОРЕНИЕ, КВАТЕРНИОН
    def sensor_combined_callback(self, msg: SensorCombined):
        dt_gyro = msg.gyro_integral_dt * 1e-6  # микросекунды -> секунды
        gyro_rad = np.array(msg.gyro_rad, dtype=np.float32)  # угловая скорость (рад/с)
        self.sensorCombined_angular_velocity = gyro_rad
         
        delta_angle = gyro_rad * dt_gyro # Угловое приращение (рад)
        self.sensorCombined_delta_angle = delta_angle
     
        self.sensorCombined_linear_acceleration = np.array(msg.accelerometer_m_s2, dtype=np.float32)
         
    # УГЛОВАЯ СКОРОСТЬ И УСКОРЕНИЕ
    def angular_velocity_callback(self, msg: VehicleAngularVelocity):
        self.angularVelocity = np.array(msg.xyz, dtype=np.float32)
        self.angularVelocity_angular_acceleration = np.array(msg.xyz_derivative, dtype=np.float32)

    def vehicle_attitude_callback(self, msg: VehicleAttitude):
        # In this system are used scipy format for quaternion. 
        # PX4 topic uses the Hamilton convention, and the order is q(w, x, y, z). So we reorder it
        self.vehicleAttitude_q = np.array([msg.q[1], msg.q[2], msg.q[3], msg.q[0]], dtype=np.float32)
        
    # УГЛОВОЕ УСКОРЕНИЕ
    def vehicle_angular_acceleration_setpoint_callback(self, msg: VehicleAngularAccelerationSetpoint):
        self.vehiclAngularAcceleration_angular_acceleration = msg.xyz

    # ЛИНЕЙНАЯ СКОРОСТЬ И УСКОРЕНИЕ  угловая скорость и ускорение
    def vehicle_imu_callback(self, msg: VehicleImu):
        # Сохраняем приращения
        delta_angle = np.array(msg.delta_angle, dtype=np.float32)         # рад
        delta_angle_dt = msg.delta_angle_dt * 1e-6     # с

        delta_velocity = np.array(msg.delta_velocity, dtype=np.float32)   # м/с
        delta_velocity_dt = msg.delta_velocity_dt * 1e-6  # с

        # УГЛОВАЯ СКОРОСТЬ
        if delta_angle_dt > 0:
            self.vehicleImu_angular_velocity = delta_angle / delta_angle_dt
        
        self.vehicleImu_velocity += delta_velocity
        if delta_velocity_dt > 0:
            # Преобразуем скорость из локальной системы в мировую
            if self.vehicleAttitude_q is not None:
                # Используем кватернион ориентации для преобразования
                rotation = R.from_quat(self.vehicleAttitude_q)
                # Преобразуем линейное ускорение в мировую систему координат
                velocity_world = rotation.apply(self.vehicleImu_velocity)
                
                # Интегрируем скорость по времени для вычисления изменения позиции
                self.position += velocity_world * delta_velocity_dt


    def publish_motor_inputs(self):
        msg = Float32MultiArray()
        msg.data = self.motor_inputs.tolist()
        self.motor_pub.publish(msg)

  
    def complementary_filter(self):
        """
        Фильтр комплементарности для вычисления ориентации (pitch, roll, yaw)
        
        :param accel_data: Данные акселерометра (accel_x, accel_y, accel_z)
        :param gyro_data: Данные гироскопа (gyro_x, gyro_y, gyro_z)
        :param mag_data: Данные магнитометра (mag_x, mag_y, mag_z)
        :param dt: Время между двумя измерениями (в секундах)
        """

        accel_data=self.vehicleImu_linear_acceleration
        gyro_data=self.vehicleImu_angular_velocity
        mag_data=self.magnetometer_data
        dt=0.01
        # Акселерометрические углы (pitch, roll) вычисляются на основе ускорения
        accel_pitch = math.atan2(accel_data[1], accel_data[2])
        accel_roll = math.atan2(-accel_data[0], math.sqrt(accel_data[1]**2 + accel_data[2]**2))
        
        # Интегрирование угловой скорости гироскопа для получения углов
        gyro_pitch = self.pitch + gyro_data[0] * dt
        gyro_roll = self.roll + gyro_data[1] * dt
        gyro_yaw = self.yaw + gyro_data[2] * dt
        
        # Использование фильтра комплементарности для комбинирования данных акселерометра и гироскопа
        alpha = 0.98  # Коэффициент фильтра (можно настраивать)
        
        self.pitch = alpha * gyro_pitch + (1 - alpha) * accel_pitch
        self.roll = alpha * gyro_roll + (1 - alpha) * accel_roll

        # Магнитометр для корректировки yaw
        mag_yaw = math.atan2(mag_data[1], mag_data[0])  # yaw из магнитометра
        
        # Коррекция yaw с использованием магнитометра
        self.yaw = gyro_yaw + (mag_yaw - gyro_yaw) * (1 - alpha)

        return self.pitch, self.roll, self.yaw
 
    def get_yaw_from_mag(self):
        r = R.from_quat(self.vehicleAttitude_q)  # [x, y, z, w]
    
        # Преобразуем магнитное поле в мировую систему координат
        mag_world = r.apply(self.magnetometer_data)

        # Проекция магнитного поля на горизонтальную плоскость (X, Y)
        mag_x = mag_world[0]
        mag_y = mag_world[1]
        # Вычисляем yaw (в радианах)
        yaw_from_magnetometer = np.arctan2(-mag_y, mag_x)
        return yaw_from_magnetometer
    
    def EKF(self):
         
        z = [self.baro_attitude, self.get_yaw_from_mag]

        # Предсказание следующего состояния
        self.ekf.predict(dt=0.01)

        # Обновление состояния фильтра с использованием новых данных
        self.ekf.update(z)


    def hx(self, x):
        """ модель измерений: измеряется  """

        # baro

        # mag
        

        return np.array([x[0]])  # Возвращаем только позицию из состояния

    def fx(self, x, dt):
        """
        Нелинейная динамическая модель квадрокоптера
        dt - шаг времени
        x - вектор состояния [позиция, скорость, кватернион, угловая скорость]
        u - управление (обороты моторов) [w1, w2, w3, w4] 
        """
        # параметры
        m = self.mass
        I = self.inertia
        arm = self.arm_length
        kf = self.k_thrust
        km = self.k_torque
        drag = self.drag
        g = np.array([0, 0, 9.81])
        max_rate = self.max_rate
        max_speed = self.max_speed

        pos = x[0:3]
        vel = x[3:6]
        quat = x[6:10]
        omega = x[10:13]

        u = self.motor_inputs  
        #dt = 0.01 # dt - шаг времени

        quat /= np.linalg.norm(quat)

        R_bw = R.from_quat(quat).as_matrix()  # Перевод ориентации в матрицу

        rpm = np.clip(u, 0, max_speed) # Вычисляем тягу каждого мотора
        w_squared = rpm**2
        thrusts = kf * w_squared

        Fz = np.array([0, 0, np.sum(thrusts)]) # Общая тяга в теле

        F_world = R_bw @ Fz - m * g - drag * vel # Сила в мировой системе с компенсацией гравитации и сопротивления

        # Обновление линейной скорости и позиции
        acc = F_world / m # ускорение в мировой ск, второй закон ньютона
        new_vel = vel + acc * dt
        new_pos = pos + vel * dt + 0.5 * acc * dt**2

        # Моменты (упрощённо: крестовина)
        L = arm
        tau = np.array([
            L * (thrusts[1] - thrusts[3]),
            L * (thrusts[2] - thrusts[0]),
            km * (w_squared[0] - w_squared[1] + w_squared[2] - w_squared[3])
        ])

        # Обновление угловой скорости (Жуковский)
        omega_dot = np.linalg.inv(I) @ (tau - np.cross(omega, I @ omega))
        new_omega = omega + omega_dot * dt

        # Обновление ориентации через кватернион
        omega_quat = np.concatenate(([0.0], omega))
        dq = 0.5 * quat_multiply(quat, omega_quat)
        new_quat = quat + dq * dt
        new_quat /= np.linalg.norm(new_quat)

        # Собираем новое состояние
        x_next = np.zeros(13)
        x_next[0:3] = new_pos
        x_next[3:6] = new_vel
        x_next[6:10] = new_quat
        x_next[10:13] = new_omega

        return x_next
        #return np.array([new_position, new_velocity])



 

def main(args=None):
    rclpy.init(args=args)
    node = DynamicModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

