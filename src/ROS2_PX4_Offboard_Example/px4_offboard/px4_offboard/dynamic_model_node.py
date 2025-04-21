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

from dataclasses import dataclass, field

# ===== MATRIX OPERTIONS =====
# ===== QUATERNION UTILS (SCIPY-based) =====

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
        self.imu_position_pub = self.create_publisher(PoseStamped, '/drone/imu_position', qos_profile)
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
        self.create_subscription(SensorBaro, '/fmu/out/sensor_baro', self.sensor_baro_callback, self.qos_profile)
        self.create_subscription(VehicleMagnetometer, '/fmu/out/vehicle_magnetometer', self.vehicle_magnetometer_callback, self.qos_profile)

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

        # ======= PREDICT PARAMETERS ======= 
        self.state = np.zeros(12, dtype=float)  # [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]

        self.offset = np.zeros(3, dtype=np.float32)
        
        # ======= DRONE LINEAR PARAMETERS ======= 
        # :[TOPIC NAME]_[PARAM NAME] OR [TOPIC NAME] IF PARAM = TOPIC NAME
        self.vehicleImu_velocity = np.zeros(3, dtype=np.float32) # Текущая линейная скорость (в мировых координатах)
        self.vehicleImu_linear_acceleration = np.zeros(3, dtype=np.float32)
        self.sensorCombined_linear_acceleration = np.zeros(3, dtype=np.float32)
        self.imu_position = [0.0, 0.0, 0.0] # drone position estimates with IMU localization
        self.motor_inputs = np.zeros(4)  # нормализованные входы [0..1] или RPM

        self.vehicleLocalPosition_position = np.zeros(3, dtype=np.float32) # actual position from 
         
        # ======= DRONE ANGULAR PARAMETERS =======  
        # :[TOPIC NAME]_[PARAM NAME] OR [TOPIC NAME] IF PARAM = TOPIC NAME
        # QUATERNIONS
        self.sensorCombined_q = [0.0, 0.0, 0.0, 1.0] #  float32 
        self.vehicleAttitude_q = [0.0, 0.0, 0.0, 1.0]
        self.q = [0.0, 0.0, 0.0, 1.0] # actual orientation

        # DELTA ANGLE
        self.sensorCombined_delta_angle = np.zeros(3, dtype=np.float32)
        self.vehicleImu_delta_angle = np.zeros(3, dtype=np.float32)
        self.vehicleImu_delta_velocity = np.zeros(3, dtype=np.float32)

        # ANGULAR VELOCITY
        self.sensorCombined_angular_velocity = np.zeros(3, dtype=np.float32)
        self.angularVelocity = np.zeros(3, dtype=np.float32)
        self.vehicleImu_angular_velocity = np.zeros(3, dtype=np.float32)

        # ANGULAR ACCELERATION
        self.angularVelocity_angular_acceleration = np.zeros(3, dtype=np.float32)
        self.vehiclAngularAcceleration_angular_acceleration = np.zeros(3, dtype=np.float32)

        # ======= EKF =======
        # Инициализация состояния EKF (9 элементов: [x, y, z, vx, vy, vz, roll, pitch, yaw])
        # Инициализация фильтра Калмана для позиции и ориентации
        self.kalman_state = np.zeros(12)  # Состояние Калмана: [x, y, z, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz]
        self.P = np.eye(12)  # Ковариационная матрица
        self.Q = np.eye(12) * 0.1  # Шум процесса (настраиваемый)
        self.R = np.eye(6) * 0.1  # Шум измерений (позиция и ориентация)
        self.H = np.zeros((6, 12))  # Матрица измерений
        self.H[:3, :3] = np.eye(3)  # Измерения позиции
        self.H[3:7, 6:10] = np.eye(4)  # Измерения ориентации

        self.magnetometer_data = np.zeros(3, dtype=np.float32)
        self.baro_pressure = 0.0 # static pressure measurement in Pascals
        self.baro_temperature = 0.0 # temperature in degrees Celsius

        # ======= TIMERS =======
        self.timer = self.create_timer(0.01, self.step_dynamics)

        self.last_time = Time.time()
        

    def vehicle_magnetometer_callback(self, msg: VehicleMagnetometer):
        # Измерения магнитометра
        self.magnetometer_data = np.array([msg.x, msg.y, msg.z], dtype=np.float32)

    def vehicle_local_position_callback(self, msg: VehicleLocalPosition):
        #self.get_logger().info(f"vehicle_local_position_callback {msg.x} {msg.y} {msg.z}")
        self.vehicleLocalPosition_position[0] = msg.x
        self.vehicleLocalPosition_position[1] = msg.y
        self.vehicleLocalPosition_position[2] = msg.z

    def actuator_outputs_callback(self, msg: ActuatorOutputs):
        pwm_outputs = msg.output[:4]  # предполагаем, что 0-3 — это моторы
        # Преобразование PWM в радианы в секунду (линейное приближение)
        self.motor_inputs = np.clip((np.array(pwm_outputs) - 1000.0) / 1000.0 * self.max_speed, 0.0, self.max_speed)

    def sensor_combined_callback(self, msg: SensorCombined):
        # === ГИРОСКОП ===
        dt_gyro = msg.gyro_integral_dt * 1e-6  # микросекунды -> секунды
        gyro_rad = np.array(msg.gyro_rad, dtype=np.float32)  # угловая скорость (рад/с)
        self.sensorCombined_angular_velocity = gyro_rad
         
        delta_angle = gyro_rad * dt_gyro # Угловое приращение (рад)
        self.sensorCombined_delta_angle = delta_angle

        # Обновляем ориентацию на основе приращения
        self.q = quat_multiply(self.q, R.from_rotvec(delta_angle).as_quat())
        # OR self.q = normalize_quat(quat_multiply(self.q, R.from_rotvec(delta_angle).as_quat()))

        # === АКСЕЛЕРОМЕТР ===
        # average value acceleration in m/s^2 over the last accelerometer sampling period
        self.sensorCombined_linear_acceleration = np.array(msg.accelerometer_m_s2, dtype=np.float32)
         
    def angular_velocity_callback(self, msg: VehicleAngularVelocity):
        self.angularVelocity = np.array(msg.xyz, dtype=np.float32)
        self.angularVelocity_angular_acceleration = np.array(msg.xyz_derivative, dtype=np.float32)

    def vehicle_attitude_callback(self, msg: VehicleAttitude):
        # In this system are used scipy format for quaternion. 
        # PX4 topic uses the Hamilton convention, and the order is q(w, x, y, z). So we reorder it
        self.vehicleAttitude_q = np.array([msg.q[1], msg.q[2], msg.q[3], msg.q[0]], dtype=np.float32)
        
    def vehicle_angular_acceleration_setpoint_callback(self, msg: VehicleAngularAccelerationSetpoint):
        self.vehiclAngularAcceleration_angular_acceleration = msg.xyz

    def vehicle_imu_callback(self, msg: VehicleImu):
        # Сохраняем приращения
        self.vehicleImu_delta_angle = np.array(msg.delta_angle, dtype=np.float32)         # рад
        self.vehicleImu_delta_velocity = np.array(msg.delta_velocity, dtype=np.float32)   # м/с
        delta_angle_dt = msg.delta_angle_dt * 1e-6     # с
        delta_velocity_dt = msg.delta_velocity_dt * 1e-6  # с

        # Вычисляем угловую скорость и ускорение
        if delta_angle_dt > 0:
            self.vehicleImu_angular_velocity = self.vehicleImu_delta_angle / delta_angle_dt
        if delta_velocity_dt > 0:
            accel_body = self.vehicleImu_delta_velocity / delta_velocity_dt
            self.vehicleImu_linear_acceleration = accel_body

        # Переводим ускорение в мировую систему координат (через кватернион self.q)
        rotation = R.from_quat(self.q)  # self.q = [x, y, z, w]
        accel_world = rotation.apply(accel_body)

        # Компенсируем гравитацию (предположим NED)
        gravity = np.array([0, 0, 9.81])
        accel_corrected = accel_world - gravity

        # Интегрируем ускорение, чтобы получить скорость
        self.vehicleImu_velocity += accel_corrected * delta_velocity_dt

    def publish_pose(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = self.state[0]
        msg.pose.position.y = self.state[1]
        msg.pose.position.z = self.state[2]
        msg.pose.orientation.x = self.q[0]
        msg.pose.orientation.y = self.q[1]
        msg.pose.orientation.z = self.q[2]
        msg.pose.orientation.w = self.q[3]
        self.pose_pub.publish(msg)

    def publish_motor_inputs(self):
        msg = Float32MultiArray()
        msg.data = self.motor_inputs.tolist()
        self.motor_pub.publish(msg)

    def imu_update(self):
        dt = 0.01  # фиксированный шаг времени

        # Сначала выполняем предсказание с использованием текущего состояния и переходной матрицы F
        self.x[:6] = self.x[:6] + self.x[3:6] * dt  # Обновляем позицию и скорость

        # Обновление матрицы перехода состояния F
        self.F[:3, 3:6] = np.eye(3) * dt  # Дельта-позиция зависит от скорости

        # Обновляем ковариационную матрицу
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        # Формируем измерения для обновления EKF (акселерометр, гироскоп, магнитометр)
        z = np.concatenate([self.sensorCombined_linear_acceleration, self.sensorCombined_angular_velocity, self.magnetometer_data])

        # Обновляем матрицу измерений H (связь между состоянием и измерениями)
        self.H[:3, 3:6] = np.eye(3)  # Акселерометр, связанный с ускорением
        self.H[3:6, 6:9] = np.eye(3)  # Гироскоп, связанный с угловыми скоростями
        self.H[6:9, 0:3] = np.eye(3)  # Магнитометр, связанный с позицией

        # Калькуляция остатка между измерением и предсказанным состоянием
        y = z - np.dot(self.H, self.x)

        # Обновление ковариационной матрицы
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Калмановский коэффициент

        # Обновление состояния
        self.x = self.x + np.dot(K, y)

        # Обновление ковариационной матрицы
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
    
    def publish_imu_localization_err(self):
        # Count error
        imu_pos = np.array(self.imu_position)
        px4_pos = np.array(self.vehicleLocalPosition_position)

        pos_err = imu_pos - px4_pos
        self.get_logger().info(f"imu pose err {pos_err[0]} {pos_err[1]} {pos_err[2]}")

        imu_q = np.array(self.q)
        px4_q = np.array(self.vehicleAttitude_q)
        quat_diff = imu_q - px4_q
        self.get_logger().info(f"imu quaternion err {quat_diff[0]:.3f} {quat_diff[1]:.3f} {quat_diff[2]:.3f} {quat_diff[3]:.3f}")

        # Публикация ошибки
        err_msg = Vector3()
        err_msg.x = float(pos_err[0])
        err_msg.y = float(pos_err[1])
        err_msg.z = float(pos_err[2])

        self.imu_pos_err_pub.publish(err_msg)

    # def publish_imu_localization(self):
    #     current_time = Time.time()

    #     dt = current_time - self.last_time
    #     self.last_time = current_time

    #     # === ОБНОВЛЕНИЕ ОРИЕНТАЦИИ ПО УГЛОВОЙ СКОРОСТИ ===
    #     omega = self.vehicleImu_angular_velocity  # [rad/s]
    #     q = np.array(self.q)                      # [x, y, z, w]
    #     q = np.roll(q, 1)                         # → [w, x, y, z]

    #     omega_quat = np.array([0.0, *omega])
    #     q_dot = 0.5 * quat_multiply(q, omega_quat)
    #     q_new = q + q_dot * dt
    #     q_new = q_new / np.linalg.norm(q_new)
    #     q_new = np.roll(q_new, -1)               # → [x, y, z, w]

    #     self.q = q_new.tolist()

    #     # === ПЕРЕВОД УСКОРЕНИЯ В МИРОВУЮ СК ===
    #     a_body = self.sensorCombined_linear_acceleration  # [m/s^2]
    #     R = quat_to_matrix(np.roll(q_new, 1))  # [w, x, y, z]
    #     a_world = R @ a_body

    #     # === КОМПЕНСАЦИЯ ГРАВИТАЦИИ ===
    #     g = np.array([0.0, 0.0, -9.81])
    #     a_world += g

    #     # === ИНТЕГРАЦИЯ СКОРОСТИ И ПОЛОЖЕНИЯ ===
    #     self.vehicleImu_velocity += a_world * dt
    #     self.imu_position += self.vehicleImu_velocity * dt

    #     # === ПУБЛИКАЦИЯ В ЛОГ ===
    #     self.get_logger().info(
    #         f"publish_imu_localization pose: {self.imu_position[0]:.3f} {self.imu_position[1]:.3f} {self.imu_position[2]:.3f}  "
    #         f"quaternion: {self.q[0]:.3f} {self.q[1]:.3f} {self.q[2]:.3f} {self.q[3]:.3f}"
    #     )

    #     # === ПУБЛИКАЦИЯ ROS-СООБЩЕНИЯ ===
    #     pose_msg = PoseStamped()
    #     pose_msg.header.stamp = self.get_clock().now().to_msg()
    #     pose_msg.header.frame_id = 'map'

    #     pose_msg.pose.position.x = float(self.imu_position[0])
    #     pose_msg.pose.position.y = float(self.imu_position[1])
    #     pose_msg.pose.position.z = float(self.imu_position[2])

    #     pose_msg.pose.orientation.x = float(self.q[0])
    #     pose_msg.pose.orientation.y = float(self.q[1])
    #     pose_msg.pose.orientation.z = float(self.q[2])
    #     pose_msg.pose.orientation.w = float(self.q[3])

    #     self.imu_position_pub.publish(pose_msg)


    def predict_state(self, dt):
        # === Извлечение состояния ===
        pos = self.state[0:3]          # [x, y, z]
        vel = self.state[3:6]          # [vx, vy, vz]
        q   = np.array(self.q)         # [x, y, z, w] кватернион
        omega = self.state[9:12]       # [wx, wy, wz] угловые скорости
        m = self.mass
        J = self.inertia
        g = np.array([0, 0, -9.81])

        # === Динамика моторов ===
        omega_cmd = self.motor_inputs * self.max_speed
        self.motor_speeds += (omega_cmd - self.motor_speeds) * (dt / self.motor_tau)
        omega_squared = np.clip(self.motor_speeds, 0, self.max_speed)**2
        thrusts = self.k_thrust * omega_squared

        # === Общая сила и момент в теле ===
        total_thrust_body = np.array([0.0, 0.0, np.sum(thrusts)])
        tau = np.zeros(3)
        tau[0] = self.arm_length * (thrusts[1] - thrusts[3])  # roll
        tau[1] = self.arm_length * (thrusts[2] - thrusts[0])  # pitch
        tau[2] = self.k_torque * (thrusts[0] - thrusts[1] + thrusts[2] - thrusts[3])  # yaw

        # === Преобразование силы в мировую систему координат ===
        R_world_from_body = quat_to_matrix(q)
        total_thrust_world = R_world_from_body @ total_thrust_body

        # === Линейное ускорение в мировой системе ===
        acc = g + (1.0 / m) * total_thrust_world - self.drag * vel

        # === Угловое ускорение в теле ===
        omega_dot = np.linalg.inv(J) @ (tau - np.cross(omega, J @ omega))

        # === Интеграция кватерниона ===
        q_dot = self.angular_velocity_to_quat_derivative(q, omega)
        q_new = self.normalize_quat(q + q_dot * dt)

        # === Обновление состояния ===
        pos_new = pos + vel * dt
        vel_new = vel + acc * dt
        omega_new = omega + omega_dot * dt

        self.state[0:3] = pos_new
        self.state[3:6] = vel_new
        self.q = q_new
        self.state[9:12] = omega_new

        # === Подготовка для фильтра Калмана ===
        # Прогнозируемое состояние (предсказание)
        predicted_state = np.concatenate([pos_new, vel_new, q_new, omega_new])

        # === Матрицы Калмана ===
        F = np.eye(12)  # Матрица перехода состояния (12x12)
        F[0:3, 3:6] = np.eye(3) * dt  # Переход от позиции к скорости
        F[6:9, 9:12] = np.eye(3) * dt  # Переход от угловой скорости к угловым ускорениям

        # Ковариация шума процесса (можно подстроить под ситуацию)
        Q = np.eye(12) * 0.001  # Шум процесса (регулируется)

        # Ковариационная матрица для предсказания состояния
        P_predicted = F @ self.P @ F.T + Q

        # === Обновление состояния с использованием наблюдений (например, IMU или GPS) ===
        # Мы будем обновлять состояние на основе полученных наблюдений (например, из IMU, GPS или других сенсоров).
        # Зависит от того, какие именно данные у тебя есть.

        # Пример обновления для позиции и ориентации:
        z_pos = self.vehicleLocalPosition_position  # Пример измерения позиции
        z_q = self.vehicleAttitude_q  # Пример измерения ориентации

        # Матрица наблюдений для позиции (позиция — это 3 значения: x, y, z)
        H_pos = np.zeros((3, 12))
        H_pos[0:3, 0:3] = np.eye(3)

        # Матрица наблюдений для ориентации (кватернионы — это 4 значения)
        H_q = np.zeros((4, 12))
        H_q[0:3, 6:9] = np.eye(3)

        # Ковариации наблюдений (позиции и ориентации)
        R_pos = np.eye(3) * 0.05  # Коэффициент ковариации для позиции
        R_q = np.eye(4) * 0.01  # Коэффициент ковариации для ориентации

        # Разница между измерениями и предсказанием (ошибки)
        y_pos = z_pos - pos_new
        y_q = quat_diff(q_new, z_q)

        # Калмановский коэффициент для позиции
        S_pos = H_pos @ P_predicted @ H_pos.T + R_pos
        K_pos = P_predicted @ H_pos.T @ np.linalg.inv(S_pos)
        self.state += K_pos @ y_pos
        self.P = (np.eye(12) - K_pos @ H_pos) @ P_predicted

        # Калмановский коэффициент для ориентации
        S_q = H_q @ P_predicted @ H_q.T + R_q
        K_q = P_predicted @ H_q.T @ np.linalg.inv(S_q)
        self.state += K_q @ y_q
        self.P = (np.eye(12) - K_q @ H_q) @ P_predicted

        # Публикация обновленного состояния
        self.publish_ekf_pose()



    def kalman_filter_update(self, pos_new, vel_new, q_new, omega_new):
        # Применение фильтра Калмана для позиции и ориентации
        # Измерения: позиция и ориентация
        z = np.concatenate([pos_new, q_new])

        # Прогнозируемое состояние
        x_pred = self.kalman_state  # Прогнозируемое состояние
        P_pred = self.P + self.Q  # Прогнозируемая ковариация

        # Измерения (позиция и ориентация)
        y = z - self.H @ x_pred  # Отклонение от измерений

        # Ковариация измерений
        S = self.H @ P_pred @ self.H.T + self.R

        # Калмановский коэффициент
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Обновление состояния и ковариации
        self.kalman_state = x_pred + K @ y
        self.P = P_pred - K @ self.H @ P_pred

    # def simulate_trajectory(self, current_state, q, motor_inputs_seq, dt, horizon):
    #     trajectory = []
    #     state = np.copy(current_state)
    #     orientation_q = np.copy(q)

    #     for i in range(horizon):
    #         self.state = state
    #         self.q = orientation_q
    #         self.motor_inputs = motor_inputs_seq[i]  # последовательность управляющих воздействий
    #         self.predict_state(dt)
    #         trajectory.append((self.state.copy(), self.q.copy()))
    #         state = self.state.copy()
    #         orientation_q = self.q.copy()
    #     return trajectory

    # def run_mpc_controller(self, target_pos, target_q, horizon=20, dt=0.05):
    #     def cost(u_flat):
    #         u = u_flat.reshape((horizon, 4))  # управление на горизонте: (N, 4 мотора)
    #         total_cost = 0.0

    #         # Сохраним начальное состояние
    #         original_state = self.state.copy()
    #         original_q = self.q.copy()

    #         for t in range(horizon):
    #             self.motor_inputs = np.clip(u[t], 0.0, 1.0)
    #             self.predict_state(dt)

    #             pos = self.state[0:3]
    #             q = self.q

    #             # Ошибка позиции
    #             pos_err = np.linalg.norm(pos - target_pos)

    #             # Ошибка ориентации (угол между кватернионами)
    #             dot_product = np.dot(q, target_q)
    #             dot_product = np.clip(dot_product, -1.0, 1.0)
    #             angle_err = 2 * np.arccos(np.abs(dot_product))

    #             total_cost += pos_err**2 + angle_err**2

    #         # Восстанавливаем состояние
    #         self.state = original_state
    #         self.q = original_q

    #         return total_cost

    #     # Начальное приближение: hover inputs
    #     u0 = np.ones((horizon, 4)) * 0.5
    #     bounds = [(0.0, 1.0)] * (horizon * 4)

    #     res = minimize(cost, u0.flatten(), bounds=bounds, method='L-BFGS-B')

    #     if res.success:
    #         u_opt = res.x.reshape((horizon, 4))
    #         self.motor_inputs = u_opt[0]  # применяем только первое управляющее воздействие
    #     else:
    #         print("MPC optimization failed:", res.message)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

