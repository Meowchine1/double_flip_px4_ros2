import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import VehicleAttitude, VehicleImu, ActuatorOutputs, VehicleLocalPosition,SensorCombined,VehicleAngularVelocity,VehicleAngularAccelerationSetpoint
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from scipy.spatial.transform import Rotation as R

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

        # ======= SUBSCRIBERS =======
        self.create_subscription(SensorCombined, '/fmu/out/sensor_combined', self.sensor_combined_callback, qos_profile)
        self.create_subscription(VehicleAngularVelocity, '/fmu/out/vehicle_angular_velocity', self.angular_velocity_callback, qos_profile)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.vehicle_attitude_callback, qos_profile)
        self.create_subscription(VehicleAngularAccelerationSetpoint,
        '/fmu/out/vehicle_angular_acceleration_setpoint', self.vehicle_angular_acceleration_setpoint_callback, qos_profile)
        self.create_subscription(VehicleImu,'/fmu/in/vehicle_imu',self.vehicle_imu_callback, qos_profile)
        self.create_subscription(ActuatorOutputs, '/fmu/out/actuator_outputs', self.actuator_outputs_callback, qos_profile)

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
        
        # ======= DRONE LINEAR PARAMETERS ======= 
        # :[TOPIC NAME]_[PARAM NAME] OR [TOPIC NAME] IF PARAM = TOPIC NAME
        self.vehicleImu_velocity = np.zeros(3) # Текущая линейная скорость (в мировых координатах)
        self.vehicleImu_linear_acceleration = np.zeros(3, dtype=np.float32)
        self.sensorCombined_linear_acceleration = np.zeros(3, dtype=np.float32)
        self.position = [0.0, 0.0, 0.0] # drone position estimates with IMU localization
        self.motor_inputs = np.zeros(4)  # нормализованные входы [0..1] или RPM
         
        # ======= DRONE ANGULAR PARAMETERS =======  
        # :[TOPIC NAME]_[PARAM NAME] OR [TOPIC NAME] IF PARAM = TOPIC NAME
        # QUATERNIONS
        self.sensorCombined_q = [0.0, 0.0, 0.0, 1.0] #  float32 
        self.vehicleAttitude_q = [0.0, 0.0, 0.0, 1.0]
        self.orientation_q = [0.0, 0.0, 0.0, 1.0] # actual orientation
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

        # ======= TIMERS =======
        self.timer = self.create_timer(0.01, self.step_dynamics)

    def actuator_outputs_callback(self, msg):
        pwm_outputs = msg.output[:4]  # предполагаем, что 0-3 — это моторы
        # Преобразование PWM в радианы в секунду (линейное приближение)
        self.motor_inputs = np.clip((np.array(pwm_outputs) - 1000.0) / 1000.0 * self.max_speed, 0.0, self.max_speed)

    def sensor_combined_callback(self, msg):
        # === ГИРОСКОП ===
        dt_gyro = msg.gyro_integral_dt * 1e-6  # микросекунды -> секунды
        gyro_rad = np.array(msg.gyro_rad, dtype=np.float32)  # угловая скорость (рад/с)
        self.sensorCombined_angular_velocity = gyro_rad
         
        delta_angle = gyro_rad * dt_gyro # Угловое приращение (рад)
        self.sensorCombined_delta_angle = delta_angle

        # Обновляем ориентацию на основе приращения
        dq = R.from_rotvec(delta_angle)  # Кватернион из углового приращения
        q_prev = R.from_quat(self.q)  # Предыдущий кватернион ориентации
        self.q = (q_prev * dq).as_quat()  # Обновлённый кватернион

        # === АКСЕЛЕРОМЕТР ===
        # average value acceleration in m/s^2 over the last accelerometer sampling period
        self.sensorCombined_linear_acceleration = np.array(msg.accelerometer_m_s2, dtype=np.float32)
         
    def angular_velocity_callback(self, msg):
        self.angularVelocity = np.array(msg.xyz, dtype=np.float32)
        self.angularVelocity_angular_acceleration = np.array(msg.xyz_derivative, dtype=np.float32)

    def vehicle_attitude_callback(self, msg):
        self.vehicleAttitude_q = np.array(msg.q, dtype=np.float32)
        
    def vehicle_angular_acceleration_setpoint_callback(self, msg):
        self.vehiclAngularAcceleration_angular_acceleration = msg.xyz

    def vehicle_imu_callback(self, msg):
        # Сохраняем приращения
        self.vehicleImu_delta_angle = np.array(msg.delta_angle, dtype=np.float32)         # рад
        self.vehicleImu_delta_velocity = np.array(msg.delta_velocity, dtype=np.float32)   # м/с
        delta_angle_dt = msg.delta_angle_dt * 1e-6     # с
        delta_velocity_dt = msg.delta_velocity_dt * 1e-6  # с

        # Вычисляем угловую скорость и ускорение
        self.vehicleImu_angular_velocity = self.vehicleImu_delta_angle / delta_angle_dt
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

    def step_dynamics(self):
        dt = 0.01
        vel = self.state[3:6]
        ang_vel = self.state[9:12]

        # Сила тяги в теле
        thrusts = self.k_thrust * self.motor_inputs ** 2
        total_thrust_body = np.array([0, 0, np.sum(thrusts)])

        # Переход в мировую СК
        R_body_to_world = R.from_quat(self.orientation_q).as_matrix()
        total_thrust_world = R_body_to_world @ total_thrust_body

        gravity = np.array([0, 0, -self.mass * 9.81])
        drag_force = -self.drag * vel
        net_force = total_thrust_world + drag_force + gravity
        acc = net_force / self.mass

        tau_x = self.arm_length * (thrusts[1] - thrusts[3])
        tau_y = self.arm_length * (thrusts[2] - thrusts[0])
        tau_z = self.k_torque * (thrusts[0] - thrusts[1] + thrusts[2] - thrusts[3])
        torques = np.array([tau_x, tau_y, tau_z])
        ang_acc = np.linalg.inv(self.inertia) @ (torques - np.cross(ang_vel, self.inertia @ ang_vel))

        self.state[0:3] += vel * dt
        self.state[3:6] += acc * dt
        self.state[9:12] += ang_acc * dt
        self.state[9:12] = np.clip(self.state[9:12], -self.max_rate, self.max_rate)

        self.publish_pose()
        self.publish_motor_inputs()
        

    def publish_pose(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = self.state[0]
        msg.pose.position.y = self.state[1]
        msg.pose.position.z = self.state[2]
        msg.pose.orientation.x = self.orientation_q[0]
        msg.pose.orientation.y = self.orientation_q[1]
        msg.pose.orientation.z = self.orientation_q[2]
        msg.pose.orientation.w = self.orientation_q[3]
        self.pose_pub.publish(msg)

    def publish_motor_inputs(self):
        msg = Float32MultiArray()
        msg.data = self.motor_inputs.tolist()
        self.motor_pub.publish(msg)

    
    def quat_normalize(q):
        return q / np.linalg.norm(q)

    def quat_to_rotmat(q):
        return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

    def angular_velocity_to_quat_derivative(q, omega):
        # q = [w, x, y, z], omega = [wx, wy, wz]
        wx, wy, wz = omega
        Omega = 0.5 * np.array([
            [0.0, -wx, -wy, -wz],
            [wx,  0.0,  wz, -wy],
            [wy, -wz,  0.0, wx],
            [wz,  wy, -wx, 0.0]
        ])
        return Omega @ q

    def predict_state(self, dt):
        # === Extract state ===
        pos = self.state[0:3]          # [x, y, z]
        vel = self.state[3:6]          # [vx, vy, vz]
        q   = np.array(self.q)         # [x, y, z, w] ROS-style
        omega = self.state[9:12]       # [wx, wy, wz] in body frame
        m = self.mass
        J = self.inertia
        g = np.array([0, 0, -9.81])

        # === Motor dynamics (first-order filter) ===
        omega_cmd = self.motor_inputs * self.max_speed
        self.motor_speeds += (omega_cmd - self.motor_speeds) * (dt / self.motor_tau)
        omega_squared = np.clip(self.motor_speeds, 0, self.max_speed)**2
        thrusts = self.k_thrust * omega_squared

        # === Total force and torque in body frame ===
        total_thrust_body = np.array([0.0, 0.0, np.sum(thrusts)])
        tau = np.zeros(3)
        tau[0] = self.arm_length * (thrusts[1] - thrusts[3])  # roll
        tau[1] = self.arm_length * (thrusts[2] - thrusts[0])  # pitch
        tau[2] = self.k_torque * (thrusts[0] - thrusts[1] + thrusts[2] - thrusts[3])  # yaw

        # === Transform thrust to world frame ===
        R_world_from_body = self.quat_to_rotmat(q)
        total_thrust_world = R_world_from_body @ total_thrust_body

        # === Linear acceleration in world frame ===
        acc = g + (1.0 / m) * total_thrust_world - self.drag * vel

        # === Angular acceleration in body frame ===
        omega_dot = np.linalg.inv(J) @ (tau - np.cross(omega, J @ omega))

        # === Quaternion integration ===
        q_dot = self.angular_velocity_to_quat_derivative(q, omega)
        q_new = self.quat_normalize(q + q_dot * dt)

        # === State update ===
        pos_new = pos + vel * dt
        vel_new = vel + acc * dt
        omega_new = omega + omega_dot * dt

        self.state[0:3] = pos_new
        self.state[3:6] = vel_new
        self.q = q_new
        self.state[9:12] = omega_new
        self.orientation_q = self.q  # для ROS/визуализации


def main(args=None):
    rclpy.init(args=args)
    node = DynamicModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
