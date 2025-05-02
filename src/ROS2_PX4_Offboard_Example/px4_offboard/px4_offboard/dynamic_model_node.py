import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3, Quaternion
from visualization_msgs.msg import Marker, Point
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
from datetime import datetime
from sensor_msgs.msg import Imu, MagneticField, FluidPressure
import os
import casadi as ca

import matplotlib.pyplot as plt
from std_msgs.msg import String

import jax
import jax.numpy as jnp

# ======= CONSTANTS =======
SEA_LEVEL_PRESSURE = 101325.0
EKF_DT = 0.01
# ======= DRONE CONSTRUCT PARAMETERS =======
MASS = 0.82
INERTIA = np.diag([0.045, 0.045, 0.045])
ARM_LEN = 0.15
K_THRUST = 1.48e-6
K_TORQUE = 9.4e-8
MOTOR_TAU = 0.02
MAX_SPEED = 2100.0
DRAG = 0.1
MAX_RATE = 25.0  # рад/с, ограничение на угловую скорость (roll/pitch)

def plot_trajectory(x_traj_opt, x_goal=None, title="Optimized Trajectory"):
    x = x_traj_opt[:, 0]
    y = x_traj_opt[:, 1]
    z = x_traj_opt[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='Predicted trajectory', color='blue')
    ax.scatter(x[0], y[0], z[0], color='green', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='red', label='End')

    if x_goal is not None:
        ax.scatter(x_goal[0], x_goal[1], x_goal[2], color='orange', label='Goal', s=100, marker='X')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.show()

def publish_trajectory_marker(pub, x_traj_opt, frame_id="map"):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = node.get_clock().now().to_msg()
    marker.ns = "mpc_trajectory"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.02
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 0.5
    marker.color.b = 1.0

    for state in x_traj_opt:
        p = Point()
        p.x, p.y, p.z = state[:3]
        marker.points.append(p)

    pub.publish(marker)

    # self.traj_pub = self.create_publisher(Marker, "/mpc/trajectory", 10)
    #publish_trajectory_marker(self.traj_pub, x_traj_opt)

# def __init__(self):
#         super().__init__('mpc_controller')
        
#         self.traj_pub = self.create_publisher(Marker, "/mpc/trajectory", 10)

#         # Остальные инициализации...

#     def publish_trajectory_marker(self, x_traj_opt, frame_id="map"):
#         marker = Marker()
#         marker.header.frame_id = frame_id
#         marker.header.stamp = self.get_clock().now().to_msg()
#         marker.ns = "mpc_trajectory"
#         marker.id = 0
#         marker.type = Marker.LINE_STRIP
#         marker.action = Marker.ADD
#         marker.scale.x = 0.02
#         marker.color.a = 1.0
#         marker.color.r = 0.0
#         marker.color.g = 0.7
#         marker.color.b = 1.0

#         for state in x_traj_opt:
#             p = Point()
#             p.x, p.y, p.z = state[:3]
#             marker.points.append(p)

#         self.traj_pub.publish(marker)

# ===== MATRIX OPERTIONS =====
# QUATERNION UTILS (SCIPY-based)
def quat_multiply(q1, q2):
    """Multiply two quaternions [x, y, z, w] using scipy."""
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    return (r1 * r2).as_quat()

class MyEKF(ExtendedKalmanFilter):
    def __init__(self, dim_x, dim_z):
        self.get_logger().info(f"MyEKF")
        super().__init__(dim_x, dim_z)
        self.dt = EKF_DT

    def predict_x(self, u=np.zeros(4)):
        # Custom fx(x, u, dt) function
        return self.fx(self.x, u, self.dt)
    #predict new state with dynamic physic model
    def fx(self,x, u, dt):
        """
        Нелинейная динамическая модель квадрокоптера
        dt - шаг времени
        x - вектор состояния [позиция, скорость, кватернион, угловая скорость]
        u - управление (обороты моторов) [w1, w2, w3, w4] 
        """
        # параметры
        m = MASS
        I = INERTIA
        arm = ARM_LEN
        kf = K_THRUST
        km = K_TORQUE
        drag = DRAG
        g = np.array([0, 0, 9.81])
        max_rate = MAX_RATE
        max_speed = MAX_SPEED

        pos = x[0:3]
        vel = x[3:6]
        quat = x[6:10]
        omega = x[10:13]

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
        tau = np.array([L * (thrusts[1] - thrusts[3]),
                        L * (thrusts[2] - thrusts[0]),
                        km * (w_squared[0] - w_squared[1] + w_squared[2] - w_squared[3]) ])
        # Обновление угловой скорости (Жуковский)
        omega_dot = np.linalg.inv(I) @ (tau - np.cross(omega, I @ omega))
        new_omega = omega + omega_dot * dt

        # Обновление ориентации через кватернион
        omega_quat = np.concatenate(([0.0], omega))
        dq = 0.5 * quat_multiply(quat, omega_quat)
        new_quat = quat + dq * dt
        norm = np.linalg.norm(new_quat)
        if norm > 1e-6:
            new_quat /= norm
        else:
            new_quat = np.array([0, 0, 0, 1]) # fallback

        # Новое состояние
        x_next = np.zeros(13)
        x_next[0:3] = new_pos
        x_next[3:6] = new_vel
        x_next[6:10] = new_quat
        x_next[10:13] = new_omega
        return x_next

class DynamicModelNode(Node):
    def __init__(self):
        super().__init__('dynamic_model_node')
        
        self.get_logger().info(f"DynamicModelNode")

        # == == == =CLIENT SERVER INTERACTION= == == ==
        self.pub_optimized_traj = self.create_publisher(Vector3, '/drone/optimized_traj', qos_profile)
        self.pub_to_client = self.create_publisher(Vector3, '/drone/mpc_data', qos_profile)
        self.create_subscription(String, '/drone/client_data', self.client_data_callback, qos_profile)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        # == == == =PUBLISHERS= == == ==
 
        # Паблишеры для ekf_filter_node
        self.imu_pub = self.create_publisher(Imu, '/imu/data', qos_profile)
        self.mag_pub = self.create_publisher(MagneticField, '/imu/mag', qos_profile)
        self.baro_pub = self.create_publisher(FluidPressure, '/baro', qos_profile)
        self.ekf_state_pub = self.create_publisher(Float32MultiArray, '/ekf/state', qos_profile)

        #trajectory visualisation
        self.traj_pub = self.create_publisher(Marker, "/mpc/trajectory", qos_profile)

        # == == == =SUBSCRIBERS= == = ==
        self.create_subscription(SensorCombined, '/fmu/out/sensor_combined', self.sensor_combined_callback, qos_profile)
        self.create_subscription(VehicleAngularVelocity, '/fmu/out/vehicle_angular_velocity', self.angular_velocity_callback, qos_profile)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.vehicle_attitude_callback, qos_profile)
        self.create_subscription(VehicleAngularAccelerationSetpoint,
        '/fmu/out/vehicle_angular_acceleration_setpoint', self.vehicle_angular_acceleration_setpoint_callback, qos_profile)
        self.create_subscription(VehicleImu,'/fmu/out/vehicle_imu',self.vehicle_imu_callback, qos_profile)
        self.create_subscription(ActuatorOutputs, '/fmu/out/actuator_outputs', self.actuator_outputs_callback, qos_profile)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.create_subscription(SensorBaro, '/fmu/out/sensor_baro', self.sensor_baro_callback, qos_profile)
        self.create_subscription(VehicleMagnetometer, '/fmu/out/vehicle_magnetometer', self.vehicle_magnetometer_callback, qos_profile)
        
        # ekf_filter_node data
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, qos_profile)

        # == == == =DATA USED IN METHODS= == == == 
        self.angularVelocity = np.zeros(3, dtype=np.float32)
        self.angular_acceleration = np.zeros(3, dtype=np.float32)
        self.vehicleImu_velocity_w = np.zeros(3, dtype=np.float32) # в мировых координатах 
        self.sensorCombined_linear_acceleration = np.zeros(3, dtype=np.float32)
        self.position = np.zeros(3, dtype=np.float32) # drone position estimates with IMU localization
        self.motor_inputs = np.zeros(4, dtype=np.float32)  # нормализованные входы [0..1] или RPM
        self.vehicleAttitude_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32) # quaternion from topic
        self.magnetometer_data = np.zeros(3, dtype=np.float32)
        self.baro_attitude = 0.0
        self.baro_pressure = 0.0
        self.baro_altitude = 0.0
        self.mag_yaw = 0.0
        
        # FOR SITL TESTING  
        self.vehicleLocalPosition_position = np.zeros(3, dtype=np.float32)
        # ekf_filter_node data
        self.odom_callback_position = np.zeros(3, dtype=np.float32)
        self.odom_callback_orientation = np.zeros(4, dtype=np.float32)

        # ======= OTHER TOPIC DATA ======= 
        # :[TOPIC NAME]_[PARAM NAME] OR [TOPIC NAME] IF PARAM = TOPIC NAME
        self.sensorCombined_angular_velocity = np.zeros(3, dtype=np.float32)
        self.angularVelocity_angular_acceleration = np.zeros(3, dtype=np.float32)
        self.baro_temperature = 0.0 # temperature in degrees Celsius

        # ======= EKF =======
        # вектор состояния (13 штук: позиция, скорость, ориентация (4), угловые скорости).
        # наблюдаемые величины (14 штук) позиция, линейная скорость, mag yaw, baro pressure, 
        self.ekf = MyEKF(dim_x=13, dim_z=14)
        self.ekf.x = np.zeros(13)
        self.ekf.x[6] = 1.0  # qw = 1 (единичный кватернион)
        #Covariance matrix
        self.ekf.P *= 0.1
        #Process noise matrix
        self.ekf.Q = np.diag([
            0.001, 0.001, 0.001,   # x, y, z (позиция)
            0.01, 0.01, 0.01,      # vx, vy, vz (скорость)
            0.0001, 0.0001, 0.0001, 0.0001, # qw, qx, qy, qz (ориентация)
            0.00001, 0.00001, 0.00001, # wx, wy, wz (угловая скорость)
        ])
        #Measurement noise matrix
        self.ekf.R = np.diag([
            0.1, 0.1, 0.1,         # позиция x, y, z (м²)
            0.0001, 0.0001, 0.0001, # скорость vx, vy, vz (м²/с²)
            0.00001, 0.00001, 0.00001, 0.00001,  # ориентация qw, qx, qy, qz (кватернионы - погрешность малюсенькая)
            0.00001, 0.00001, 0.00001, # угловые скорости wx, wy, wz (рад²/с²)
            0.5                  # барометр (м²)
        ])
        #    ====    ====   Параметры ModelPredictiveController    ====     ====     ====
        self.dt = 0.1  # Шаг времени (с)
        self.horizon = 50  # Горизонт предсказания
        self.n = 13  # Размерность состояния квадрокоптера (позиция, скорость, ориентация, угловая скорость)
        self.m = 4  # Размерность управления (4 мотора)
        
        # стоимости
        # x = [x, y, z,      # позиция
        # vx, vy, vz,   # скорость
        # qw, qx, qy, qz,  # ориентация (кватернион)
        # wx, wy, wz]   # угловая скорость
        self.Q = 100.0  # важность траектории (позиция + ориентация) 
        self.R = 0.001   # разрешаем моторам быть агрессивными
        self.Qf = np.diag([
            10.0, 10.0, 10.0,      # позиция — средне важно
            0.1, 0.1, 0.1,         # скорость — не критично
            0.0, 100.0, 100.0, 0.0, # ориентация — важны qx/qy (например, pitch flip)
            10.0, 10.0, 1.0        # угловые скорости — важно, чтобы стабилизировался
        ])
        # Целевое состояние (по умолчанию всё единицы)
        self.x_goal = np.ones(self.n)   
        # Целевая траектория управления (по умолчанию нулевая)
        self.u_target_traj = np.zeros((self.horizon, self.m))
        # Инициализация ModelPredictiveController с оптимизатором
        self.mpc = ModelPredictiveController(
            dt=self.dt,
            horizon=self.horizon,
            n=self.n,
            m=self.m
        )

        self.phase = 'init'
        self.takeoff_altitude = 5.0  # м
        self.flip_started_time = None
        self.flip_duration = 1.0  # с, продолжительность флипа
        self.hover_time = 2.0  # с, стабилизация после флипа
        self.hover_start_time = None
        self.landing_altitude = 0.2  # м
        #   ====    ====    ====     ====     ====

        now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_file_name_pos = f"{now_str}_pos.log"
        self.log_file_name_quat = f"{now_str}_quat.log"
        self.log_file_name_vel = f"{now_str}_vel.log"
        self.log_file_name_ang_vel = f"{now_str}_ang_vel.log"

         # ======= TIMERS =======
        #self.timer = self.create_timer(0.01, self.step_dynamics)
        self.EKF_timer = self.create_timer(EKF_DT, self.EKF)
        self.ekf_filter_node_timer = self.create_timer(0.01, self.ekf_filter_node_t) 
        self.mpc_controller = self.create_timer(0.01, self.mpc_control_loop)
        #self.optimized_traj = self.create_timer(EKF_DT, self.send_optimized_traj_t)
        self.optimized_traj_f = False
        self.to_client = self.create_timer(EKF_DT, self.to_client_t)
        self.to_client_f = False
 
    def to_client_t(self):
        if self.to_client_f:
            msg = None
            msg[0] = self.ekf.x[2] # altitude
            self.pub_to_client.publish(msg)

    def client_data_callback(self, msg):
        """GET CLIENT MESSAGES"""
        command = msg.data.strip().lower()
        self.get_logger().info(f"Received command: {command}")

        if command == "takeoff":
            self.phase = command
            self.to_client_f = True
            self.optimized_traj_f = True 
        else:
            self.get_logger().warn(f"Unknown command: {command}") 
 
    def publish_trajectory_marker(pub, x_traj_opt, frame_id="map"):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = node.get_clock().now().to_msg()#?
        marker.ns = "mpc_trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0

        for state in x_traj_opt:
            p = Point()
            p.x, p.y, p.z = state[:3]
            marker.points.append(p)

        pub.publish(marker)
    
    def send_optimized_traj(self):
        if self.optimized_traj_f:
            msg = None
            msg[0] = self.x_goal
            msg[1] = self.u_target_traj 
            self.pub_optimized_traj.publish(msg)

    def quaternion_from_roll(self, roll_rad):
        r = R.from_euler('x', roll_rad)
        return r.as_quat()  # формат [x, y, z, w]
    
    def mpc_control_loop(self):
        if self.phase != 'init':
            current_time = self.get_clock().now().nanoseconds * 1e-9
            x0 = self.ekf.x.copy()  # [13] x, y, z, vel, q, omega
            u_init = np.tile(self.motor_inputs, (self.horizon, 1))  # [horizon, 4]
            x_target_traj = np.zeros((self.horizon, 13))  # 13 — x, y, z, vel, q, omega
            u_target_traj = np.zeros((self.horizon, 4))   # 4 — управляющие усилия

            if self.phase == 'takeoff':
                pos = x0[0:3].copy()
                pos[2] = self.takeoff_altitude
                vel = np.zeros(3)
                q = np.array([0.0, 0.0, 0.0, 1.0])  # Нулевая ориентация
                omega = np.zeros(3)
                x_target_traj[i] = np.concatenate([pos, vel, q, omega])
                u_target_traj[i] = self.hover_thrust.copy()

                if self.ekf.x[2] + 0.5 >= self.takeoff_altitude:
                    self.phase = 'flip'
                    self.flip_started_time = current_time

            elif self.phase == 'flip':
                # Фаза флипа вокруг оси X
                t_local = current_time - self.flip_started_time

                # Оценка текущего угла roll из кватерниона EKF
                q_current = self.ekf.x[6:10]  # qx, qy, qz, qw
                roll_current, _, _ = self.euler_from_quaternion(q_current)

                # Целевой угол на текущий момент времени (плановая траектория)
                roll_time_based = 2 * np.pi * (t_local / self.flip_duration)

                # Добавим корректировку на недолет или перелет
                roll_error = roll_time_based - roll_current
                gain = 0.5  # коэффициент обратной связи — можно подстроить под модель
                roll_target = roll_current + gain * roll_error


                # Уточнённый угол поворота
                angle_i = roll_target * (i / self.horizon)

                pos = x0[0:3]
                vel = np.zeros(3)
                q = self.quaternion_from_roll(angle_i)
                omega = np.array([2 * np.pi / self.flip_duration, 0.0, 0.0])

                x_target_traj[i] = np.concatenate([pos, vel, q, omega])
                u_target_traj[i] = self.hover_thrust.copy()

                # Завершаем флип по углу или по времени
                if abs(roll_current) >= 2 * np.pi or t_local >= self.flip_duration:
                    self.phase = 'hover'
                    self.hover_start_time = current_time

            elif self.phase == 'hover':
                # Стабилизация после флипа
                for i in range(self.horizon):
                    pos = x0[0:3].copy()
                    pos[2] = self.takeoff_altitude
                    vel = np.zeros(3)
                    q = np.array([0.0, 0.0, 0.0, 1.0])  # Нулевая ориентация
                    omega = np.zeros(3)
                    x_target_traj[i] = np.concatenate([pos, vel, q, omega])
                    u_target_traj[i] = self.hover_thrust.copy()

                if current_time - self.hover_start_time >= self.hover_time:
                    self.phase = 'land'

            elif self.phase == 'land':
                self.to_client_f = False
                self.optimized_traj_f = False

            # Вызов MPC
            x_traj_opt, u_mpc = self.mpc.step(
                t=current_time,
                x0=x0,
                u_init=u_init,
                x_goal=x_target_traj[-1],
                u_target_traj=u_target_traj,
            )

            self.send_optimized_traj_t(x_traj_opt, u_mpc)

            # visualize trajectory
            publish_trajectory_marker(self.traj_pub, x_traj_opt)

    def ekf_filter_node_t(self):
        self.get_logger().info("ekf_filter_node_t")
        imu_msg = Imu()
        mag_msg = MagneticField()
        baro_msg = FluidPressure()
        # Стамп времени для сообщений
        current_time = self.get_clock().now().to_msg()
        # ======== /imu/data ========
        imu_msg.header.stamp = current_time
        imu_msg.header.frame_id = "base_link"  # или другой фрейм, в зависимости от системы координат
        # Ориентация в виде кватерниона
        imu_msg.orientation = Quaternion(x=self.vehicleAttitude_q[0],
                                        y=self.vehicleAttitude_q[1],
                                        z=self.vehicleAttitude_q[2],
                                        w=self.vehicleAttitude_q[3])
        # Угловая скорость (из поля angularVelocity)
        imu_msg.angular_velocity = Vector3(x=self.angularVelocity[0],
                                        y=self.angularVelocity[1],
                                        z=self.angularVelocity[2])
        # Линейное ускорение (из поля sensorCombined_linear_acceleration)
        imu_msg.linear_acceleration = Vector3(x=self.sensorCombined_linear_acceleration[0],
                                            y=self.sensorCombined_linear_acceleration[1],
                                            z=self.sensorCombined_linear_acceleration[2])
        self.imu_pub.publish(imu_msg)
        # ======== /imu/mag ========
        mag_msg.header.stamp = current_time
        mag_msg.header.frame_id = "base_link"  # или другой фрейм
        # данные с магнитометра
        mag_msg.magnetic_field = Vector3(x=self.magnetometer_data[0],
                                        y=self.magnetometer_data[1],
                                        z=self.magnetometer_data[2])
        self.mag_pub.publish(mag_msg)
        # ======== /baro ========
        baro_msg.header.stamp = current_time
        baro_msg.header.frame_id = "base_link"  # или другой фрейм

        # Давление/высота
        baro_msg.fluid_pressure = self.baro_pressure  # если это давление в Па или высота, нужно уточнить
        self.baro_pub.publish(baro_msg)

    def publish_ekf_state(self):
        # Создание сообщения
        msg = Float32MultiArray()
        msg.data = self.ekf.x.tolist()  # Преобразуем numpy массив в список для публикации

        # Публикуем сообщение
        self.ekf_state_pub.publish(msg)
        self.get_logger().info("Published EKF state")

    def logger(self):
        self.compare_and_log()

    def compare_and_log(self):
        self.get_logger().info("compare_and_log")
        
        # Данные из EKF, одометра, и сенсоров
        pos_my_kf = self.ekf.x[0:3]
        pos_odom = self.odom_callback_position
        pos_real = self.vehicleLocalPosition_position

        quat_my_ekf = self.ekf.x[6:10]
        px4_quat = self.vehicleAttitude_q
        quat_odom = self.odom_callback_orientation

        vel_my_ekf = self.ekf.x[3:6]
        integral_vel = self.vehicleImu_velocity_w
        
        omega_my_ekf = self.ekf.x[10:13]
        omega_from_sensor = self.angularVelocity

        # Файлы для записи
        file_paths = {
            'pos': self.log_file_name_pos,
            'quat': self.log_file_name_quat,
            'vel': self.log_file_name_vel,
            'ang_vel': self.log_file_name_ang_vel
        }
        
        # Сохранение позиции
        self._write_to_file(file_paths['pos'], pos_my_kf, pos_odom, pos_real)
        
        # Сохранение ориентации (кватернионы)
        self._write_to_file(file_paths['quat'], quat_my_ekf, px4_quat, quat_odom)
        
        # Сохранение скорости
        self._write_to_file(file_paths['vel'], vel_my_ekf, integral_vel)
        
        # Сохранение угловой скорости
        self._write_to_file(file_paths['ang_vel'], omega_my_ekf, omega_from_sensor)

    def _write_to_file(self, file_path, *data):
        """
        Метод для записи данных в файл. Если файл существует, дописывает данные.
        Если файла нет, создаёт новый.
        Записывает данные в одну строку, разделяя их табуляцией и добавляя новую строку в конце.
        """
        # Проверяем, существует ли файл
        file_exists = os.path.exists(file_path)
        
        # Открываем файл для записи в режиме добавления
        with open(file_path, 'a') as file:
            # Преобразуем данные в строку, разделяя табуляцией
            data_line = '\t'.join(map(str, data)) + '\n'
            
            # Если файл не существует, можно добавить заголовок
            if not file_exists:
                file.write("pos_my_kf\tpos_odom\tpos_real\t|\tquat_my_ekf\tpx4_quat\tquat_odom\t|\tvel_my_ekf\tintegral_vel\t|\tomega_my_ekf\tomega_from_sensor\n")
            
            # Записываем данные
            file.write(data_line)

    def odom_callback(self, msg: Odometry):
        self.get_logger().info("odom_callback")
        self.odom_callback_position = msg.pose.pose.position
        self.odom_callback_orientation = msg.pose.pose.orientation
        
    def sensor_baro_callback(self, msg):
        self.get_logger().info("sensor_baro_callback")
        self.baro_temperature = msg.baro_temperature
        self.baro_pressure = msg.baro_pressure
        self.baro_attitude = 44330.0 * (1.0 - (msg.baro_pressure / SEA_LEVEL_PRESSURE) ** 0.1903)

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

    def vehicle_magnetometer_callback(self, msg: VehicleMagnetometer):
        # Измерения магнитометра
        self.get_logger().info("vehicle_magnetometer_callback")
        self.magnetometer_data = np.array(msg.magnetometer_ga, dtype=np.float32)
        self.mag_yaw = self.get_yaw_from_mag()

    # ПОЗИЦИЯ ДЛЯ ОЦЕНКИ ИНЕРЦИАЛНОЙ ЛОКАЛИЗАЦИИ
    def vehicle_local_position_callback(self, msg: VehicleLocalPosition):
        self.get_logger().info(f"vehicle_local_position_callback {msg.x} {msg.y} {msg.z}")
        self.vehicleLocalPosition_position[0] = msg.x
        self.vehicleLocalPosition_position[1] = msg.y
        self.vehicleLocalPosition_position[2] = msg.z

    def actuator_outputs_callback(self, msg: ActuatorOutputs):
        pwm_outputs = msg.output[:4]  # предполагаем, что 0-3 — это моторы
        # Преобразование PWM в радианы в секунду (линейное приближение)
        self.motor_inputs = np.clip((np.array(pwm_outputs) - 1000.0) / 1000.0 * MAX_SPEED, 0.0, MAX_SPEED)
        self.get_logger().info("actuator_outputs_callback")

    # ЛИНЕЙНОЕ УСКОРЕНИЕ, УГЛОВОЕ УСКОРЕНИЕ, КВАТЕРНИОН
    def sensor_combined_callback(self, msg: SensorCombined):
        dt_gyro = msg.gyro_integral_dt * 1e-6  # микросекунды -> секунды
        gyro_rad = np.array(msg.gyro_rad, dtype=np.float32)  # угловая скорость (рад/с)
        self.sensorCombined_angular_velocity = gyro_rad
         
        delta_angle = gyro_rad * dt_gyro # Угловое приращение (рад)
        self.sensorCombined_delta_angle = delta_angle
        self.sensorCombined_linear_acceleration = np.array(msg.accelerometer_m_s2, dtype=np.float32)
         
    def angular_velocity_callback(self, msg: VehicleAngularVelocity):
        self.angularVelocity = np.array(msg.xyz, dtype=np.float32)
        self.angularVelocity_angular_acceleration = np.array(msg.xyz_derivative, dtype=np.float32)

    def vehicle_attitude_callback(self, msg: VehicleAttitude):
        # In this system we use scipy format for quaternion. 
        # PX4 topic uses the Hamilton convention, and the order is q(w, x, y, z). So we reorder it
        self.vehicleAttitude_q = np.array([msg.q[1], msg.q[2], msg.q[3], msg.q[0]], dtype=np.float32)
        
    def vehicle_angular_acceleration_setpoint_callback(self, msg: VehicleAngularAccelerationSetpoint):
        self.angular_acceleration = msg.xyz

    def vehicle_imu_callback(self, msg: VehicleImu):
        delta_velocity = np.array(msg.delta_velocity, dtype=np.float32)  # м/с
        delta_velocity_dt = msg.delta_velocity_dt * 1e-6  # с
        # Проверяем наличие ориентации и валидного времени интеграции
        if delta_velocity_dt > 0.0:
            rotation = R.from_quat(self.vehicleAttitude_q)
            delta_velocity_world = rotation.apply(delta_velocity)
            gravity = np.array([0.0, 0.0, 9.80665], dtype=np.float32)
            delta_velocity_world += gravity * delta_velocity_dt
            self.vehicleImu_velocity_w += delta_velocity_world
            self.position += self.vehicleImu_velocity_w * delta_velocity_dt

    def publish_motor_inputs(self):
        msg = Float32MultiArray()
        msg.data = self.motor_inputs.tolist()
        self.motor_pub.publish(msg)

    def EKF(self):
        """ Основная функция обновления фильтра Калмана. """
        vel_world = self.vehicleImu_velocity_w
        z = np.array([
            self.position[0],  # x
            self.position[1],  # y
            self.position[2],  # z
            vel_world[0],  # vx
            vel_world[1],  # vy
            vel_world[2],  # vz
            self.vehicleAttitude_q[0],
            self.vehicleAttitude_q[1],
            self.vehicleAttitude_q[2],
            self.vehicleAttitude_q[3],
            self.angularVelocity[0],
            self.angularVelocity[1],
            self.angularVelocity[2],
            self.baro_altitude  # высота по барометру
        ])

        # Вызываем предсказание и обновление фильтра
        self.ekf.predict_update(z, HJacobian=self.HJacobian, Hx=self.hx, u=self.motor_inputs)

        self.logger()

    def hx(self, x):
        """ Модель измерений: что бы показали датчики при текущем состоянии. """
        return np.array([
        x[0],  # x (позиция)
        x[1],  # y (позиция)
        x[2],  # z (позиция)
        x[3],  # vx (скорость)
        x[4],  # vy (скорость)
        x[5],  # vz (скорость)
        x[6],  # qw (ориентация)
        x[7],  # qx (ориентация)
        x[8],  # qy (ориентация)
        x[9],  # qz (ориентация)
        x[10], # wx (угловая скорость)
        x[11], # wy (угловая скорость)
        x[12], # wz (угловая скорость)
        x[2],  # барометр (ещё раз высота z)
    ])

    def HJacobian(self, x):
        """ Якобиан модели измерений по состоянию. """
        H = np.zeros((14, 13))  # 14 измерений на 13 состояний
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        H[2, 2] = 1.0  # z
        H[3, 3] = 1.0  # vx
        H[4, 4] = 1.0  # vy
        H[5, 5] = 1.0  # vz
        H[6, 6] = 1.0  # qw
        H[7, 7] = 1.0  # qx
        H[8, 8] = 1.0  # qy
        H[9, 9] = 1.0  # qz
        H[10, 10] = 1.0  # wx
        H[11, 11] = 1.0  # wy
        H[12, 12] = 1.0  # wz
        H[13, 2] = 1.0   # барометрический z
        return H



# Функция для предсказания успешности флипа
def predict_flip_success(x_init, u_init, duration=2.0):
    """
    Предсказание успешности флипа.
    
    x_init: Начальное состояние (позиция, скорость, ориентация, угловая скорость).
    u_init: Управление (обороты моторов).
    duration: Длительность флипа в секундах.
    
    Возвращает True, если флип успешен, False - если нет.
    """
    # Инициализация переменных
    x = np.copy(x_init)
    t = 0
    time_steps = int(duration / DT)
    
    # Симуляция траектории на несколько шагов вперед
    for step in range(time_steps):
        x = dynamic_model(x, u_init, DT)  # Обновляем состояние по динамической модели
        
        # Проверяем ограничения
        pos = x[0:3]
        vel = x[3:6]
        quat = x[6:10]
        omega = x[10:13]
        
        # Ограничение по высоте
        if pos[2] < 0 or pos[2] > MAX_HEIGHT:
            return False  # Если высота выходит за пределы, флип неуспешен
        
        # Ограничение по угловой скорости
        if np.abs(omega[0]) > MAX_RATE or np.abs(omega[1]) > MAX_RATE or np.abs(omega[2]) > MAX_RATE:
            return False  # Если угловая скорость превышает допустимую, флип неуспешен
        
        # Ограничение по вертикальной скорости
        if np.abs(vel[2]) > 10:  # можно изменить это значение, если нужно
            return False  # Если вертикальная скорость слишком велика, флип неуспешен
    
    # Если все ограничения соблюдены, флип успешен
    return True 


# итеративный оптимизатор управления для квадрокоптера
class ILQROptimizer:
    def __init__(self, horizon, state_dim, control_dim, dt):
        """
        dynamics_func: функция f(x, u, dt), принимает батчи
        horizon: горизонт планирования (N)
        state_dim: размерность состояния
        control_dim: размерность управления
        dt: шаг дискретизации
        """
        self.f = self.fx_batch
        self.N = horizon
        self.n = state_dim
        self.m = control_dim
        self.dt = dt
        # Целевая позиция и ориентация для флипа
        self.target_position = np.array([0, 0, 5])  # Достигаем высоты 5 метров
 
    def cost_function_traj_flip(self, x_traj, u_traj, x_target_traj, u_target_traj, Q, R, Qf, total_time, time_step):
        """
        Функция стоимости для флипа квадрокоптера на всей траектории.
        Вычисляет общую стоимость движения по всей траектории флипа дрона, 
        включая терминальную (финальную) стоимость.

        x_traj - список состояний квадрокоптера на всей траектории (позиция, ориентация и угловая скорость)
        u_traj - список управляющих воздействий (обороты моторов) на всей траектории
        x_target_traj - список целевых состояний на всей траектории
        u_target_traj - список целевых управляющих воздействий на всей траектории
        Q,Qf,R - коэффициенты для ошибок положения/ориентации и управления
        Q, R — скаляры, а Qf — матрица, применимая в терминале как квадратичная форма:
        total_time - общее время траектории
        time_step - шаг времени
        """

        total_cost = 0.0
        num_steps = int(total_time / time_step)

        for i in range(num_steps - 1):  # обычные шаги
            x = x_traj[i]
            u = u_traj[i]
            x_target = x_target_traj[i]
            u_target = u_target_traj[i]

            position_error = np.linalg.norm(x[0:3] - x_target[0:3])
            q_current = x[6:10] / np.linalg.norm(x[6:10])
            q_target = x_target[6:10] / np.linalg.norm(x_target[6:10])
            dot_product = np.clip(np.dot(q_current, q_target), -1.0, 1.0)
            orientation_error = 2.0 * np.arccos(np.abs(dot_product))
            control_error = np.linalg.norm(u - u_target)

            step_cost = Q * (position_error**2 + orientation_error**2) + R * control_error**2
            total_cost += step_cost

        # Терминальная стоимость
        x_final = x_traj[num_steps - 1]
        x_final_target = x_target_traj[num_steps - 1]
        terminal_error = x_final - x_final_target
        terminal_cost = terminal_error.T @ Qf @ terminal_error
        total_cost += terminal_cost

        return total_cost

    def fx_batch(self, x, u, dt):
        """
        Нелинейная модель квадрокоптера в режиме батча.
        
        x: (B, 13) батч состояний
        u: (B, 4) батч управлений
        dt: скалярный шаг времени
        return: (B, 13) новые состояния
        """
        # Параметры модели
        m = MASS
        I = INERTIA
        arm = ARM_LEN
        kf = K_THRUST
        km = K_TORQUE
        drag = DRAG
        g = np.array([0, 0, 9.81])
        max_speed = MAX_SPEED

        B = x.shape[0]  # размер батча

        pos = x[:, 0:3]
        vel = x[:, 3:6]
        quat = x[:, 6:10]
        omega = x[:, 10:13]

        # Нормируем кватернионы
        quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)

        # Матрицы поворота (batch)
        R_bw = R.from_quat(quat).as_matrix().reshape(B, 3, 3)

        # Тяга моторов
        rpm = np.clip(u, 0, max_speed)
        w_squared = rpm**2
        thrusts = kf * w_squared

        Fz = np.zeros((B, 3))
        Fz[:, 2] = np.sum(thrusts, axis=1)

        F_world = np.einsum('bij,bj->bi', R_bw, Fz) - m * g - drag * vel
        acc = F_world / m

        new_vel = vel + acc * dt
        new_pos = pos + vel * dt + 0.5 * acc * dt**2

        # Моменты
        L = arm # расстояние от центра масс до мотора (плечо рычага).
        tau = np.stack([
            L * (thrusts[:, 1] - thrusts[:, 3]), # τ_roll   = L * (F2 - F4)
            L * (thrusts[:, 2] - thrusts[:, 0]), # τ_pitch  = L * (F3 - F1)
            km * (w_squared[:, 0] - w_squared[:, 1] + w_squared[:, 2] - w_squared[:, 3]) # τ_yaw
        ], axis=1)

        omega_cross = np.cross(omega, (I @ omega.T).T)
        omega_dot = np.linalg.solve(I, (tau - omega_cross).T).T
        new_omega = omega + omega_dot * dt

        # Обновление кватерниона
        omega_quat = np.concatenate([np.zeros((B,1)), omega], axis=1)
        dq = 0.5 * self.quat_multiply_batch(quat, omega_quat)
        new_quat = quat + dq * dt
        new_quat = new_quat / np.linalg.norm(new_quat, axis=1, keepdims=True)

        # Новый x
        x_next = np.zeros_like(x)
        x_next[:, 0:3] = new_pos
        x_next[:, 3:6] = new_vel
        x_next[:, 6:10] = new_quat
        x_next[:, 10:13] = new_omega

        return x_next

    def quat_multiply_batch(self, q, r):
        """
        Умножение кватернионов батчем.
        q, r: (B, 4)
        return: (B, 4)
        """
        w0, x0, y0, z0 = q[:,0], q[:,1], q[:,2], q[:,3]
        w1, x1, y1, z1 = r[:,0], r[:,1], r[:,2], r[:,3]
        return np.stack([
            w0*w1 - x0*x1 - y0*y1 - z0*z1,
            w0*x1 + x0*w1 + y0*z1 - z0*y1,
            w0*y1 - x0*z1 + y0*w1 + z0*x1,
            w0*z1 + x0*y1 - y0*x1 + z0*w1
        ], axis=1)

    def finite_difference_jacobian_batch(self, f, x, u, dt, epsilon=1e-5):
        """
        Численно приближает Якобианы динамической модели 
        по состоянию A = df/dx и по управлению B = df/du 
        методом конечных разностей.

        Порождает батчи слегка возмущённых x и u.
        Вызывает модель f() для всех возмущённых данных.
        Вычисляет разности и делит на 2*epsilon, что реализует центральную разностную аппроксимацию.
        """
        n, m = x.size, u.size

        # Строим батчи для x и u
        delta_x = np.eye(n) * epsilon  # (n, n) - отклонения по состоянию
        delta_u = np.eye(m) * epsilon  # (m, m) - отклонения по управлению

        # Создаем батчи для состояний и управляющих воздействий
        x_batch = np.vstack((x + delta_x, x - delta_x))  # (2n, n)
        u_batch = np.vstack((u + delta_u, u - delta_u))  # (2m, m)

        # Один вызов функции для всех perturbations
        f_batch = f(x_batch, u_batch, dt)  # (2n * 2m, ...) - выход из функции динамики

        # Разделяем на части для вычисления A и B
        f_x = f_batch[:2 * n]  # Результаты для изменения x
        f_u = f_batch[2 * n:]  # Результаты для изменения u
        
        # Вычисление Якобиана A (df/dx)
        A = (f_x[:n] - f_x[n:]) / (2 * epsilon)
        A = A.T  # Транспонируем, чтобы соответствовать размерности

        # Вычисление Якобиана B (df/du)
        B = (f_u[:m] - f_u[m:]) / (2 * epsilon)
        B = B.T  # Транспонируем для соответствия размерности

        return A, B

    def solve(self, x0, u_init, Q, R, Qf, x_goal, u_target_traj, max_iter=100, tol=1e-6):
        """
        Основной метод решения задачи оптимального управления с помощью iLQR. 
        Он минимизирует функцию стоимости, корректируя управляющие воздействия.

        Инициализация траектории состояний и управления.
        -    Вычисление начальной стоимости.
        -    Основной цикл (до max_iter):
        -    Вычисление Якобианов динамики A, B по всей траектории.
        Backward pass:
        -    Итеративно вычисляются приросты стоимостей Vx, Vxx, и матрицы обратной связи K и смещения d.
        Forward pass (пока не реализован в обрезанном коде): строится новая траектория с учётом K, d.
        -    Пересчитывается стоимость и проверяется сходимость по tol.

        """
        horizon = self.N
        state_dim = self.n

        u_traj = u_init.copy()
        x_traj = np.zeros((horizon, state_dim))
        x_traj[0] = x0

        for k in range(horizon-1):
            x_traj[k+1] = self.fx_batch(x_traj[k:k+1], u_traj[k:k+1], self.dt)[0]

        prev_cost = self.cost_function_traj_flip(x_traj, u_traj, x_goal, u_target_traj, Q, R, Qf, self.N * self.dt, self.dt)

        for iteration in range(max_iter):
            A_list = []
            B_list = []

            for k in range(horizon-1):
                A, B = self.finite_difference_jacobian_batch(self.fx_batch, x_traj[k], u_traj[k], self.dt)
                A_list.append(A)
                B_list.append(B)

            # Backward pass
            Vx = Qf @ (x_traj[-1] - x_goal)
            Vxx = Qf.copy()

            K_list = []
            d_list = []

            for k in reversed(range(horizon-1)):
                A = A_list[k]
                B = B_list[k]

                Qx = Q @ (x_traj[k] - x_goal) + A.T @ Vx
                Qu = R @ u_traj[k] + B.T @ Vx
                Qxx = Q + A.T @ Vxx @ A
                Quu = R + B.T @ Vxx @ B
                Qux = B.T @ Vxx @ A

                du = -np.linalg.solve(Quu, Qu) #du = -np.linalg.lstsq(Quu, Qu, rcond=None)[0]
                K = -np.linalg.solve(Quu, Qux) 

                K_list.insert(0, K)
                d_list.insert(0, du)

                Vx = Qx + K.T @ Quu @ du + K.T @ Qu + Qux.T @ du
                Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K

            # Forward pass
            x_new = np.zeros_like(x_traj)
            u_new = np.zeros_like(u_traj)
            x_new[0] = x0

            for k in range(horizon-1):
                du = d_list[k] + K_list[k] @ (x_new[k] - x_traj[k])
                u_new[k] = u_traj[k] + du
                x_new[k+1] = self.f(x_new[k:k+1], u_new[k:k+1], self.dt)[0]

            cost = self.cost_function_traj_flip(x_new, u_new, x_goal, u_target_traj, Q, R, Qf, self.N * self.dt, self.dt)

            if iteration > 0 and abs(prev_cost - cost) < tol:
                print(f'Converged at iteration {iteration}, cost: {cost}')
                break

            prev_cost = cost
            x_traj = x_new
            u_traj = u_new

        return x_traj, u_traj

class ModelPredictiveController:
    def __init__(self, dt, horizon, Q, R, Qf, n, m):
        """
        f: функция динамики: f(x, u, dt) → x_next
        fx_batch: функция для вычисления Якобианов A, B по заданному состоянию и управлению
        dt: шаг по времени
        horizon: длина горизонта предсказания
        Q, R, Qf: матрицы весов стоимости
        n: размерность состояния
        m: размерность управления
        """
        self.dt = dt
        self.horizon = horizon
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.n = n
        self.m = m
        self.optimizer = ILQROptimizer(
            dt=dt,
            horizon=horizon,
            state_dim=n,
            control_dim=m
        )

    def step(self, x0, u_init, x_goal, u_target_traj):
        """
        Вычисляет оптимальную траекторию состояний и управляющих воздействий
        от текущего состояния x0, используя iLQR.
        """
        x_traj_opt, u_traj_opt = self.optimizer.solve(
            x0=x0,
            u_init=u_init,
            Q=self.Q,
            R=self.R,
            Qf=self.Qf,
            x_goal=x_goal,
            u_target_traj=u_target_traj
        )
        return x_traj_opt, u_traj_opt[0]  # Возвращаем x_traj_opt и первое управляющее воздействие (как в MPC)
   
 
def main(args=None):
    rclpy.init(args=args)
    node = DynamicModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

