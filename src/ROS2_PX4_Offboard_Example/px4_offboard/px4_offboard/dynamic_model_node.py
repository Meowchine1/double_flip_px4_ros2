import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3, Quaternion
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

 
class iLQR:
    def __init__(self, dynamics_func, horizon, state_dim, control_dim, dt):
        """
        dynamics_func: функция f(x, u, dt), принимает батчи
        horizon: горизонт планирования (N)
        state_dim: размерность состояния
        control_dim: размерность управления
        dt: шаг дискретизации
        """
        self.f = dynamics_func
        self.N = horizon
        self.n = state_dim
        self.m = control_dim
        self.dt = dt
        # Целевая позиция и ориентация для флипа
        self.target_position = np.array([0, 0, 5])  # Достигаем высоты 5 метров

    def update_target_orientation(self, time_step, total_time):
        """
        Линейное изменение ориентации по roll от 0 до 180 градусов.
        """
        roll_target = time_step / total_time  # Линейное изменение от 0 до 1
        angle = roll_target * np.pi  # от 0 до π рад (180 градусов)
        qx = np.array([np.cos(angle/2), np.sin(angle/2), 0, 0])
        return qx

    def cost_function_traj_flip(self, x_traj, u_traj, x_target_traj, u_target_traj, Q, R, total_time, time_step):
        """
        Функция стоимости для флипа квадрокоптера на всей траектории.

        x_traj - список состояний квадрокоптера на всей траектории (позиция, ориентация и угловая скорость)
        u_traj - список управляющих воздействий (обороты моторов) на всей траектории
        x_target_traj - список целевых состояний на всей траектории
        u_target_traj - список целевых управляющих воздействий на всей траектории
        Q, R - коэффициенты для ошибок положения/ориентации и управления
        total_time - общее время траектории
        time_step - шаг времени
        """

        total_cost = 0.0
        num_steps = int(total_time / time_step)

        for i in range(num_steps):
            # Получаем текущее состояние и управление на i-м шаге
            x = x_traj[i]
            u = u_traj[i]

            # Получаем целевые состояние и управление на i-м шаге
            x_target = x_target_traj[i]
            u_target = u_target_traj[i]

            # Обновляем целевую ориентацию для флипа (с изменением roll от 0 до 180 градусов)
            target_orientation = self.update_target_orientation(i * time_step, total_time)

            # Ошибка положения (позиция квадрокоптера относительно целевой позиции)
            position_error = np.linalg.norm(x[0:3] - x_target[0:3])

            # Ошибка ориентации (угловая ошибка между кватернионами)
            q_current = x[6:10] / np.linalg.norm(x[6:10])  # нормируем на всякий случай
            q_target = target_orientation / np.linalg.norm(target_orientation)
            dot_product = np.clip(np.dot(q_current, q_target), -1.0, 1.0)  # защита от численных ошибок
            orientation_error = 2.0 * np.arccos(np.abs(dot_product))  # угловая ошибка в радианах

            # Ошибка управления (различие в оборотах моторов)
            control_error = np.linalg.norm(u - u_target)

            # Стоимость для текущего шага (с учётом коэффициентов Q и R)
            step_cost = Q * (position_error**2 + orientation_error**2) + R * control_error**2

            # Добавляем стоимость текущего шага к общей стоимости
            total_cost += step_cost

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
        L = arm
        tau = np.stack([
            L * (thrusts[:, 1] - thrusts[:, 3]),
            L * (thrusts[:, 2] - thrusts[:, 0]),
            km * (w_squared[:, 0] - w_squared[:, 1] + w_squared[:, 2] - w_squared[:, 3])
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
        n, m = x.size, u.size

        # Строим батчи для x и u
        delta_x = np.eye(n) * epsilon  # (n, n) - отклонения по состоянию
        delta_u = np.eye(m) * epsilon  # (m, m) - отклонения по управлению

        # Создаем батчи для состояний и управляющих воздействий
        x_batch = np.vstack((x + delta_x, x - delta_x))  # (2n, n)
        u_batch = np.vstack((u + delta_u, u - delta_u))  # (2m, m)

         
        x_batch = np.vstack([x + delta_x, x - delta_x])  # (2n, n)
        u_batch = np.vstack([u + delta_u, u - delta_u])  # (2m, m)

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


    def solve(self, x0, u_init, Q, R, Qf, x_goal, max_iter=100, tol=1e-6):
        """
        Основной цикл iLQR с использованием полной функции стоимости по траектории.
        """

        N = self.N
        n = self.n
        m = self.m

        u_traj = u_init.copy()
        x_traj = np.zeros((N, n))
        x_traj[0] = x0

        for k in range(N-1):
            x_traj[k+1] = self.f(x_traj[k:k+1], u_traj[k:k+1], self.dt)[0]

        prev_cost = self.cost_function(x_traj, u_traj, x_goal, Q, R, Qf)

        for iteration in range(max_iter):
            A_list = []
            B_list = []

            for k in range(N-1):
                A, B = self.finite_difference_jacobian_batch(self.f, x_traj[k], u_traj[k], self.dt)
                A_list.append(A)
                B_list.append(B)

            # Backward pass
            Vx = Qf @ (x_traj[-1] - x_goal)
            Vxx = Qf.copy()

            K_list = []
            d_list = []

            for k in reversed(range(N-1)):
                A = A_list[k]
                B = B_list[k]

                Qx = Q @ (x_traj[k] - x_goal) + A.T @ Vx
                Qu = R @ u_traj[k] + B.T @ Vx
                Qxx = Q + A.T @ Vxx @ A
                Quu = R + B.T @ Vxx @ B
                Qux = B.T @ Vxx @ A

                # Решаем Quu * du = -Qu
                du = -np.linalg.solve(Quu, Qu)
                K = -np.linalg.solve(Quu, Qux)

                K_list.insert(0, K)
                d_list.insert(0, du)

                Vx = Qx + K.T @ Quu @ du + K.T @ Qu + Qux.T @ du
                Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K

            # Forward pass (линейная коррекция)
            x_new = np.zeros_like(x_traj)
            u_new = np.zeros_like(u_traj)

            x_new[0] = x0

            for k in range(N-1):
                du = d_list[k] + K_list[k] @ (x_new[k] - x_traj[k])
                u_new[k] = u_traj[k] + du
                x_new[k+1] = self.f(x_new[k:k+1], u_new[k:k+1], self.dt)[0]

            # Новая стоимость по всей траектории
            cost = self.cost_function(x_new, u_new, x_goal, Q, R, Qf)

            if iteration > 0 and abs(prev_cost - cost) < tol:
                print(f'Converged at iteration {iteration}, cost: {cost}')
                break

            prev_cost = cost
            x_traj = x_new
            u_traj = u_new

        return x_traj, u_traj

 


# Функция для MPC
def mpc_controller(x_current, N, dt, Q, R, target_position):
    # Предсказание оптимальной траектории с помощью iLQR
    u0 = np.zeros(4)  # Начальные условия управления
    x0 = x_current  # Текущее состояние
    X_opt, U_opt = ilqr_flip(x0, u0, N, dt, Q, R, target_position)
    
    # Решение задачи MPC на текущем шаге
    # Для MPC можно решить задачу оптимизации на основе текущего состояния
    # и предсказанных управляющих воздействий (U_opt).
    U_mpc = U_opt[0]  # Возьмём первый управляющий сигнал из предсказанной траектории

    return U_mpc

def start_flip():
    # Начальные условия для флипа
    x0 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])  # Начальная позиция и ориентация
    u0 = np.array([1000, 1000, 1000, 1000])  # Начальные обороты моторов

    # Параметры задачи флипа
    N = 50  # Количество шагов
    dt = 0.05  # Шаг времени
    Q = 1  # Вес для положения и ориентации
    R = 0.01  # Вес для управления

    # Целевая позиция для флипа
    target_position = np.array([0, 0, 1])  # Например, флип до высоты 1 метр

    # Использование MPC для флипа
    U_mpc = mpc_controller(x0, N, dt, Q, R, target_position)
    print("Оптимальные управляющие воздействия от MPC:")
    print(U_mpc)

# ===== MATRIX OPERTIONS =====
# QUATERNION UTILS (SCIPY-based)

def quat_multiply(q1, q2):
    """Multiply two quaternions [x, y, z, w] using scipy."""
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    return (r1 * r2).as_quat()

def normalize_quat(q):
    """Normalize a quaternion."""
    return R.from_quat(q).as_quat()
 
def quat_to_matrix(q):
    """Convert quaternion [x, y, z, w] to rotation matrix."""
    return R.from_quat(q).as_matrix()

class MyEKF(ExtendedKalmanFilter):
    def __init__(self, dim_x, dim_z):
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

        # Паблишеры для ekf_filter_node
        self.imu_pub = self.create_publisher(Imu, '/imu/data', qos_profile)
        self.mag_pub = self.create_publisher(MagneticField, '/imu/mag', qos_profile)
        self.baro_pub = self.create_publisher(FluidPressure, '/baro', qos_profile)
        self.ekf_state_pub = self.create_publisher(Float32MultiArray, '/ekf/state', qos_profile)

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
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, qos_profile)

        # ======= DATA USED IN METHODS  ======= 
        self.angularVelocity = np.zeros(3, dtype=np.float32)
        self.angularVelocity_angular_acceleration = np.zeros(3, dtype=np.float32)
        self.vehicleImu_velocity_w = np.zeros(3, dtype=np.float32) # в мировых координатах 
        self.sensorCombined_linear_acceleration = np.zeros(3, dtype=np.float32)
        self.position = np.zeros(3, dtype=np.float32) # drone position estimates with IMU localization
        self.motor_inputs = np.zeros(4, dtype=np.float32)  # нормализованные входы [0..1] или RPM
        self.vehicleAttitude_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32) # quaternion from topic
        self.magnetometer_data = np.zeros(3, dtype=np.float32)
        self.baro_attitude = 0.0
        self.baro_pressure = 0.0
        self.mag_yaw = 0.0
        # FOR TESTING IN SITL
        self.vehicleLocalPosition_position = np.zeros(3, dtype=np.float32)
        # FOR TESTING INNER EKF
        self.odom_callback_position = np.zeros(3, dtype=np.float32)
        self.odom_callback_orientation = np.zeros(4, dtype=np.float32)

        # ======= OTHER TOPIC DATA ======= 
        # :[TOPIC NAME]_[PARAM NAME] OR [TOPIC NAME] IF PARAM = TOPIC NAME
        self.sensorCombined_angular_velocity = np.zeros(3, dtype=np.float32)
        self.vehiclAngularAcceleration_angular_acceleration = np.zeros(3, dtype=np.float32)
        self.baro_temperature = 0.0 # temperature in degrees Celsius

        # ======= EKF =======
        # вектор состояния (13 штук: позиция, скорость, ориентация (4), угловые скорости).
        # наблюдаемые величины (6 штук) позиция и линейная скорость
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
        
        # ======= TIMERS =======
        self.timer = self.create_timer(0.01, self.step_dynamics)
        self.EKF_timer = self.create_timer(EKF_DT, self.EKF)
        self.ekf_filter_node_timer = self.create_timer(0.01, self.ekf_filter_node_t) 
        #self.logger_timer = self.create_timer(0.1, self.logger)
        self.last_time = Time.time()

        now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_file_name_pos = f"{now_str}_pos.log"
        self.log_file_name_quat = f"{now_str}_quat.log"
        self.log_file_name_vel = f"{now_str}_vel.log"
        self.log_file_name_ang_vel = f"{now_str}_ang_vel.log"
         
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
        self.vehiclAngularAcceleration_angular_acceleration = msg.xyz

    def vehicle_imu_callback(self, msg: VehicleImu):
        # Извлекаем приращение угла и скорости
        #delta_angle = np.array(msg.delta_angle, dtype=np.float32)  # рад
        #delta_angle_dt = msg.delta_angle_dt * 1e-6  # с
        delta_velocity = np.array(msg.delta_velocity, dtype=np.float32)  # м/с
        delta_velocity_dt = msg.delta_velocity_dt * 1e-6  # с
        # Проверяем наличие ориентации и валидного времени интеграции
        if self.vehicleAttitude_q is not None and delta_velocity_dt > 0.0:
            # Преобразование ориентации в матрицу поворота
            rotation = R.from_quat(self.vehicleAttitude_q)
            # Переводим приращение скорости в мировую систему координат
            delta_velocity_world = rotation.apply(delta_velocity)
            # Добавляем гравитацию
            gravity = np.array([0.0, 0.0, 9.80665], dtype=np.float32)
            delta_velocity_world += gravity * delta_velocity_dt
            # Интегрируем скорость и позицию
            self.vehicleImu_velocity_w += delta_velocity_world
            self.position += self.vehicleImu_velocity_w * delta_velocity_dt

    def publish_motor_inputs(self):
        msg = Float32MultiArray()
        msg.data = self.motor_inputs.tolist()
        self.motor_pub.publish(msg)

    def EKF(self):
        """ Основная функция обновления фильтра Калмана. """
        vel_world = self.vehicleImu_velocity_w #R_bw @ self.vehicleImu_velocity_bf
        # Задаём данные от датчиков (барометр и магнитометр)
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



    # что бы показали датчики при таком состоянии.
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


def main(args=None):
    rclpy.init(args=args)
    node = DynamicModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

