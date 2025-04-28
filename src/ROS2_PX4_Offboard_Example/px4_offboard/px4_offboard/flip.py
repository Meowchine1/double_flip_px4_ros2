import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Quaternion, Vector3, Twist, PoseStamped
from scipy.spatial.transform import Rotation as R
import tf_transformations
from math import degrees
 
from std_msgs.msg import Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    VehicleTorqueSetpoint, VehicleAngularVelocity, VehicleCommand, VehicleAttitude,
    SensorCombined, VehicleImu, VehicleOdometry, OffboardControlMode, TrajectorySetpoint, 
    VehicleStatus, VehicleAttitudeSetpoint, VehicleRatesSetpoint, VehicleLocalPosition, 
    ActuatorMotors, VehicleAngularAccelerationSetpoint
)
import numpy as np
from enum import Enum
from px4_offboard.mpc_controller import MPCController
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry  # inner ROS2 SEKF

import casadi as ca

BOUNCE_TIME = 0.6
ACCELERATE_TIME = 0.07
BRAKE_TIME = ACCELERATE_TIME
ARM_TIMEOUT = 5.0
OFFBOARD_TIMEOUT = 5.0
TAKEOFF_HEIGHT = -5.0
FLIP_HEIGHT = -7.0

class DroneState(Enum):
    INIT = 7
    DISARMED = 0
    ARMING = 1
    ARMED = 2
    OFFBOARD = 3
    TAKEOFF = 4
    FLIP = 6
    READY_FOR_FLIP = 8
    LANDING = 16

class DroneFlipState(Enum):
    INIT = 0
    FLIP_INIT = 7
    FLIP_TURNED_315 = 1
    FLIP_TURNED_45 = 2
    BRAKE = 3
    POST_BRAKE = 4
    RECOVERY = 5
    LAND = 6

class FlipControlNode(Node):
    def __init__(self):
        super().__init__('flip_control_node')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        # Публикации команд 
        self.vehicle_command_publisher = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.offboard_control_mode_publisher = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.quaternion_setpoint_publisher = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint', qos_profile)
        self.vehicle_rates_publisher = self.create_publisher(VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile)
        self.vehicle_torque_publisher = self.create_publisher(VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', qos_profile)
        self.publisher_rates = self.create_publisher(VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile)
        self.publisher_att = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint', qos_profile)

        # Подписки на состояние дрона
        self.create_subscription(SensorCombined, '/fmu/out/sensor_combined', self.sensor_combined_callback, qos_profile)
        self.create_subscription(VehicleAngularVelocity, '/fmu/out/vehicle_angular_velocity', self.angular_velocity_callback, qos_profile)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.vehicle_attitude_callback, qos_profile)
        self.create_subscription(VehicleAngularAccelerationSetpoint,
        '/fmu/out/vehicle_angular_acceleration_setpoint', self.vehicle_angular_acceleration_setpoint_callback, qos_profile)
        self.create_subscription(VehicleImu,'/fmu/in/vehicle_imu',self.vehicle_imu_callback, qos_profile)
        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos_profile)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.create_subscription(ActuatorMotors, '/fmu/out/actuator_motors', self.actuator_motors_callback, qos_profile)
        self.create_subscription(TrajectorySetpoint, '/fmu/out/trajectory_setpoint', self.trajectory_setpoint_callback, qos_profile)
         
        # drone_dynamics node topic
        self.create_subscription(PoseStamped,'/quad/pose_pred',self.pose_callback, qos_profile) 

        self.ekf_state_sub = self.create_subscription(Float32MultiArray,'/ekf/state',self.ekf_state_callback, qos_profile)
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, qos_profile)

        # STATE
        self.main_state = DroneState.INIT
        self.flip_state = DroneFlipState.INIT
        self.arming_state = 0 
        self.nav_state = 0 
        self.vehicle_status = VehicleStatus()
        self.vehicle_local_position = VehicleLocalPosition()
        self.stage_time = time.time()
        self.offboard_is_active = False
  
        #actuator_motors_callback
        self.torque_roll = 0.0
        self.torque_pitch = 0.0
        self.torque_yaw = 0.0

        #trajectory setpoint callback
        self.position = np.zeros(3, dtype=np.float32) # in meters
        self.velocity = np.zeros(3, dtype=np.float32) # in meters/second
        self.acceleration = np.zeros(3, dtype=np.float32) # in meters/second^2

        #sensor_combined_callback
        self.q = np.array([0.0, 0.0, 0.0, 1.0])  # Начальная ориентация (кватернион, идентичность)
        self.omega = np.zeros(3)  # Начальная угловая скорость (вектор, 3D)
        self.theta = np.zeros(3)  # Начальный угол поворота (вектор, 3D)
        self.linear_acceleration = np.zeros(3)  # Начальное линейное ускорение (вектор, 3D)

        # Accumulators
        self.roll_accum = 0.0
        self.prev_roll = 0.0
        self.flip_count = 0

        self.odom_callback_position = np.zeros(3, dtype=np.float32)
        self.odom_callback_orientation = np.zeros(4, dtype=np.float32)
        
        self.ekf.x = np.zeros(13) # вектор состояния (13 штук: позиция, скорость, ориентация (4), угловые скорости).

        # Таймеры
        self.create_timer(0.1, self.update)
        self.create_timer(0.1, self.offboard_heartbeat)
        self.create_timer(0.01, self.flip_thrust_max) 
        self.create_timer(0.001, self.flip_thrust_recovery) 
        self.create_timer(0.0001, self.flip_pitch_t)
        # Timer flags
        self.flip_thrust_max_f = False
        self.flip_thrust_recovery_f = False
        self.flip_pitch_f = False

    
    # Maths functions start
    def rotate_vector_by_quaternion(self, vector, quat):# ФУНКЦИЯ НЕ НУЖНА
        qx, qy, qz, qw = quat.x, quat.y, quat.z, quat.w
        vx, vy, vz = vector.x, vector.y, vector.z

        tx = 2 * (qy * vz - qz * vy)
        ty = 2 * (qz * vx - qx * vz)
        tz = 2 * (qx * vy - qy * vx)

        result_x = vx + qw * tx + (qy * tz - qz * ty)
        result_y = vy + qw * ty + (qz * tx - qx * tz)
        result_z = vz + qw * tz + (qx * ty - qy * tx)

        return Vector3(x=result_x, y=result_y, z=result_z)

    def quaternion_from_euler(self, roll, pitch, yaw):# ФУНКЦИЯ НЕ НУЖНА
        r = R.from_euler('xyz', [roll, pitch, yaw])
        return np.array(r.as_quat(), dtype=np.float32)
    
    def euler_from_self_quaternion(self):# ФУНКЦИЯ НЕ НУЖНА
        """Преобразование кватерниона в углы Эйлера (возвращает roll, pitch, yaw)."""
        r = R.from_quat([self.quaternion[0], self.quaternion[1], self.quaternion[2], self.quaternion[3]])
        return r.as_euler('xyz')

    # Maths functions end

    # callbacks start
    # ======== USED ============= 
    def odom_callback(self, msg: Odometry):
        self.get_logger().info("odom_callback")
        self.odom_callback_position = msg.pose.pose.position
        self.odom_callback_orientation = msg.pose.pose.orientation

    def ekf_state_callback(self, msg):
        self.get_logger().info("ekf_state_callback")
        self.ekf_x = msg.data # вектор состояния (13 штук: позиция, скорость, ориентация (4), угловые скорости).

    def vehicle_status_callback(self, msg):# YES
        """Обновляет состояние дрона."""
        #self.get_logger().info('vehicle_status_callback')
        self.vehicle_status = msg
        self.arming_state = msg.arming_state

    # send functions
    def send_land_command(self):
        land_cmd = VehicleCommand()
        land_cmd.command = VehicleCommand.VEHICLE_CMD_NAV_LAND
        land_cmd.param1 = 0.0  # Приземление (не отменять)
        land_cmd.param2 = 0.0  # Опция (оставляем по умолчанию)
        land_cmd.param3 = 0.0  # Нет конкретной цели (текущая позиция)
        self.vehicle_command_publisher.publish(land_cmd)
        self.get_logger().info('Sending land command.')
        
    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        """Отправка команды дрону."""
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(time.time() * 1e6)
        self.vehicle_command_publisher.publish(msg)
        self.get_logger().info(f'publish_vehicle_command')

    def publish_position_setpoint(self, x, y, z):
        trajectory_msg = TrajectorySetpoint()
        trajectory_msg.position = [x, y, -z]
        trajectory_msg.timestamp = int(time.time() * 1e6)
        self.trajectory_setpoint_publisher.publish(trajectory_msg)
        #self.get_logger().info(f'publish_position_setpoint')

    def publish_rate_setpoint(self, roll_rate, pitch_rate, yaw_rate):
        rate_msg = VehicleRatesSetpoint()
        rate_msg.roll = roll_rate 
        rate_msg.pitch = pitch_rate
        rate_msg.yaw = yaw_rate
        rate_msg.timestamp = int(time.time() * 1e6)
        self.vehicle_rates_publisher.publish(rate_msg)
        self.get_logger().info(f'publish_rate_setpoint')

    def publish_torque_setpoint(self, roll_torque, pitch_torque, yaw_torque):
        torque_msg = VehicleTorqueSetpoint()
        torque_msg.xyz[0] = roll_torque  # Torque around X-axis
        torque_msg.xyz[1] = pitch_torque  # Torque around Y-axis
        torque_msg.xyz[2] = yaw_torque  # Torque around Z-axis
        torque_msg.timestamp = int(time.time() * 1e6)
        self.vehicle_torque_publisher.publish(torque_msg)
        self.get_logger().info(f'publish_torque_setpoint')

    def publish_attitude_setpoint(self, roll, pitch, yaw, thrust):
        attitude_msg = VehicleAttitudeSetpoint()
        attitude_msg.q_d = self.quaternion_from_euler(roll, pitch, yaw)  # Устанавливаем кватернион
        attitude_msg.thrust_body[2] = -thrust  # Negative Z thrust
        attitude_msg.timestamp = int(time.time() * 1e6)
        self.quaternion_setpoint_publisher.publish(attitude_msg)
        self.get_logger().info(f'publish_attitude_setpoint')
    
    def publish_thrust_setpoint(self, thrust):
        attitude_msg = VehicleAttitudeSetpoint()
        attitude_msg.thrust_body[2] = -thrust  # Negative Z thrust
        attitude_msg.timestamp = int(time.time() * 1e6)
        self.quaternion_setpoint_publisher.publish(attitude_msg)
        self.get_logger().info(f'publish_thrust_setpoint')
    
    def set_stabilization_mode(self):
        """Переводит дрон в режим стабилизации (STABILIZE)."""
        msg = VehicleCommand()
        msg.command = 1  # Команда для перехода в режим стабилизации (STABILIZE)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(time.time() * 1e6)
        self.vehicle_command_publisher.publish(msg)
        self.get_logger().info(f'set_stabilization_mode')

    def set_motor_commands(self, motor_rear_left, motor_rear_right, motor_front_left, motor_front_right):
        """
        Функция для отправки команд включения моторов в режим Offboard.
        Каждый мотор включается/выключается через команду VehicleCommand.
        """
        # Команда VehicleCommand для управления моторами
        vehicle_command = VehicleCommand()
        # Для включения моторов можно использовать команду типа MOTOR_SET, которая управляет каждым из моторов
        # Важно: значения могут быть от 0 (выключен) до 1 (включен)
        vehicle_command.command = 183  # 183 - это команда для управления моторами
        vehicle_command.param1 = motor_rear_left
        vehicle_command.param2 = motor_rear_right
        vehicle_command.param3 = motor_front_left
        vehicle_command.param4 = motor_front_right
        vehicle_command.timestamp = int(time.time() * 1e6)
        # Публикуем команду на топик для управления моторами
        self.vehicle_command_publisher.publish(vehicle_command)
        self.get_logger().info(f'set_motor_commands')

    def reset_rate_pid(self): 
        self.publish_rate_setpoint(roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0)
        self.get_logger().info(f'reset_rate_pid')

    def set_offboard_mode(self):
        """Switch to offboard mode."""
        self.offboard_is_active = True
    
    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def set_rates(self, roll_rate, pitch_rate, yaw_rate, thrust):
        msg = VehicleRatesSetpoint()
        msg.roll = roll_rate
        msg.pitch = pitch_rate
        msg.yaw = yaw_rate
        msg.thrust_body[2] = -thrust  # PX4 uses Z-down
        self.publisher_rates.publish(msg)

    def set_thrust(self, thrust):
        msg = VehicleAttitudeSetpoint()
        msg.thrust_body[2] = -thrust
        self.publisher_att.publish(msg)

    def send_velocity(self, linear_x, linear_y, linear_z, angular_z):
        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = linear_y
        twist.linear.z = linear_z
        twist.angular.z = angular_z
        self.publisher.publish(twist)

    def get_current_state(self):
        #  OR return self.droneDynamics_state # метод, возвращающий [pos, vel, q, omega]
        pos = self.droneDynamics_IMU_loc_data# TODO
        vel = self.droneDynamics_velocity# TODO
        q = self.droneDynamics_attitude# TODO
        omega = self.droneDynamics_angular_velocity# TODO
        return np.concatenate([pos, vel, q, omega])
    
    def send_motor_commands(self, motor_inputs):
        # нормализованные значения [0,1] — переводим в pwm/thrust команды
        msg = ActuatorMotors()
        msg.control = list(np.clip(motor_inputs, 0.0, 1.0))
        self.motor_pub.publish(msg)

    """ Дрон должен постоянно получать это сообщение чтобы оставаться в offboard """
    def offboard_heartbeat(self):
         if self.offboard_is_active:
                #self.get_logger().info("Sending SET_MODE OFFBOARD") 
                """Publish the offboard control mode."""
                msg = OffboardControlMode()
                msg.position = True
                msg.velocity = False
                msg.acceleration = False
                msg.attitude = False
                msg.body_rate = False
                msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                self.offboard_control_mode_publisher.publish(msg)

    def flip_thrust_max(self):
        if self.flip_thrust_max_f:
            self.set_thrust(1.0)

    def flip_thrust_recovery(self):
        if self.flip_thrust_recovery_f:
            self.set_thrust(0.6)

    def flip_pitch_t(self):
        if self.flip_pitch_f:
            #self.set_rates(17.0, 0.0, 0.0, 0.25)# roll_max_rate should be 1000 in QGC vechicle setup
            self.set_rates(25.0, 0.0, 0.0, 0.25)

    # main spinned function
    def update(self):
        #self.get_logger().info(f"self.main_state={self.main_state}  self.flip_state={self.flip_state}")
        #self.get_logger().info(f"UPDATE self.alt={self.alt}  self.vehicle_local_position.z={self.vehicle_local_position.z}")
        if self.main_state == DroneState.INIT:
            self.set_offboard_mode()
            self.arm()
            self.main_state = DroneState.ARMING 
         
        elif self.main_state == DroneState.ARMING:
            self.get_logger().warn(f"arm_state: {self.arming_state}")
            self.get_logger().info(f"nav_state: {self.nav_state}")
            if self.arming_state == VehicleStatus.ARMING_STATE_ARMED:
                self.get_logger().info('ARMING_STATE_ARMED')
                self.main_state = DroneState.TAKEOFF
            else:
                self.set_offboard_mode()
                self.arm()

        elif self.main_state == DroneState.TAKEOFF:
            self.publish_position_setpoint(0.0, 0.0, TAKEOFF_HEIGHT)
            #self.get_logger().info(f'self.arming_state {self.arming_state} {self.health_warning}')
            #self.get_logger().info(f"position.z: {self.vehicle_local_position.z} self.takeoff_height: {self.takeoff_height} res:{self.vehicle_local_position.z  + 0.2 <= -self.takeoff_height}") 
            if self.vehicle_local_position.z  - 0.5 <= TAKEOFF_HEIGHT:
                self.main_state = DroneState.READY_FOR_FLIP

        elif self.main_state == DroneState.READY_FOR_FLIP:
            #self.get_logger().info(f'self.roll angular_velocity={self.angular_velocity[0]} abs(self.roll={abs(self.roll)}')
            if self.roll is not None:
                if (abs(self.angular_velocity[0]) < 0.05 and (abs(self.roll) < 2.0)):
                    self.main_state = DroneState.FLIP
                    self.stage_time =  time.time()

        ## PITCH FLIP
        elif self.main_state == DroneState.FLIP:
            target_pos = np.array([0.0, 0.0, FLIP_HEIGHT])
            target_q = R.from_euler('xyz', [np.pi, 0, 0]).as_quat()  # 180° roll

            current_state = self.get_current_state()  # метод, возвращающий [pos, vel, q, omega]
             
            optimal_motor_inputs = self.mpc.solve(current_state, target_pos, target_q) # Вычисление оптимального управляющего воздействия
             
            self.send_motor_commands(optimal_motor_inputs)
            # Обновление накопленного roll с учётом wrap-around
            # roll_diff = self.roll - self.prev_roll
            # if roll_diff > 180.0:
            #     roll_diff -= 360.0
            # elif roll_diff < -180.0:
            #     roll_diff += 360.0

            # self.roll_accum += roll_diff
            # self.prev_roll = self.roll

            # self.get_logger().info(
            #     f"[FLIP] roll={self.roll:.2f}, self.flip_count={self.flip_count}, roll_diff={roll_diff}, roll_accum={self.roll_accum:.2f}, alt={self.alt:.2f}, flip_state={self.flip_state.name}")

            # # 1) Взлёт на высоту
            # if self.flip_state == DroneFlipState.INIT:
            #     if self.alt - 0.5 <= FLIP_HEIGHT:
            #         self.flip_thrust_max_f = True
            #     else:
            #         self.flip_thrust_max_f = False
            #         self.flip_pitch_f = True
            #         self.roll_accum = 0.0  # сброс перед началом вращения
            #         self.prev_roll = self.roll
            #         self.flip_state = DroneFlipState.FLIP_INIT

            # # 2) Отслеживание прогресса флипа
            # elif self.flip_state == DroneFlipState.FLIP_INIT and self.roll_accum > 45.0:
            #     self.flip_state = DroneFlipState.FLIP_TURNED_45

            # elif self.flip_state == DroneFlipState.FLIP_TURNED_45 and self.roll_accum > 315.0:
            #     self.flip_state = DroneFlipState.FLIP_TURNED_315
            #     self.flip_pitch_f = False

            # # 3) Если первый флип завершён, начинаем второй флип (повторяем состояния)
            # elif self.flip_state == DroneFlipState.FLIP_TURNED_315 and self.roll_accum > 360.0:
            #     # Второй флип — сбросим состояние и начнём снова отслеживать углы
            #     self.flip_count += 1  # Увеличиваем счетчик флипов
            #     if self.flip_count < 2:  # Проверяем, если это первый флип, то повторяем
            #         self.flip_state = DroneFlipState.FLIP_INIT  # Сброс состояния для второго флипа
            #         self.roll_accum = 0.0  # Сброс накопленного угла для второго флипа
            #     else:  # После второго флипа — выход в landing
            #         self.flip_state = DroneFlipState.LANDING
            #         self.main_state = DroneState.LANDING  # Переход в состояние посадки

            # # 4) Восстановление после флипа
            # elif self.flip_state == DroneFlipState.FLIP_TURNED_315:
            #     if self.alt < 5.0:
            #         self.flip_thrust_recovery_f = True
            #     else:
            #         self.flip_thrust_recovery_f = False
            #         self.main_state = DroneState.LANDING
            #         self.flip_count = 0
            
                
        elif self.main_state == DroneState.LANDING:
            self.publish_position_setpoint(0.0, 0.0, 1.0)

def main(args=None):
    rclpy.init(args=args)
    node = FlipControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
