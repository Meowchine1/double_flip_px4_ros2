import time
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Quaternion, Vector3
import tf2_ros
import tf2_geometry_msgs
import tf_transformations
from std_msgs.msg import Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    VehicleTorqueSetpoint, VehicleAngularVelocity, VehicleCommand,
    OffboardControlMode, TrajectorySetpoint, VehicleStatus, VehicleAttitudeSetpoint,
    VehicleRatesSetpoint, VehicleLocalPosition
)

from enum import Enum 

# Временные параметры
BOUNCE_TIME = 0.6
ACCELERATE_TIME = 0.07
BRAKE_TIME = ACCELERATE_TIME
ARM_TIMEOUT = 5.0
OFFBOARD_TIMEOUT = 5.0

# Перечисление состояний
class FlipStage(Enum):
    DISARMED = 0
    ARMING = 1
    ARMED = 2
    OFFBOARD = 3
    TAKEOFF = 4
    WAIT = 5
    BOUNCE = 6
    ACCELERATE = 7
    ROTATE = 8
    BRAKE = 9
    POST_BRAKE = 10
    RECOVERY = 11
    LAND = 12
    INIT = 13
    READY_FOR_FLIP = 14
    FLIPPING = 15
    LANDING = 16

    
# class FlipStage(Enum):
#     INIT = 0
#     ARMING = 1
#     TAKEOFF = 2
#     READY_FOR_FLIP = 3
#     FLIPPING = 4
#     LANDING = 5
#     DISARMED = 6


class EulerAngles:
    """
    Класс для представления углов Эйлера.
    """
    def __init__(self, roll, pitch, yaw):
        self.roll = roll   # roll
        self.pitch = pitch  # pitch
        self.yaw = yaw    # yaw

class FlipControlNode(Node):
    def __init__(self):
        super().__init__('flip_control_node')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # Публикации команд 
        self.vehicle_command_publisher = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.offboard_control_mode_publisher = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.attitude_setpoint_publisher = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint', qos_profile)
        self.vehicle_rates_publisher = self.create_publisher(VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile)
        self.vehicle_torque_publisher = self.create_publisher(VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', qos_profile)

        # Подписки на состояние дрона
        # Subscribers
        #self.create_subscription(Float32, 'rc_channel_mode', self.interpret_rc, qos_profile)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)

        self.create_subscription(Vector3, '/fmu/out/sensor_combined', self.imu_callback, qos_profile)
        self.create_subscription(Quaternion, '/fmu/out/vehicle_attitude', self.imu_attitude_callback, qos_profile) # содержит ориентацию дрона в виде кватерниона

        # Переменные состояния
        self.vehicle_status = VehicleStatus()
        self.vehicle_local_position = VehicleLocalPosition()
        
        #position_callback
        self.x = 0
        self.y = 0
        self.z = 0
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.timestamp = 0

        self.flip_stage = FlipStage.INIT
        #self.state = FlipStage.INIT
        self.stage_time = time.time()
        self.takeoff_height = 5.0
        self.offboard_is_active = False

        self.thrust_target = 0.0
        self.last_time = -999
        self.max_rate = 0.0 
        self.acceleration_angle_z = 0.0
        self.acceleration_when_rotating_time = 0.0 
        self.rates = Vector3()
        self.attitude = Quaternion()

        # Таймеры
        self.create_timer(0.1, self.update)
        self.create_timer(0.1, self.offboard_heartbeat)


    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position

    def vehicle_status_callback(self, msg):
        """Обновляет состояние дрона."""
        self.get_logger().info('vehicle_status_callback')
        self.vehicle_status = msg

    # def position_callback(self, msg): 
    #     self.x = msg.x
    #     self.y = msg.y
    #     self.z = msg.z
    #     self.vx = msg.vx
    #     self.vy = msg.vy
    #     self.vz = msg.vz
    #     self.timestamp = msg.timestamp 
    #     # self.get_logger().info(f'Updated Position: x={self.x} y={self.y} z={self.z}')
    #     # self.get_logger().info(f'Updated Speed: vx={self.vx} vy={self.vy} vz={self.vz}')
    #     # self.get_logger().info(f'Updated Timestamp: {self.timestamp}')

    def imu_callback(self, msg):
        self.rates = msg

    def imu_attitude_callback(self, msg):
        self.attitude = msg


    # def toEulerZYX(self):
    #     quaternion = Quaternion(x=self.attitude.x, y=self.attitude.y, z=self.attitude.z, w=self.attitude.w)
    #     roll, pitch, yaw = tf_transformations.euler_from_quaternion(quaternion)
    #     return roll, pitch, yaw  # Возвращает кортеж (рад)

    def toEulerZYX(self):
        """
        Преобразует кватернион в углы Эйлера (roll, pitch, yaw).
        
        :return: Объект EulerAngles с аттрибутами roll, pitch, yaw
        """
        if self.attitude is None:
            return None  # Если кватернион не задан
        
        q = [self.attitude.x, self.attitude.y, self.attitude.z, self.attitude.w]
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(q)

        # Возвращаем объект EulerAngles с углами Эйлера
        return EulerAngles(roll, pitch, yaw)
    # def toEulerZYX(self):
    #     """
    #     Преобразует кватернион в углы Эйлера (roll, pitch, yaw).
        
    #     :return: Объект с аттрибутами x (roll), y (pitch), z (yaw)
    #     """
    #     if self.attitude is None:
    #         return None

    #     # Преобразуем кватернион в углы Эйлера с использованием tf2
    #     quaternion = Quaternion(x=self.attitude.x, y=self.attitude.y, z=self.attitude.z, w=self.attitude.w)
        
    #     # Инициализация tf2 Buffer и TransformListener
    #     self.tf_buffer = tf2_ros.Buffer()
    #     self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
    #     # Используем tf2 для преобразования кватерниона в углы Эйлера
    #     euler_angles = tf2_geometry_msgs.do_transform_euler(quaternion)
        
    #     # Создаем объект с аттрибутами x, y, z для углов Эйлера
    #     EulerZYX = type('EulerZYX', (object,), {})  # Динамично создаем класс
    #     euler = EulerZYX()
    #     euler.x = euler_angles.x
    #     euler.y = euler_angles.y
    #     euler.z = euler_angles.z

    #     return euler

    def send_land_command(self):
        land_cmd = VehicleCommand()
        land_cmd.command = VehicleCommand.VEHICLE_CMD_DO_LAND
        land_cmd.param1 = 0  # Приземление (не отменять)
        land_cmd.param2 = 0  # Опция (оставляем по умолчанию)
        land_cmd.param3 = 0  # Нет конкретной цели (текущая позиция)
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

    def publish_position_setpoint(self, x, y, z):
        trajectory_msg = TrajectorySetpoint()
        trajectory_msg.position = [x, y, -z]
        trajectory_msg.timestamp = int(time.time() * 1e6)
        self.trajectory_setpoint_publisher.publish(trajectory_msg)

    def publish_rate_setpoint(self, rates):
        rate_msg = VehicleRatesSetpoint()
        rate_msg.roll = rates.x  # Roll rate
        rate_msg.pitch = rates.y  # Pitch rate
        rate_msg.yaw = rates.z  # Yaw rate
        rate_msg.timestamp = int(time.time() * 1e6)
        self.vehicle_rates_publisher.publish(rate_msg)

    def publish_torque_setpoint(self, roll_torque, pitch_torque, yaw_torque):
        torque_msg = VehicleTorqueSetpoint()
        torque_msg.xyz[0] = roll_torque  # Torque around X-axis
        torque_msg.xyz[1] = pitch_torque  # Torque around Y-axis
        torque_msg.xyz[2] = yaw_torque  # Torque around Z-axis
        torque_msg.timestamp = int(time.time() * 1e6)
        self.vehicle_torque_publisher.publish(torque_msg)

    def publish_attitude_setpoint(self, roll, pitch, yaw):
        attitude_msg = VehicleAttitudeSetpoint()
        attitude_msg.roll_body = roll
        attitude_msg.pitch_body = pitch
        attitude_msg.yaw_body = yaw
        attitude_msg.thrust_body[2] = -self.thrust_target  # Negative Z thrust
        attitude_msg.timestamp = int(time.time() * 1e6)
        self.attitude_setpoint_publisher.publish(attitude_msg)
    
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

    def rotate_vector_by_quaternion(self, vector, quat):
            qx, qy, qz, qw = quat.x, quat.y, quat.z, quat.w
            vx, vy, vz = vector.x, vector.y, vector.z

            tx = 2 * (qy * vz - qz * vy)
            ty = 2 * (qz * vx - qx * vz)
            tz = 2 * (qx * vy - qy * vx)

            result_x = vx + qw * tx + (qy * tz - qz * ty)
            result_y = vy + qw * ty + (qz * tx - qx * tz)
            result_z = vz + qw * tz + (qx * ty - qy * tx)

            return Vector3(x=result_x, y=result_y, z=result_z)

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

    def reset_rate_pid(self): 
        self.publish_rate_setpoint(Vector3(x=0.0, y=0.0, z=0.0))

    def set_offboard_mode(self):
        """Switch to offboard mode."""
        self.offboard_is_active = True
        # self.publish_vehicle_command(
        #     VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        # self.get_logger().info("Switching to offboard mode")
    
    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')
  

    # Должен постоянно отсылаться, чтобы оставаться в offboard
    def offboard_heartbeat(self):
         if self.offboard_is_active:
                self.get_logger().info("Sending SET_MODE OFFBOARD") 
                """Publish the offboard control mode."""
                msg = OffboardControlMode()
                msg.position = True
                msg.velocity = False
                msg.acceleration = False
                msg.attitude = False
                msg.body_rate = False
                msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                self.offboard_control_mode_publisher.publish(msg)

    def update(self):
        #self.get_logger().info(f"flip_stage: {self.flip_stage}")  
        self.get_logger().info(f"flip_stage: {self.flip_stage}")
        # new from base
        if self.flip_stage == FlipStage.INIT:
            self.set_offboard_mode()
            self.arm()
            self.flip_stage = FlipStage.ARMING 
         
        elif self.flip_stage == FlipStage.ARMING:
            self.get_logger().info('FlipStage.ARMING')
            self.get_logger().warn(f"vehicle_status.arming_state: {self.vehicle_status.arming_state}")
            self.get_logger().info(f"self.vehicle_status.nav_state: {self.vehicle_status.nav_state}")
            if self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED:
                self.get_logger().info('ARMING_STATE_ARMED')
                self.flip_stage = FlipStage.TAKEOFF
            else:
                self.set_offboard_mode()
                self.arm()

        
        elif self.flip_stage == FlipStage.TAKEOFF:
            self.get_logger().info('FlipStage.TAKEOFF')
            self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)
            #self.get_logger().info(f"self.vehicle_local_position.z: {self.vehicle_local_position.z} self.takeoff_height: {self.takeoff_height} res:{self.vehicle_local_position.z  + 0.2 <= -self.takeoff_height}") 
            if self.vehicle_local_position.z  - 0.5 <= -self.takeoff_height:
                self.flip_stage = FlipStage.READY_FOR_FLIP
                self.get_logger().info('FlipStage.READY_FOR_FLIP')

        elif self.flip_stage == FlipStage.READY_FOR_FLIP:
            self.flip_active = True
            euler_angles = self.toEulerZYX()# wait for ideal conditions
            self.get_logger().info(f"euler_angles[{euler_angles.roll},{euler_angles.pitch}, {euler_angles.yaw}] self.rates.x:{self.rates.x} val:{abs(self.rates.x) < 0.05 and (abs(euler_angles.roll) < 0.02)}")
            if euler_angles != None:
                if (abs(self.rates.x) < 0.05 and (abs(euler_angles.roll) < 0.02)):
                    self.flip_stage = FlipStage.BOUNCE
                    self.stage_time =  time.time()

        elif self.flip_stage == FlipStage.BOUNCE:
            # msg = TrajectorySetpoint()
            # msg.timestamp = int(time.time() * 1e6)
            # msg.velocity = [0.0, 0.0, 0.5]
            self.thrust_target = 0.8  # Устанавливаем желаемую тягу, отправляется с publish_attitude_setpoint
            attitude_target = self.quaternion_from_euler(0, 0, self.attitude.yaw)  # Установка целевой ориентации
            self.publish_attitude_setpoint(attitude_target)  # Управление ориентацией
            self.publish_rate_setpoint(self.rates)  # Управление угловыми скоростями, self.rates обновл в callback
            self.publish_torque_setpoint(self.rates)  # Управление крутящими моментами
            #controlAttitude()
		    #controlRate()
		    #controlTorque()
            #self.trajectory_setpoint_publisher.publish(msg)
            if t - self.stage_time > BOUNCE_TIME:
                self.flip_stage = FlipStage.ACCELERATE
                self.stage_time = t


            # else:
            #     self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)  # Поддержание высоты
            #     self.publish_attitude_setpoint(-roll * 0.1, 0.0, 0.0)  # Стабилизация крена

        # new from base
        #  change
        # """Основная логика управления."""
        # if self.vehicle_status is None:
        #     return
        # t = time.time()

        # if self.flip_stage == FlipStage.DISARMED:
        #     self.get_logger().info("DISARMED")
        #     self.arm()
        #     self.offboard_is_active = True
        #     self.stage_time = t
        #     self.flip_stage = FlipStage.OFFBOARD 

        # elif self.flip_stage == FlipStage.OFFBOARD: 
        #     self.get_logger().info(f"self.vehicle_status.nav_state: {self.vehicle_status.nav_state}") 
        #     if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and self.vehicle_status.arming_state == 2:
        #         self.get_logger().info("Drone armed and in offboard mode")
        #         self.stage_time = t
        #         self.flip_stage = FlipStage.TAKEOFF
        #         self.publish_vehicle_command(400, 1.0)  # Команда ARM
        #     elif t - self.stage_time > OFFBOARD_TIMEOUT:
        #         self.get_logger().warn("OFFBOARD timeout!")
        #         self.flip_stage = FlipStage.DISARMED
        #chage






        # elif self.flip_stage == FlipStage.ARMING:
        #     self.get_logger().info(f"self.vehicle_status.nav_state: {self.vehicle_status.nav_state}")
        #     self.publish_vehicle_command(400, 1.0)  # может не нужно пулинг команды арм
        #     if self.vehicle_status.arming_state == 2:  
        #         self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)
        #         self.flip_stage = FlipStage.TAKEOFF
        #         self.stage_time = t
        #     elif t - self.stage_time > ARM_TIMEOUT:
        #         self.get_logger().warn("ARM timeout!")
        #         self.flip_stage = FlipStage.TAKEOFF

        
        # elif self.flip_stage == FlipStage.WAIT:
        #     euler_angles = self.toEulerZYX()# wait for ideal conditions
        #     if euler_angles != None:
        #         if (abs(self.rates.x) < 0.05 and (abs(euler_angles.x) < 0.02)):
        #             self.flip_stage = FlipStage.BOUNCE
        #             self.stage_time = t
        #     # else:
        #     #     self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)  # Поддержание высоты
        #     #     self.publish_attitude_setpoint(-roll * 0.1, 0.0, 0.0)  # Стабилизация крена
		
        # elif self.flip_stage == FlipStage.BOUNCE:
        #     # msg = TrajectorySetpoint()
        #     # msg.timestamp = int(time.time() * 1e6)
        #     # msg.velocity = [0.0, 0.0, 0.5]
        #     self.thrust_target = 0.8  # Устанавливаем желаемую тягу, отправляется с publish_attitude_setpoint
        #     attitude_target = self.quaternion_from_euler(0, 0, self.attitude.yaw)  # Установка целевой ориентации
        #     self.publish_attitude_setpoint(attitude_target)  # Управление ориентацией
        #     self.publish_rate_setpoint(self.rates)  # Управление угловыми скоростями, self.rates обновл в callback
        #     self.publish_torque_setpoint(self.rates)  # Управление крутящими моментами
        #     #controlAttitude()
		#     #controlRate()
		#     #controlTorque()
        #     #self.trajectory_setpoint_publisher.publish(msg)
        #     if t - self.stage_time > BOUNCE_TIME:
        #         self.flip_stage = FlipStage.ACCELERATE
        #         self.stage_time = t

        # elif self.flip_stage == FlipStage.ACCELERATE:
        #     self.thrust_target = 0.1  # Уменьшенная тяга, передается в publish_attitude_setpoint

        #     # Управление моментами для ускорения
        #     roll_torque = 1.0  # Ускорение вокруг оси крена
        #     pitch_torque = 0.0  # Нет ускорения по оси тангажа
        #     yaw_torque = 0.0  # Нет ускорения по оси рыскания

        #     # Публикуем команды для управления моментами
        #     self.publish_torque_setpoint(roll_torque, pitch_torque, yaw_torque)

        #     # Обнуление углов ориентации
        #     self.publish_attitude_setpoint(0.0, 0.0, 0.0)

        #     # Обнуляем угловые скорости
        #     self.publish_rate_setpoint(Vector3(x=0.0, y=0.0, z=0.0))

        #     # Включаем моторы (управление моторами)
        #     self.set_motor_commands(
        #         motor_rear_left=1,
        #         motor_rear_right=0,
        #         motor_front_left=1,
        #         motor_front_right=0
        #     )

        #     # Логируем состояние моторов
        #     #self.get_logger().info(f"Motors: {self.motors}")

        #     if t - self.stage_time > 0.3:  # ACCELERATE_TIME
        #         self.flip_stage = FlipStage.ROTATE
        #         self.stage_time = t

        # elif self.flip_stage == FlipStage.ROTATE:
        #     self.thrust_target = 0.2  # Уменьшенная тяга для свободного вращения
        #     self.publish_attitude_setpoint(0.0, 0.0, 0.0)  # Убрать управление углами
        #     self.publish_torque_setpoint(0.0, 0.0, 0.0)  # Обнуление моментов
        #     self.publish_rate_setpoint(Vector3(x=0.0, y=0.0, z=0.0))
        #     self.set_motor_commands(
        #         motor_rear_left=0, 
        #         motor_rear_right=0, 
        #         motor_front_left=0, 
        #         motor_front_right=0)

        #     up_vector = self.rotate_vector_by_quaternion(Vector3(x=0, y=0, z=-1), self.attitude)

        #     if abs(self.rates.x) > abs(self.max_rate):
        #         self.max_rate = self.rates.x
        #         self.acceleration_angle_z = up_vector.z
        #         self.acceleration_when_rotating_time = t - self.stage_time

        #     if up_vector.z < self.acceleration_angle_z:
        #         self.flip_stage = FlipStage.BRAKE
        #         self.stage_time = t

        # elif self.flip_stage == FlipStage.BRAKE:
        #     self.thrust_target = 0.3  # Установка тяги для торможения
        #     # Публикация команд для контроля тяги и моментов
        #     self.publish_attitude_setpoint(0.0, 0.0, 0.0)  # Обнуление углов
        #     self.publish_rate_setpoint(Vector3(x=0.0, y=0.0, z=0.0))  # Стабилизация угловых скоростей
        #     self.publish_torque_setpoint(0.0, 0.0, 0.0)  # Обнуление моментов
        #     # Управление моторами для торможения
        #     self.set_motor_commands(
        #         motor_rear_left=0, 
        #         motor_rear_right=1, 
        #         motor_front_left=0, 
        #         motor_front_right=1)

        #     if t - self.stage_time > BRAKE_TIME:  # Проверка времени торможения
        #         self.flip_stage = FlipStage.POST_BRAKE  # Переход к следующему этапу
        #         self.stage_time = t  # Обновление времени для следующего этапа

        # elif self.flip_stage == FlipStage.POST_BRAKE:
        #     self.thrust_target = 0.4
        #     self.publish_attitude_setpoint(0.0, 0.0, 0.0)  # Обнуление углов
        #     self.set_motor_commands(
        #         motor_rear_left=0.3, 
        #         motor_rear_right=0.3, 
        #         motor_front_left=0.3, 
        #         motor_front_right=0.3)
            
        #     if t - self.stage_time > self.acceleration_when_rotating_time:
        #         self.flip_stage = FlipStage.RECOVERY
        #         self.stage_time = t
        #         #self.reset_rate_pid()
        #         self.publish_rate_setpoint(Vector3(x=0.0, y=0.0, z=0.0))
        #         # rollRatePID.reset()
		# 	    # pitchRatePID.reset()
		# 	    # yawRatePID.reset()

        # elif self.flip_stage == FlipStage.RECOVERY:
        #     # Переводим дрон в стабилизационный режим
        #     self.set_offboard_control_mode(False)  # Отключаем оффборд-режим и включаем стабилизацию
        #     self.offboard_is_active = False
        #     self.set_stabilization_mode()
        #     self.publish_attitude_setpoint(0.0, 0.0, 0.0)  # Обнуляем углы для стабилизации
        #     self.publish_rate_setpoint(Vector3(x=0.0, y=0.0, z=0.0))  # Обнуляем углы скорости
        #     self.publish_thrust_setpoint(0.0)  # Начальная тяга для стабилизации


        #     # Увеличиваем тягу в первые 0.4 секунды для стабилизации
        #     if t - self.stage_time < 0.4:
        #         self.thrust_target = 0.8
        #         self.publish_thrust_setpoint(self.thrust_target)  # Отправляем команду для увеличения тяги

        #     # После первых 0.4 секунд можем обновить режим или другие параметры
        #     if t - self.stage_time > 0.4:
        #         # Продолжаем обработку дрона, например, переключаемся на следующий этап
        #         self.flip_stage = FlipStage.LAND
        #         self.stage_time = t
        # elif self.flip_stage == FlipStage.LAND:
        #     self.send_land_command()



def main(args=None):
    rclpy.init(args=args)
    node = FlipControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
