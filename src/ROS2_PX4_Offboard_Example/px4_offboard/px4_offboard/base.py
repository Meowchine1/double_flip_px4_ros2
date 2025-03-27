# offboard_control.py

from pathlib import Path
import sys
import rclpy
from rclpy.node import Node
from mavros_msgs.srv import SetMode
from mavros_msgs.msg import State

from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, SensorCombined, VehicleMagnetometer
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from mavros_msgs.msg import AttitudeTarget
import sys
#sys.path.append('/home/vor/ros2_px4_offboard_example_ws/src/ROS2_PX4_Offboard_Example/px4_offboard')
print(sys.path)
# import px4_offboard.double_flip
#from px4_offboard import DoubleFlip
#from double_flip import DoubleFlip
from enum import Enum
from datetime import datetime
import os
import time
import math

from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.srv import CommandTOL, SetMode
from sensor_msgs.msg import Imu
# class DoubleFlip(Node):
#     def __init__(self):
#         super().__init__('double_flip')
#         qos_profile = QoSProfile(
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             durability=DurabilityPolicy.TRANSIENT_LOCAL,
#             history=HistoryPolicy.KEEP_LAST,
#             depth=1
#         )
      

class DroneState(Enum):
    INIT = 0
    ARMING = 1
    TAKEOFF = 2
    READY_FOR_FLIP = 3
    FLIPPING = 4
    LANDING = 5
    DISARMED = 6

class BaseControl(Node):
    """Node for controlling a vehicle in offboard mode."""
    def __init__(self) -> None:
        super().__init__('Base_control')
        # Log paths
        self.log_position_path = '/home/vor/ros2_px4_offboard_example_ws/src/ROS2_PX4_Offboard_Example/log/position'
        self.log_accel_path = '/home/vor/ros2_px4_offboard_example_ws/src/ROS2_PX4_Offboard_Example/log/sensors/accel'
        self.log_gyro_path = '/home/vor/ros2_px4_offboard_example_ws/src/ROS2_PX4_Offboard_Example/log/sensors/gyro'
        self.log_mag_path = '/home/vor/ros2_px4_offboard_example_ws/src/ROS2_PX4_Offboard_Example/log/sensors/mag'
        os.makedirs(self.log_position_path, exist_ok=True)
        self.log_filename = os.path.join(self.log_position_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log')
        self.get_logger().warn(f"Log position path: {self.log_filename}")
        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # Publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.publisher = self.create_publisher(AttitudeTarget, '/mavros/setpoint_raw/attitude', 10)
        # Subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        # Подписываемся на топики гироскопа, акселерометра и магнитометра
        self.create_subscription(SensorCombined, '/fmu/out/sensor_combined', self.sensor_combined_callback, qos_profile)
        self.create_subscription(VehicleMagnetometer, '/fmu/out/vehicle_magnetometer', self.magnetometer_callback, qos_profile)
        # Initialize variables
        self.sensors = SensorCombined()
        self.mag = VehicleMagnetometer()
        self.accel_x = 0
        self.accel_y = 0
        self.accel_z = 0
        self.gyro_x = 0
        self.gyro_y = 0
        self.gyro_z = 0
        self.mag_x = 0
        self.mag_y = 0
        self.mag_z = 0
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.takeoff_height = -5.0
        #for flip
        self.current_roll = 0.0
        self.flip_active = False 
        #command timer
        self.timer = self.create_timer(0.1, self.timer_callback) 
        self.offboard_timer = self.create_timer(0.1, self.offboard_heartbeat) 
        self.state = DroneState.INIT



        # from double flip class
        # Сервисы MAVROS
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.land_client = self.create_client(CommandTOL, '/mavros/cmd/land')
        # Публикация команд
        self.pose_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', qos_profile)
        self.attitude_pub = self.create_publisher(TwistStamped, '/mavros/setpoint_attitude/cmd_vel', qos_profile)
        # Подписка на данные IMU (ориентация)
        self.imu_sub = self.create_subscription(Imu, '/mavros/imu/data', self.imu_callback, qos_profile)
        self.current_roll = 0.0
        self.vehicle_status = VehicleStatus()
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        self.vehicle_local_position = VehicleLocalPosition()

    def log_position(self) -> None:
        # Записываем лог в файл
        if self.vehicle_local_position:
            log_message = f"x={self.vehicle_local_position.x}, y={self.vehicle_local_position.y}, z={self.vehicle_local_position.z}\n"
            with open(self.log_filename, "a") as log_file:
                log_file.write(log_message)
        else:
            self.get_logger().warn("Vehicle local position not available yet.")

    def sensor_combined_callback(self, msg: SensorCombined):
        """Обработчик сообщений с гироскопа и акселерометра.""" 
        self.sensors = msg
        self.accel_x = msg.accelerometer_m_s2[0]
        self.accel_y = msg.accelerometer_m_s2[1]
        self.accel_z = msg.accelerometer_m_s2[2]
        self.gyro_x = msg.gyro_rad[0]
        self.gyro_y = msg.gyro_rad[1]
        self.gyro_z = msg.gyro_rad[2] 

    def magnetometer_callback(self, msg: VehicleMagnetometer):
        """Обработчик сообщений с магнитометра.""" 
        self.mag = msg
        self.mag_x = msg.accelerometer_m_s2[0]
        self.mag_y = msg.accelerometer_m_s2[1]
        self.mag_z = msg.accelerometer_m_s2[2] 

    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = 1.57079  # (90 degree)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        #self.get_logger().info(f"Publishing position setpoints {[x, y, z]}")

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg) 

    def imu_callback(self, msg):
        """ Получение ориентации дрона (Roll, Pitch, Yaw) """
        q = msg.orientation
        self.current_roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    
    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def send_position(self, x, y, z, yaw=0.0):
        """ Отправка целевой позиции """
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        self.pose_pub.publish(pose)

    def set_attitude_rates(self, roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0, thrust=0.5):
        """ Отправка управляющих команд по угловым скоростям """
        cmd = TwistStamped()
        cmd.twist.angular.x = roll_rate
        cmd.twist.angular.y = pitch_rate
        cmd.twist.angular.z = yaw_rate
        cmd.twist.linear.z = thrust
        self.attitude_pub.publish(cmd)

    def flip(self) -> bool:
        """ Двойной флип (переворот) """
        self.get_logger().info("Starting flip...")

        if self.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.get_logger().error(f"Offboard mode is not active! Current state: {self.vehicle_status.nav_state}")
            return False
        self.get_logger().info(f"Offboard mode current state: {self.vehicle_status.nav_state}")

        if self.vehicle_local_position.z > -0.5:  # Дрон слишком низко
            self.get_logger().error("Altitude too low for flip!")
            return False

        # Первый флип
        self.set_attitude_rates(pitch_rate=30.0, thrust=0.2) #(roll_rate=30.0, thrust=0.8)
        timeout = time.time() + 2  # 2 секунды на выполнение манёвра
        while self.current_roll > -math.pi + 0.1:
            self.get_logger().error(f"self.current_roll: {self.current_roll}")  
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() > timeout:
                self.get_logger().error("First flip failed: timeout")
                return False
        self.get_logger().info("First flip done!")

        # Второй флип
        self.set_attitude_rates(roll_rate=30.0, thrust=0.2)#(roll_rate=-50.0, thrust=0.8)
        timeout = time.time() + 2
        while self.current_roll < -0.1:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() > timeout:
                self.get_logger().error("Second flip failed: timeout")
                return False
        self.get_logger().info("Second flip done!")

        # Восстановление положения
        self.publish_position_setpoint(0, 0, 2)
        self.get_logger().info("Position reset.")
        return True

    def land(self):
        """ Посадка """
        req = CommandTOL.Request()
        self.land_client.call_async(req)
        self.get_logger().info("Landing...")


    def offboard_heartbeat(self) -> None:
        self.publish_offboard_control_heartbeat_signal() # Этот метод должен вызываться периодически (обычно 10 Гц), иначе дрон автоматически отключит Offboard-режим
        #self.get_logger().info('publish_offboard_control_heartbeat_signal')

    def timer_callback(self) -> None:   # move drone from "off" to "ready to flip"
        #self.publish_offboard_control_heartbeat_signal() # Этот метод должен вызываться периодически (обычно 10 Гц), иначе дрон автоматически отключит Offboard-режим

        if self.state == DroneState.INIT:
            self.engage_offboard_mode()
            self.arm()
            self.state = DroneState.ARMING
            self.get_logger().warn(f"vehicle_status.arming_state: {self.vehicle_status.arming_state}")

         
        elif self.state == DroneState.ARMING:
            self.get_logger().info('DroneState.ARMING')
            self.get_logger().warn(f"vehicle_status.arming_state: {self.vehicle_status.arming_state}")
            if self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED:
                self.get_logger().info('ARMING_STATE_ARMED')
                self.state = DroneState.TAKEOFF
            else:
                self.engage_offboard_mode()
                self.arm()

        
        elif self.state == DroneState.TAKEOFF:
            self.get_logger().info('DroneState.TAKEOFF')
            self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)
            self.get_logger().info(f"self.vehicle_local_position.z: {self.vehicle_local_position.z}") 
            if self.vehicle_local_position.z  - 0.2 <= self.takeoff_height:
                self.state = DroneState.READY_FOR_FLIP
                self.get_logger().info('DroneState.READY_FOR_FLIP')

        elif self.state == DroneState.READY_FOR_FLIP:
            self.flip_active = True
            self.state = DroneState.FLIPPING

        elif self.state == DroneState.FLIPPING:
            self.get_logger().info(f"Offboard mode current state: {self.vehicle_status.nav_state}")
            if self.flip():
                self.state = DroneState.LANDING

        elif self.state == DroneState.LANDING:
            self.land()
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_LAND:
                self.state = DroneState.DISARMED

        elif self.state == DroneState.DISARMED:
            self.disarm()
            rclpy.shutdown()


        self.offboard_setpoint_counter += 1


def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    #nodes
    base_control = BaseControl()

    #loops
    rclpy.spin(base_control)

    #on exit
    base_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)