#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, SensorCombined, VehicleMagnetometer

import os
from datetime import datetime

#flip
from geometry_msgs.msg import Quaternion
from mavros_msgs.msg import AttitudeTarget
import math
from rclpy.duration import Duration
# how import?
#from quaternion import quaternion_from_euler 

def quaternion_from_euler(roll, pitch, yaw):
    """Конвертация углов в кватернионы"""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    return Quaternion(
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
        w=cr * cp * cy + sr * sp * sy
    )

class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
        super().__init__('offboard_control_takeoff_and_land')

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
        #self.timer = self.create_timer(0.1, self.publish_attitude)  # Таймер для публикации сообщений


        #command timer
        self.timer = self.create_timer(0.1, self.timer_callback)

 
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
        # self.get_logger().info(
        #     f"Gyroscope: x={msg.gyro_rad[0]:.6f}, y={msg.gyro_rad[1]:.6f}, z={msg.gyro_rad[2]:.6f} | "
        #     f"Accelerometer: x={msg.accelerometer_m_s2[0]:.6f}, y={msg.accelerometer_m_s2[1]:.6f}, z={msg.accelerometer_m_s2[2]:.6f}"
        # )
        self.sensors = msg

        self.accel_x = msg.accelerometer_m_s2[0]
        self.accel_y = msg.accelerometer_m_s2[1]
        self.accel_z = msg.accelerometer_m_s2[2]

        self.gyro_x = msg.gyro_rad[0]
        self.gyro_y = msg.gyro_rad[1]
        self.gyro_z = msg.gyro_rad[2]

        # self.get_logger().info(
        #     f"Gyroscope: x={self.gyro_x:.6f}, y={self.gyro_y:.6f}, z={self.gyro_z:.6f} | "
        #     f"Accelerometer: x={self.accel_x:.6f}, y={self.accel_y:.6f}, z={self.accel_z:.6f}"
        # )


    def magnetometer_callback(self, msg: VehicleMagnetometer):
        """Обработчик сообщений с магнитометра."""
        # self.get_logger().info(
        #     f"Magnetometer: x={msg.magnetometer_ga[0]:.6f}, y={msg.magnetometer_ga[1]:.6f}, z={msg.magnetometer_ga[2]:.6f}"
        # )
        self.mag = msg
        self.mag_x = msg.accelerometer_m_s2[0]
        self.mag_y = msg.accelerometer_m_s2[1]
        self.mag_z = msg.accelerometer_m_s2[2]

        # self.get_logger().info(
        #     f"Magnetometer: x={self.mag_x:.6f}, y={self.mag_y:.6f}, z={self.mag_z:.6f}"
        # )

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
        self.get_logger().info(f"Publishing position setpoints {[x, y, z]}")

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
        
    def stop_flip(self):
        """Останавливает флип и возвращает дрон в нормальное положение."""
        self.get_logger().info("Stopping flip, stabilizing drone...")

        stable_attitude = AttitudeTarget()
        stable_attitude.type_mask = 0  # Управление углом
        stable_attitude.orientation = quaternion_from_euler(0.0, 0.0, 0.0)  # Выравниваем дрон

        for _ in range(20):  
            self.publisher.publish(stable_attitude)
            self.get_clock().sleep_for(Duration(seconds=0.1))

        self.flip_active = False  # Отключаем флип
        self.get_logger().info("Flip stopped, drone stabilized.")

    def timer_callback(self) -> None:
        self.publish_offboard_control_heartbeat_signal()
        self.log_position()

        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()

        if self.vehicle_local_position.z > self.takeoff_height and \
        self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)

        if 50 < self.offboard_setpoint_counter < 80:  
            self.flip_active = True

        if self.flip_active:
            # Резкое изменение угла для флипа
            att_target = AttitudeTarget()
            att_target.type_mask = 0
            # Установка ориентации для флипа
            att_target.orientation = quaternion_from_euler(math.pi, 0, 0)  # 180° по оси roll для флипа
            self.publisher.publish(att_target)

        if self.offboard_setpoint_counter > 200:
            self.flip_active = False
            self.stop_flip()
            self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)  
            self.land()
            self.disarm()  # Остановить двигатели
            rclpy.shutdown()

        self.offboard_setpoint_counter += 1

        # """Callback function for the timer."""
        # self.publish_offboard_control_heartbeat_signal()
        # #self.get_logger().info(f"timer_callback, local_position: x={self.vehicle_local_position.x}, y={self.vehicle_local_position.y}, z={self.vehicle_local_position.z}")
        # self.log_position()
        # if self.offboard_setpoint_counter == 10:
        #     self.engage_offboard_mode()
        #     self.arm()

        # if self.vehicle_local_position.z > self.takeoff_height and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
        #     self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)

        # elif self.vehicle_local_position.z <= self.takeoff_height:
        #     self.land()
        #     exit(0)

        # if self.offboard_setpoint_counter < 11:
        #     self.offboard_setpoint_counter += 1


def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    #nodes
    offboard_control = OffboardControl()

    #loops
    rclpy.spin(offboard_control)

    #on exit
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)