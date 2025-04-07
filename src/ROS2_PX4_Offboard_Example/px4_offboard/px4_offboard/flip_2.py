import rclpy
from rclpy.node import Node
import time
from px4_msgs.msg import VehicleAttitudeSetpoint, VehicleRatesSetpoint, VehicleOdometry
from geometry_msgs.msg import Quaternion
from tf_transformations import euler_from_quaternion

class FlipController(Node):
    def __init__(self):
        super().__init__('flip_controller')
        self.publisher_rates = self.create_publisher(VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', 10)
        self.publisher_att = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint', 10)
        self.subscription = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, 10)

        self.pitch = 0.0
        self.roll = 0.0
        self.alt = 0.0

    def odom_callback(self, msg):
        q = msg.q  # quaternion [w, x, y, z]
        euler = euler_from_quaternion([q[1], q[2], q[3], q[0]])
        self.roll = euler[0] * 180.0 / 3.14159
        self.pitch = euler[1] * 180.0 / 3.14159
        self.alt = msg.position[2]

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

    def flip_roll(self):
        while self.alt < 6.0:
            self.set_thrust(1.0)

        state = 'INIT'
        while state != 'TURNED_315':
            self.set_rates(360.0, 0.0, 0.0, 0.25)
            time.sleep(0.01)

            if state == 'INIT' and self.roll > 45.0:
                state = 'TURNED_45'
            elif state == 'TURNED_45' and -45.0 < self.roll < 0.0:
                state = 'TURNED_315'

        while abs(self.roll) > 3.0:
            self.set_thrust(0.6)
            time.sleep(0.01)

    def flip_pitch(self):
        while self.alt < 10.0:
            self.set_thrust(1.0)

        state = 'INIT'
        while state != 'TURNED_315':
            self.set_rates(0.0, 360.0, 0.0, 0.25)
            time.sleep(0.01)

            if state == 'INIT' and self.pitch > 45.0:
                state = 'TURNED_45'
            elif state == 'TURNED_45' and self.pitch < -60.0:
                state = 'TURNED_240'
            elif state == 'TURNED_240' and -45.0 < self.pitch < 0.0:
                state = 'TURNED_315'

        while abs(self.pitch) > 3.0:
            self.set_thrust(0.6)
            time.sleep(0.01)

def main(args=None):
    rclpy.init(args=args)
    flip_controller = FlipController()
    time.sleep(1.0)  # give time to receive odometry

    flip_controller.flip_pitch()  # or flip_controller.flip_roll()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
