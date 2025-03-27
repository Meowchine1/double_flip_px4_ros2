
#new
# before
# import math
# import rospy
# from clover import srv
# from std_srvs.srv import Trigger
# from sensor_msgs.msg import Range
# from mavros_msgs.srv import SetMode  

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.srv import CommandTOL, SetMode
from sensor_msgs.msg import Imu
from rclpy.qos import QoSProfile


class DoubleFlip(Node):
    def __init__(self):
        super().__init__('double_flip')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Сервисы MAVROS
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.land_client = self.create_client(CommandTOL, '/mavros/cmd/land')

        # Публикация команд
        self.pose_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', qos_profile)
        self.attitude_pub = self.create_publisher(TwistStamped, '/mavros/setpoint_attitude/cmd_vel', qos_profile)

        # Подписка на данные IMU (ориентация)
        self.imu_sub = self.create_subscription(Imu, '/mavros/imu/data', self.imu_callback, qos_profile)

        self.current_roll = 0.0

    def imu_callback(self, msg):
        """ Получение ориентации дрона (Roll, Pitch, Yaw) """
        q = msg.orientation
        self.current_roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))

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
            self.get_logger().error("Offboard mode is not active!")
            return False

        if self.vehicle_local_position.z > -0.5:  # Дрон слишком низко
            self.get_logger().error("Altitude too low for flip!")
            return False

        # Первый флип
        self.set_attitude_rates(roll_rate=30, thrust=0.8)
        timeout = time.time() + 2  # 2 секунды на выполнение манёвра
        while self.current_roll > -math.pi + 0.1:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() > timeout:
                self.get_logger().error("First flip failed: timeout")
                return False

        self.get_logger().info("First flip done!")

        # Второй флип
        self.set_attitude_rates(roll_rate=-50, thrust=0.8)
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

# def main(args=None):
#     rclpy.init(args=args)
#     drone = DroneFlip()

#     # Взлет
#     drone.send_position(0, 0, 2)
#     rclpy.sleep(3)

#     # Запуск двойного флипа
#     drone.flip()
#     rclpy.sleep(3)

#     # Посадка
#     drone.land()
#     rclpy.spin()

# if __name__ == '__main__':
#     main()



# rospy.init_node('fly')
# get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry, persistent=True)
# navigate = rospy.ServiceProxy('navigate', srv.Navigate)
# set_position = rospy.ServiceProxy('set_position', srv.SetPosition)
# set_rates = rospy.ServiceProxy('set_rates', srv.SetRates)
# land = rospy.ServiceProxy('land', Trigger)
# set_mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)


# def flip():
#     first_flip = 0
#     second_ready = 0
#     debug_flip = "start"

#     start = get_telemetry()  # memorize starting position

#     set_rates(thrust=1)  # bump up
#     rospy.sleep(0.2)

#     set_rates(roll_rate=30, thrust=0.2)  # maximum roll rate

#     while True:
#         telem = get_telemetry()
#         a = telem.roll

#         if -math.pi + 0.1 < a < -0.2:
#             first_flip = 1

#         if (math.pi * 0.9) / 6 < a < math.pi and first_flip:
#             print(telem.roll)
#             break

#     while True:
#         telem = get_telemetry()

#         if -math.pi < telem.roll < -0.1 or math.radians(134) < telem.roll < math.pi:
#             set_rates(roll_rate=-50, thrust=0.8)
#             rospy.sleep(0.15)
#             break

#     print 'set position'
#     set_position(x=start.x, y=start.y, z=start.z, yaw=start.yaw)  # finish flip


# navigate(x=0, y=0, z=1.5, speed=1.3, auto_arm=True, frame_id='body')
# rospy.sleep(3)

# navigate(x=0, y=0, z=2, yaw=math.radians(90), speed=1, frame_id='aruco_map')
# rospy.sleep(6)

# flip()

# rospy.sleep(9)

# flip()

# rospy.sleep(6)

# land()