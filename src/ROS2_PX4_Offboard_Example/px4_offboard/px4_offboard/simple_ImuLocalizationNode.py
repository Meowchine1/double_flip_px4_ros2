import rclpy
from rclpy.node import Node
from px4_msgs.msg import SensorCombined
from px4_msgs.msg import VehicleAttitude
from geometry_msgs.msg import Vector3
import numpy as np
from scipy.spatial.transform import Rotation as R

class ImuLocalizationNode(Node):
    def __init__(self):
        super().__init__('imu_localization_node')

        self.subscription_imu = self.create_subscription(
            SensorCombined,
            '/fmu/out/sensor_combined',
            self.sensor_combined_callback,
            10)

        self.subscription_attitude = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.vehicle_attitude_callback,
            10)

        self.publisher = self.create_publisher(Vector3, 'imu_localization', 10)

        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # Начальная ориентация (единичный кватернион)
        self.velocity = np.zeros(3)             # Начальная скорость (м/с)
        self.position = np.zeros(3)             # Начальное положение (м)

        self.prev_time = None

    def vehicle_attitude_callback(self, msg):
        # Обновляем ориентацию
        self.q = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]])

    def sensor_combined_callback(self, msg):
        # Получаем ускорение и время
        acc_body = np.array(msg.accelerometer_m_s2)
        t = self.get_clock().now().nanoseconds * 1e-9

        if self.prev_time is None:
            self.prev_time = t
            return

        dt = t - self.prev_time
        self.prev_time = t

        # Преобразуем ускорение в глобальную систему координат
        r = R.from_quat(self.q)
        acc_world = r.apply(acc_body)

        # Убираем гравитацию
        gravity = np.array([0, 0, 9.81])
        acc_world -= gravity

        # Интегрируем ускорение -> скорость
        self.velocity += acc_world * dt

        # Интегрируем скорость -> положение
        self.position += self.velocity * dt

        # Публикуем смещение
        displacement_msg = Vector3()
        displacement_msg.x = float(self.position[0])
        displacement_msg.y = float(self.position[1])
        displacement_msg.z = float(self.position[2])
        self.publisher.publish(displacement_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ImuLocalizationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()