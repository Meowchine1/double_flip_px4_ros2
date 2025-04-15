import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import VehicleAttitude, ActuatorOutputs, VehicleLocalPosition
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

        # DRONE PARAMETERS
        self.mass = 0.82
        self.inertia = np.diag([0.045, 0.045, 0.045])
        self.arm_length = 0.15
        self.k_thrust = 1.48e-6
        self.k_torque = 9.4e-8
        self.motor_tau = 0.02
        self.max_speed = 2100.0
        self.drag = 0.1
        self.max_rate = 25.0  # рад/с, ограничение на угловую скорость (roll/pitch)

        self.state = np.zeros(12, dtype=float)  # [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        self.motor_speeds = np.zeros(4)
        self.orientation_q = [0.0, 0.0, 0.0, 1.0]

        #SUBSCRIBERS
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.vehicle_attitude_callback, qos_profile)
        self.create_subscription(ActuatorOutputs, '/fmu/out/actuator_outputs', self.actuator_outputs_callback, qos_profile)
        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)

        #PUBLISHERS
        self.pose_pub = self.create_publisher(PoseStamped, '/quad/pose_pred', qos_profile)
        self.motor_pub = self.create_publisher(Float32MultiArray, '/fmu/motor_speeds', qos_profile)

        #TIMERS
        self.timer = self.create_timer(0.01, self.step_dynamics)

    def actuator_outputs_callback(self, msg):
        pwm_outputs = msg.output[:4]  # предполагаем, что 0-3 — это моторы
        # Преобразование PWM в радианы в секунду (линейное приближение)
        self.motor_speeds = np.clip((np.array(pwm_outputs) - 1000.0) / 1000.0 * self.max_speed, 0.0, self.max_speed)

    def vehicle_attitude_callback(self, msg):
        self.orientation_q = list(map(float, msg.q))
        # Преобразуем кватернион в углы Эйлера
        rotation = R.from_quat(self.orientation_q)
        euler_angles = rotation.as_euler('xyz', degrees=False)
        roll, pitch, yaw = euler_angles
        
        # Обновляем состояние с углами Эйлера
        self.state[6] = roll
        self.state[7] = pitch
        self.state[8] = yaw

    def vehicle_local_position_callback(self, msg):
        self.state[0:3] = [msg.x, msg.y, msg.z]
        self.state[3:6] = [msg.vx, msg.vy, msg.vz]

    def step_dynamics(self):
        #self.get_logger().info(f"step_dynamics")

        dt = 0.01
        vel = self.state[3:6]
        ang_vel = self.state[9:12]

        # Сила тяги в теле
        thrusts = self.k_thrust * self.motor_speeds ** 2
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
        self.publish_motor_speeds()
        

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

    def publish_motor_speeds(self):
        msg = Float32MultiArray()
        msg.data = self.motor_speeds.tolist()
        self.motor_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
