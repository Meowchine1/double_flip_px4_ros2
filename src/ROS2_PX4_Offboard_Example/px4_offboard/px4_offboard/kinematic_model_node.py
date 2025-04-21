import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from px4_msgs.msg import *
from geometry_msgs.msg import PoseStamped, Vector3
from rclpy.time import Time
import numpy as np
from scipy.spatial.transform import Rotation as R
from px4_msgs.msg import (VehicleAttitude, VehicleImu, ActuatorOutputs, 
                          VehicleLocalPosition,SensorCombined,VehicleAngularVelocity,
                          VehicleAngularAccelerationSetpoint, VehicleMagnetometer)

# ===== MATRIX OPERTIONS =====
# ===== QUATERNION UTILS (SCIPY-based) =====

def quat_multiply(q1, q2):
    """Multiply two quaternions [x, y, z, w] using scipy."""
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    return (r1 * r2).as_quat()

def normalize_quat(q):
    """Normalize a quaternion."""
    return R.from_quat(q).as_quat()

def quat_derivative(q, omega):
    """Compute quaternion derivative from angular velocity."""
    r = R.from_quat(q)
    Omega = R.from_rotvec(omega * 0.5)
    dq = (Omega * r).as_quat()  # Small-angle rotation
    return dq

def quat_to_matrix(q):
    """Convert quaternion [x, y, z, w] to rotation matrix."""
    return R.from_quat(q).as_matrix()

# ==========

class DroneKinematics(Node):
    def __init__(self):
        super().__init__('drone_kinematics')

        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.imu_position_pub = self.create_publisher(PoseStamped, '/drone/imu_position', self.qos_profile)
        self.imu_pos_err_pub = self.create_publisher(Vector3, '/drone/imu_pos_err', self.qos_profile)

        self.create_subscription(SensorCombined, '/fmu/out/sensor_combined', self.sensor_combined_callback, self.qos_profile)
        self.create_subscription(VehicleAngularVelocity, '/fmu/out/vehicle_angular_velocity', self.angular_velocity_callback, self.qos_profile)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.vehicle_attitude_callback, self.qos_profile)
        self.create_subscription(VehicleAngularAccelerationSetpoint, '/fmu/out/vehicle_angular_acceleration_setpoint', self.vehicle_angular_acceleration_setpoint_callback, self.qos_profile)
        self.create_subscription(VehicleImu, '/fmu/out/vehicle_imu', self.vehicle_imu_callback, self.qos_profile)
        self.create_subscription(ActuatorOutputs, '/fmu/out/actuator_outputs', self.actuator_outputs_callback, self.qos_profile)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, self.qos_profile)

        # FOR EKF
        self.create_subscription(VehicleMagnetometer, '/fmu/out/vehicle_magnetometer', self.vehicle_magnetometer_callback, self.qos_profile)

        self._init_state()

        self.timer = self.create_timer(0.1, self.kinematics_computing)

    def _init_state(self):
        self.sensorCombined_q = [0.0, 0.0, 0.0, 1.0]
        self.vehicleAttitude_q = [0.0, 0.0, 0.0, 1.0]
        self.q = [0.0, 0.0, 0.0, 1.0]

        self.sensorCombined_delta_angle = np.zeros(3, dtype=np.float32)
        self.vehicleImu_delta_angle = np.zeros(3, dtype=np.float32)
        self.vehicleImu_delta_velocity = np.zeros(3, dtype=np.float32)

        self.sensorCombined_angular_velocity = np.zeros(3, dtype=np.float32)
        self.angularVelocity = np.zeros(3, dtype=np.float32)
        self.vehicleImu_angular_velocity = np.zeros(3, dtype=np.float32)

        self.angularVelocity_angular_acceleration = np.zeros(3, dtype=np.float32)
        self.vehiclAngularAcceleration_angular_acceleration = np.zeros(3, dtype=np.float32)

        self.vehicleLocalPosition_position = np.zeros(3, dtype=np.float32)
        self.imu_position = np.zeros(3, dtype=np.float32)
        self.vehicleImu_velocity = np.zeros(3, dtype=np.float32)

        self.magnetometer_data = np.zeros(3, dtype=np.float32)

        self.motor_inputs = np.zeros(4, dtype=np.float32)
        self.max_speed = 2100.0

        self.x = np.zeros(9)
        self.P = np.eye(9)
        self.Q = np.eye(9) * 0.1
        self.R = np.eye(6) * 0.5
        self.F = np.eye(9)
        self.H = np.zeros((6, 9))

        self.last_time = Time().seconds_nanoseconds()[0]
    
    def vehicle_magnetometer_callback(self, msg: VehicleMagnetometer):
        self.magnetometer_data = np.array([msg.magnetometer_ga[0], msg.magnetometer_ga[1], msg.magnetometer_ga[2]], dtype=np.float32)

    def vehicle_local_position_callback(self, msg: VehicleLocalPosition):
        self.vehicleLocalPosition_position[:] = [msg.x, msg.y, msg.z]

    def actuator_outputs_callback(self, msg: ActuatorOutputs):
        pwm_outputs = msg.output[:4]
        self.motor_inputs = np.clip((np.array(pwm_outputs) - 1000.0) / 1000.0 * self.max_speed, 0.0, self.max_speed)

    def sensor_combined_callback(self, msg: SensorCombined):
        dt_gyro = msg.gyro_integral_dt * 1e-6
        gyro_rad = np.array(msg.gyro_rad, dtype=np.float32)
        self.sensorCombined_angular_velocity = gyro_rad

        delta_angle = gyro_rad * dt_gyro
        self.sensorCombined_delta_angle = delta_angle

        self.q = quat_multiply(self.q, R.from_rotvec(delta_angle).as_quat())

        self.sensorCombined_linear_acceleration = np.array(msg.accelerometer_m_s2, dtype=np.float32)

    def angular_velocity_callback(self, msg: VehicleAngularVelocity):
        self.angularVelocity = np.array(msg.xyz, dtype=np.float32)
        self.angularVelocity_angular_acceleration = np.array(msg.xyz_derivative, dtype=np.float32)

    def vehicle_attitude_callback(self, msg: VehicleAttitude):
        self.vehicleAttitude_q = np.array([msg.q[1], msg.q[2], msg.q[3], msg.q[0]], dtype=np.float32)

    def vehicle_angular_acceleration_setpoint_callback(self, msg: VehicleAngularAccelerationSetpoint):
        self.vehiclAngularAcceleration_angular_acceleration = msg.xyz

    def vehicle_imu_callback(self, msg: VehicleImu):
        self.vehicleImu_delta_angle = np.array(msg.delta_angle, dtype=np.float32)
        self.vehicleImu_delta_velocity = np.array(msg.delta_velocity, dtype=np.float32)
        delta_angle_dt = msg.delta_angle_dt * 1e-6
        delta_velocity_dt = msg.delta_velocity_dt * 1e-6

        if delta_angle_dt > 0:
            self.vehicleImu_angular_velocity = self.vehicleImu_delta_angle / delta_angle_dt
        if delta_velocity_dt > 0:
            accel_body = self.vehicleImu_delta_velocity / delta_velocity_dt
            self.vehicleImu_linear_acceleration = accel_body

            rotation = R.from_quat(self.q)
            accel_world = rotation.apply(accel_body)
            gravity = np.array([0, 0, 9.81])
            accel_corrected = accel_world - gravity
            self.vehicleImu_velocity += accel_corrected * delta_velocity_dt

    def publish_imu_localization_err(self):
        imu_pos = np.array(self.imu_position)
        px4_pos = np.array(self.vehicleLocalPosition_position)
        pos_err = imu_pos - px4_pos

        imu_q = np.array(self.q)
        px4_q = np.array(self.vehicleAttitude_q)
        quat_diff = imu_q - px4_q

        self.get_logger().info(f"imu pose err {pos_err[0]} {pos_err[1]} {pos_err[2]}")
        self.get_logger().info(f"imu quaternion err {quat_diff[0]:.3f} {quat_diff[1]:.3f} {quat_diff[2]:.3f} {quat_diff[3]:.3f}")

        err_msg = Vector3()
        err_msg.x = float(pos_err[0])
        err_msg.y = float(pos_err[1])
        err_msg.z = float(pos_err[2])
        self.imu_pos_err_pub.publish(err_msg)

    def publish_imu_localization(self):
        current_time = Time().seconds_nanoseconds()[0]
        dt = current_time - self.last_time
        self.last_time = current_time

        omega = self.vehicleImu_angular_velocity
        q = np.array(self.q)
        q = np.roll(q, 1)

        omega_quat = np.array([0.0, *omega])
        q_dot = 0.5 * quat_multiply(q, omega_quat)
        q_new = q + q_dot * dt
        q_new = q_new / np.linalg.norm(q_new)
        q_new = np.roll(q_new, -1)

        self.q = q_new.tolist()

        a_body = self.sensorCombined_linear_acceleration
        R_mat = quat_to_matrix(np.roll(q_new, 1))
        a_world = R_mat @ a_body

        g = np.array([0.0, 0.0, -9.81])
        a_world += g

        self.vehicleImu_velocity += a_world * dt
        self.imu_position += self.vehicleImu_velocity * dt

        self.get_logger().info(
            f"publish_imu_localization pose: {self.imu_position[0]:.3f} {self.imu_position[1]:.3f} {self.imu_position[2]:.3f}  "
            f"quaternion: {self.q[0]:.3f} {self.q[1]:.3f} {self.q[2]:.3f} {self.q[3]:.3f}")
        

    def kinematics_computing(self):
        self.publish_imu_localization()
        self.publish_imu_localization_err()

def main(args=None):
    rclpy.init(args=args)
    node = DroneKinematics()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
