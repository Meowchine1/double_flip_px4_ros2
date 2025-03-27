import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Quaternion
from mavros_msgs.msg import AttitudeTarget
import math

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

 