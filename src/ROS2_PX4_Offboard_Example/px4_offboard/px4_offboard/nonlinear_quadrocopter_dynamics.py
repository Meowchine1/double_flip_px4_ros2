import numpy as np
from scipy.spatial.transform import Rotation as R

class MulticopterDynamics:
    def __init__(self):
        self.mass = 0.82
        self.I = np.diag([0.045, 0.045, 0.045])
        self.I_inv = np.linalg.inv(self.I)
        self.g = 9.81
        
        # Состояние
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.q = np.array([1, 0, 0, 0])  # w, x, y, z
        self.omega = np.zeros(3)  # p, q, r

    def update(self, thrust: float, moments: np.ndarray, dt: float):
        # Обновление ориентации
        omega_matrix = np.array([
            [0, -self.omega[0], -self.omega[1], -self.omega[2]],
            [self.omega[0], 0, self.omega[2], -self.omega[1]],
            [self.omega[1], -self.omega[2], 0, self.omega[0]],
            [self.omega[2], self.omega[1], -self.omega[0], 0],
        ])
        dq = 0.5 * omega_matrix @ self.q
        self.q += dq * dt
        self.q /= np.linalg.norm(self.q)

        # Получение матрицы поворота
        rot = R.from_quat([self.q[1], self.q[2], self.q[3], self.q[0]])  # x, y, z, w
        z_body = rot.apply([0, 0, 1])
        F = thrust * z_body

        # Обновление линейного движения
        accel = (F / self.mass) - np.array([0, 0, self.g])
        self.velocity += accel * dt
        self.position += self.velocity * dt

        # Обновление угловой скорости
        domega = self.I_inv @ (moments - np.cross(self.omega, self.I @ self.omega))
        self.omega += domega * dt

    def get_state(self):
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'quaternion': self.q.copy(),
            'omega': self.omega.copy()
        }
