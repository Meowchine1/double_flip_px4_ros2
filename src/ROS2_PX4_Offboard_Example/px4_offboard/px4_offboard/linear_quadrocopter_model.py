import numpy as np

def linear_model():
    # Состояние: x = [phi, theta, psi, p, q, r]
    # Управление: u = [Mx, My, Mz] - моменты по осям
    
    I = np.diag([0.045, 0.045, 0.045])
    
    A = np.zeros((6, 6))
    A[0, 3] = 1.0
    A[1, 4] = 1.0
    A[2, 5] = 1.0
    
    B = np.zeros((6, 3))
    B[3:, :] = np.linalg.inv(I)  # d(angular rate)/dt = I^-1 * M

    return A, B