import numpy as np


def euler_to_rotation(pitch, yaw, roll):
    cp, cy, cr = np.cos((pitch, yaw, roll))
    sp, sy, sr = np.sin((pitch, yaw, roll))

    theta = np.zeros((3, 3))

    # front
    theta[0, 0] = cp * cy
    theta[1, 0] = cp * sy
    theta[2, 0] = sp

    # left
    theta[0, 1] = cy * sp * sr - cr * sy
    theta[1, 1] = sy * sp * sr + cr * cy
    theta[2, 1] = -cp * sr

    # up
    theta[0, 2] = -cr * cy * sp - sr * sy
    theta[1, 2] = -cr * sy * sp + sr * cy
    theta[2, 2] = cp * cr

    return theta
