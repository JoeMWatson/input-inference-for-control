import autograd.numpy as np
from autograd import jacobian


def pendulum_dynamics(xu):
    dt = 0.05
    m = 1.0
    l = 1.0
    d = 1e-2  # damping
    g = 9.80665
    u_mx = 2.0
    x, u = xu[:, :2], xu[:, 2:]
    u = np.clip(u, -u_mx, u_mx)
    th_dot_dot = -3.0 * g / (2 * l) * np.sin(x[:, 0] + np.pi) - d * x[:, 1]
    th_dot_dot += 3.0 / (m * l ** 2) * u.squeeze()
    x_dot = x[:, 1] + th_dot_dot * dt
    x_pos = x[:, 0] + x_dot * dt
    x2 = np.vstack((x_pos, x_dot)).T
    return x2


pendulum_dydxu = jacobian(pendulum_dynamics, 0)


def cartpole_dynamics(xu):

    g = 9.81
    Mc = 0.37
    Mp = 0.127
    Mt = Mc + Mp
    l = 0.3365
    fs_hz = 250.0
    dt = 1 / fs_hz
    u_mx = 5.0
    x, u = xu[:, :4], xu[:, 4:]
    _u = np.clip(u, -u_mx, u_mx).squeeze()

    th = x[:, 1]
    dth2 = np.power(x[:, 3], 2)
    sth = np.sin(th)
    cth = np.cos(th)

    _num = -Mp * l * sth * cth * dth2 + Mt * g * sth - _u * cth
    _denom = l * ((4.0 / 3.0) * Mt - Mp * cth ** 2)
    th_acc = _num / _denom
    x_acc = (Mp * l * sth * dth2 - Mp * l * th_acc * cth + _u) / Mt

    y1 = x[:, 0] + dt * x[:, 2]
    y2 = x[:, 1] + dt * x[:, 3]
    y3 = x[:, 2] + dt * x_acc
    y4 = x[:, 3] + dt * th_acc

    y = np.vstack((y1, y2, y3, y4)).T
    return y


cartpole_dydxu = jacobian(cartpole_dynamics, 0)


def double_cartpole_dynamics(xu):
    """
    http://www.lirmm.fr/~chemori/Temp/Wafa/double%20pendule%20inverse.pdf
    """
    fs_hz = 125
    dt = 1 / fs_hz
    g = 9.81
    Mc = 0.37
    Mp1 = 0.127
    Mp2 = 0.127
    Mt = Mc + Mp1 + Mp2
    L1 = 0.3365
    L2 = 0.3365
    l1 = L1 / 2
    l2 = L2 / 2
    J1 = Mp1 * L1 / 12
    J2 = Mp2 * L2 / 12
    u_mx = 10.0
    input_amp = 3.0  # simulate gear ratio etc

    x, u = xu[:, :6], xu[:, 6:]
    N = x.shape[0]

    q = x[:, 0]
    th1 = x[:, 1]
    th2 = x[:, 2]
    q_dot = x[:, 3]
    th_dot1 = x[:, 4]
    th_dot2 = x[:, 5]

    sth1 = np.sin(th1)
    cth1 = np.cos(th1)
    sth2 = np.sin(th2)
    cth2 = np.cos(th2)
    sdth = np.sin(th1 - th2)
    cdth = np.cos(th1 - th2)

    # helpers
    l1_mp1_mp2 = Mp1 * l1 + Mp2 * L2
    l1_mp1_mp2_cth1 = l1_mp1_mp2 * cth1
    Mp2_l2 = Mp2 * l2
    Mp2_l2_cth2 = Mp2_l2 * cth2
    l1_l2_Mp2 = L1 * l2 * Mp2
    l1_l2_Mp2_cdth = l1_l2_Mp2 * cdth

    # inertia
    M11 = Mt * np.ones((N,))
    M12 = l1_mp1_mp2_cth1
    M13 = Mp2_l2_cth2
    M21 = l1_mp1_mp2_cth1
    M22 = ((l1 ** 2) * Mp1 + (L1 ** 2) * Mp2 + J1) * np.ones((N,))
    M23 = l1_l2_Mp2_cdth
    M31 = Mp2_l2_cth2
    M32 = l1_l2_Mp2_cdth
    M33 = ((l2 ** 2) * Mp2 + J2) * np.ones((N,))

    # coreolis
    C11 = np.zeros((N,))
    C12 = -l1_mp1_mp2 * th_dot1 * sth1
    C13 = -Mp2_l2 * th_dot2 * sth2
    C21 = np.zeros((N,))
    C22 = np.zeros((N,))
    C23 = l1_l2_Mp2 * th_dot2 * sdth
    C31 = np.zeros((N,))
    C32 = -l1_l2_Mp2 * th_dot1 * sdth
    C33 = np.zeros((N,))

    # gravity
    G11 = np.zeros((N,))
    G21 = -(Mp1 * l1 + Mp2 * L1) * g * sth1
    G31 = -Mp2 * l2 * g * sth2

    # make matrices
    M = np.stack(
        (
            np.stack((M11, M21, M31), axis=1),
            np.stack((M12, M22, M32), axis=1),
            np.stack((M13, M23, M33), axis=1),
        ),
        axis=2,
    )

    C = np.stack(
        (
            np.stack((C11, C21, C31), axis=1),
            np.stack((C12, C22, C32), axis=1),
            np.stack((C13, C23, C33), axis=1),
        ),
        axis=2,
    )

    G = np.stack((G11, G21, G31), axis=1)[:, :, None]

    u = input_amp * np.clip(u, -u_mx, u_mx)

    action = np.stack((u, np.zeros(u.shape), np.zeros(u.shape)), axis=1)

    M_inv = np.linalg.inv(M)

    C_x_dot = np.matmul(C, x[:, 3:, None])
    x_dot_dot = np.matmul(M_inv, action - C_x_dot - G).squeeze()

    x_dot = x[:, 3:] + x_dot_dot * dt
    x_pos = x[:, :3] + x_dot * dt

    x2 = np.hstack((x_pos, x_dot))

    return x2


double_cartpole_dydxu = jacobian(double_cartpole_dynamics, 0)
