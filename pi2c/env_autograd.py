import autograd.numpy as np
from autograd import jacobian

def pendulum_dynamics(x, u):
    dt = 0.05
    m = 1.
    l = 1.
    d = 1e-2 # damping
    g = 9.80665
    u_mx = 2.
    u = np.clip(u, -u_mx, u_mx)
    th_dot_dot = -3.0 * g / (2 * l) * np.sin(x[0] + np.pi) + d * x[1]
    th_dot_dot += 3.0 / (m * l**2) * u.squeeze()
    x_dot = x[1] + th_dot_dot * dt
    x_pos = x[0] + x_dot * dt
    x2 = np.vstack((x_pos, x_dot)).reshape(x.shape)
    return x2

pendulum_dydx = jacobian(pendulum_dynamics, 0)
pendulum_dydu = jacobian(pendulum_dynamics, 1)

def cartpole_dynamics(x, u):

    g = 9.81
    Mc = 0.37
    Mp = 0.127
    Mt = Mc + Mp
    l = 0.3365
    fs_hz = 500.
    dt = 1 / fs_hz
    u_mx = 5.
    x_mx = 10.

    _u = np.clip(u, -u_mx, u_mx)

    th = x[(1)]
    dth2 = np.power(x[(3)], 2)
    sth = np.sin(th)
    cth = np.cos(th)

    _num = -Mp * l * sth * dth2 + Mt * g * sth -_u * cth
    _denom = l * ((4./3.) * Mt - Mp * cth**2)
    th_acc = _num / _denom
    x_acc = (Mp * l * sth * dth2 - Mp * l * th_acc * cth +_u) / Mt

    y1 = x[(0)] + dt * x[(2)]
    y2 = x[(1)] + dt * x[(3)]
    y3 = x[(2)] + dt * x_acc
    y4 = x[(3)] + dt * th_acc

    # clip x and simulate impact
    # _y1 = np.clip(y1, -x_mx, x_mx)
    # if _y1 != y1:
    #     y3 = -y3
    #     y1 = _y1

    y = np.vstack((y1, y2, y3, y4)).reshape(x.shape)

    return y

cartpole_dydx = jacobian(cartpole_dynamics, 0)
cartpole_dydu = jacobian(cartpole_dynamics, 1)

class QuanserCartpole(object):

    long = False

    g = 9.81              # Gravitational acceleration [m/s^2]
    eta_m = 1.            # Motor efficiency  []
    eta_g = 1.            # Planetary Gearbox Efficiency []
    Kg = 3.71             # Planetary Gearbox Gear Ratio
    Jm = 3.9E-7           # Rotor inertia [kg.m^2]
    r_mp = 6.35E-3        # Motor Pinion radius [m]
    Rm = 2.6              # Motor armature Resistance [Ohm]
    Kt = .00767           # Motor Torque Constant [N.zz/A]
    Km = .00767           # Motor Torque Constant [N.zz/A]
    mc = 0.37             # Mass of the cart [kg]

    if long:
        mp = 0.23         # Mass of the pole [kg]
        pl = 0.641 / 2.   # Half of the pole length [m]

        Beq = 5.4         # Equivalent Viscous damping Coefficient 5.4
        Bp = 0.0024       # Viscous coefficient at the pole 0.0024
        gain = 1.3
        scale = np.array([0.45, 1.])
    else:
        mp = 0.127        # Mass of the pole [kg]
        pl = 0.3365 / 2.  # Half of the pole length [m]
        Beq = 5.4         # Equivalent Viscous damping Coefficient 5.4
        Bp = 0.0024       # Viscous coefficient at the pole 0.0024
        gain = 1.5
        scale = np.array([1., 1.])

        # Compute Inertia:
        Jp = pl ** 2 * mp / 3.   # Pole inertia [kg.m^2]
        Jeq = mc + (eta_g * Kg ** 2 * Jm) / (r_mp ** 2)

    dt = 1e-3
    v_mx = 24.

    @classmethod
    def dynamics_fwd(cls, s, v_m):

        v_m = np.clip(v_m, -cls.v_mx, cls.v_mx)

        x, theta, x_dot, theta_dot = s

        # Compute force acting on the cart:
        F = ((cls.eta_g * cls.Kg * cls.eta_m * cls.Kt) / (cls.Rm * cls.r_mp) *
             (-cls.Kg * cls.Km * x_dot / cls.r_mp + cls.eta_m * v_m))

        # Compute acceleration:
        A11 = cls.mp + cls.Jeq
        A12 = cls.mp * cls.pl * np.cos(theta)
        A21 = cls.mp * cls.pl * np.cos(theta)
        A22 = cls.Jp + cls.mp * cls.pl ** 2

        b11 = F - cls.Beq * x_dot - cls.mp * cls.pl * np.sin(theta) * theta_dot ** 2
        b21 = 0. - cls.Bp * theta_dot - cls.mp * cls.pl * cls.g * np.sin(theta)

        A = np.vstack((np.hstack((A11, A12)),
                       np.hstack((A21, A22))))

        b = np.vstack((b11, b21))

        Ainv = np.linalg.inv(A)
        s_ddot = np.dot(Ainv, b).squeeze()
        s_vel = np.hstack((x_dot, theta_dot)) + s_ddot * cls.dt
        s_pos = np.hstack((x, theta)) + s_vel * cls.dt
        s_next = np.hstack((s_pos, s_vel))
        return s_next

    @classmethod
    def dynamics_dydx(cls):
        return jacobian(cls.dynamics_fwd, 0)

    @classmethod
    def dynamics_dydu(cls):
        return jacobian(cls.dynamics_fwd, 1)



def double_cartpole_dynamics(x, u):
    """
    http://www.lirmm.fr/~chemori/Temp/Wafa/double%20pendule%20inverse.pdf
    """
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
    fs_hz = 500.
    dt = 1 / fs_hz
    u_mx = 10.
    input_amp = 3. # simulate gear ratio etc

    q = x[0]
    th1 = x[1]
    th2 = x[2]
    q_dot = x[3]
    th_dot1 = x[4]
    th_dot2 = x[5]

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
    M11 = Mt
    M12 = l1_mp1_mp2_cth1
    M13 = Mp2_l2_cth2
    M21 = l1_mp1_mp2_cth1
    M22 = (l1 ** 2) * Mp1 + (L1 ** 2) * Mp2 + J1
    M23 = l1_l2_Mp2_cdth
    M31 = Mp2_l2_cth2
    M32 = l1_l2_Mp2_cdth
    M33 = (l2 ** 2) * Mp2 + J2

    # coreolis
    C11 = 0.
    C12 = -l1_mp1_mp2 * th_dot1 * sth1
    C13 = -Mp2_l2 * th_dot2 *sth2
    C21 = 0.
    C22 = 0.
    C23 = l1_l2_Mp2 * th_dot2 * sdth
    C31 = 0.
    C32 = -l1_l2_Mp2 * th_dot1 * sdth
    C33 = 0.

    # gravity
    G11 = 0.
    G21 = -(Mp1 * l1 + Mp2 * L1) *g * sth1
    G31 = -Mp2 * l2 * g * sth2

    # make matrices
    M = np.vstack((np.hstack((M11, M12, M13)),
                   np.hstack((M21, M22, M23)),
                   np.hstack((M31, M32, M33))))

    C = np.vstack((np.hstack((C11, C12, C13)),
                   np.hstack((C21, C22, C23)),
                   np.hstack((C31, C32, C33))))

    G = np.vstack((G11, G21, G31))

    u = input_amp * np.clip(u, -u_mx, u_mx)

    action = np.vstack((u, 0.0, 0.0))

    M_inv = np.linalg.inv(M)
    C_x_dot = np.dot(C, x[3:].reshape((-1, 1)))
    x_dot_dot = np.dot(M_inv, action - C_x_dot - G).squeeze()

    x_dot = x[3:] + x_dot_dot * dt
    x_pos = x[:3] + x_dot * dt

    x2 = np.hstack((x_pos, x_dot))

    return x2

double_cartpole_dydx = jacobian(double_cartpole_dynamics, 0)
double_cartpole_dydu = jacobian(double_cartpole_dynamics, 1)

def two_link_elastic_joint_robot_arm_dynamics(x, u):

    g = 9.81
    m1 = 0.5
    m2 = 0.5
    l1, l2 = 0.5, 0.5
    K = 5e3 * np.diag([1, 1])
    D = 0.5 * np.diag([1., 1.])
    B = 1. * np.eye(2)
    u_mx = 3.0
    dt = 1e-3

    # M q_dd + C q_d + g = tau_j
    # B th_dd + tau_j = u
    # tau_j = D (th_d - q_d) + K (th - q)

    qd1 = x[0]
    qd2 = x[1]
    thd1 = x[2]
    thd2 = x[3]
    q1 = x[4]
    q2 = x[5]
    th1 = x[6]
    th2 = x[7]

    qd = np.vstack((qd1, qd2))
    thd = np.vstack((thd1, thd2))
    q = np.vstack((q1, q2))
    th = np.vstack((th1, th2))

    sq1 = np.sin(q1)
    sq2 = np.sin(q2)
    cq2 = np.cos(q2)
    sq1q2 = np.sin(q1 + q2)

    # inertia
    M11 = (m1 + m2) * l1 ** 2 + m2 * l2 ** 2 + 2 * m2 * l1 * l2 * cq2
    M12 = m2 * l2 ** 2 + m2 * l1 * l2 * cq2
    M21 = m2 * l2 ** 2 + m2 * l1 * l2 * cq2
    M22 = m2 * l2 ** 2
    # coreolis
    C11 = -m2 * l1 * l2 * sq2 * (2 * qd1 * qd2 + q2 ** 2)
    C21 = -m2 * l1 * l2 * sq2 * qd1 * qd2
    # gravity
    G11 = -(m1 + m2) * g * l1 * sq1 - m2 * g * l2 * sq1q2
    G21 = -m2 * g * l2 * sq1q2

    M = np.vstack((np.hstack((M11, M12)), np.hstack((M21, M22))))
    C = np.vstack((C11, C21))
    G = np.vstack((G11, G21))

    # elastic joint dynamics
    tau_j = D.dot(thd - qd) + K.dot(th - q)

    # motor dynamics
    _u = np.clip(u, -u_mx, u_mx)

    B_inv = np.linalg.inv(B)
    thdd = np.dot(B_inv, _u - tau_j)

    # tau_j = u
    # arm dynamics
    # print(M.shape, tau_j.shape, G.shape, C.shape, qd.shape)
    M_inv = np.linalg.inv(M)
    qdd = np.dot(M_inv, tau_j - G - C)

    # assert not np.any(np.isnan(qdd))


    qd = qd + dt * qdd
    thd = th + dt * thdd
    q = q + dt * qd
    th = th + dt * thd

    x2 = np.vstack((qd, thd, q, th))

    return x2

two_link_dydx = jacobian(two_link_elastic_joint_robot_arm_dynamics, 0)
two_link_dydu = jacobian(two_link_elastic_joint_robot_arm_dynamics, 1)