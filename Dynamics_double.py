import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
from numba import njit
def fx(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    M = 2
    m1, m2 = 1, 1
    l1, l2 = 2, 2
    g = 9.81
    
    xc, t1, t2, xcp, t1p, t2p = x
    F = u[0]
    
    s1, c1 = np.sin(t1), np.cos(t1)
    s2, c2 = np.sin(t2), np.cos(t2)
    s12, c12 = np.sin(t1 - t2), np.cos(t1 - t2)

    R1 = F + (m1 + m2) * l1 * (t1p**2) * s1 + m2 * l2 * (t2p**2) * s2
    R2 = -m2 * l1 * l2 * (t2p**2) * s12 - (m1 + m2) * g * l1 * s1
    R3 = m2 * l1 * l2 * (t1p**2) * s12 - m2 * g * l2 * s2
    
    M11 = M + m1 + m2
    M12 = (m1 + m2) * l1 * c1
    M13 = m2 * l2 * c2
    M22 = (m1 + m2) * (l1**2)
    M23 = m2 * l1 * l2 * c12
    M33 = m2 * (l2**2)

    T1 = M22 * M33 - M23**2
    T2 = M12 * M33 - M13 * M23
    T3 = M12 * M23 - M13 * M22
    T4 = M11 * M33 - M13**2
    T5 = M12 * M13 - M11 * M23
    T6 = M11 * M22 - M12**2
    
    det = M11 * T1 - M12 * T2 + M13 * T3
    
    xcpp = ( T1 * R1 - T2 * R2 + T3 * R3) / det
    t1pp = (-T2 * R1 + T4 * R2 + T5 * R3) / det
    t2pp = ( T3 * R1 + T5 * R2 + T6 * R3) / det
    
    return np.array([xcp, t1p, t2p, xcpp, t1pp, t2pp])

@njit(cache=True, fastmath=True)
def fx_numba(x, u, M, m1, m2, l1, l2, g):
    xc, t1, t2, xcp, t1p, t2p = x
    F = u[0]
    
    s1, c1 = np.sin(t1), np.cos(t1)
    s2, c2 = np.sin(t2), np.cos(t2)
    s12, c12 = np.sin(t1 - t2), np.cos(t1 - t2)
    
    R1 = F + (m1 + m2) * l1 * (t1p**2) * s1 + m2 * l2 * (t2p**2) * s2
    R2 = -m2 * l1 * l2 * (t2p**2) * s12 - (m1 + m2) * g * l1 * s1
    R3 = m2 * l1 * l2 * (t1p**2) * s12 - m2 * g * l2 * s2
    
    M11 = M + m1 + m2
    M12 = (m1 + m2) * l1 * c1
    M13 = m2 * l2 * c2
    M22 = (m1 + m2) * (l1**2)
    M23 = m2 * l1 * l2 * c12
    M33 = m2 * (l2**2)
    
    T1 = M22 * M33 - M23**2
    T2 = M12 * M33 - M13 * M23
    T3 = M12 * M23 - M13 * M22
    T4 = M11 * M33 - M13**2
    T5 = M12 * M13 - M11 * M23
    T6 = M11 * M22 - M12**2
    
    det = M11 * T1 - M12 * T2 + M13 * T3
    
    xcpp = ( T1 * R1 - T2 * R2 + T3 * R3) / det
    t1pp = (-T2 * R1 + T4 * R2 + T5 * R3) / det
    t2pp = ( T3 * R1 + T5 * R2 + T6 * R3) / det
    
    return np.array([xcp, t1p, t2p, xcpp, t1pp, t2pp])

def RK4(F, x, U, h):
    k0 = F(x, U)
    k1 = F(x + k0 * h / 2, U)
    k2 = F(x + k1 * h / 2, U)
    k3 = F(x + k2 * h, U)
    k_final = k0/6 + k1/3 + k2/3 + k3/6
    x_final = x + k_final * h
    return x_final

@njit(cache=True, fastmath=True)
def rk4_step_numba(x, u, dt, M, m1, m2, l1, l2, g):
    k1 = fx_numba(x, u, M, m1, m2, l1, l2, g)
    k2 = fx_numba(x + 0.5 * dt * k1, u, M, m1, m2, l1, l2, g)
    k3 = fx_numba(x + 0.5 * dt * k2, u, M, m1, m2, l1, l2, g)
    k4 = fx_numba(x + dt * k3, u, M, m1, m2, l1, l2, g)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

cartW = 2.0
cartH = 1.0
L1 = 2.0
L2 = 2.0
pendulumR = 0.3

def discrete_dynamics(x, u, h): # 망할 연속이면 안되지 이걸 까먹네
    return RK4(fx, x, u, h)

# 그리기

def draw_double_cartpole(ax, x, t1, t2):
    ax.clear()
    ax.set_xlim(-30, 30)
    ax.set_ylim(-8, 8) 
    ax.set_aspect('equal')
    ax.grid(True)
    ax.axhline(0, color='gray', lw=0.5)
    
    ax.add_patch(Rectangle((x - cartW/2, -cartH/2), cartW, cartH,
                           facecolor='steelblue', edgecolor='k'))
    
    x1 = x + L1 * np.sin(t1)
    y1 = -L1 * np.cos(t1)
    
    x2 = x1 + L2 * np.sin(t2)
    y2 = y1 - L2 * np.cos(t2)
    
    ax.plot([x, x1], [0, y1], 'k-', lw=3)
    ax.plot([x1, x2], [y1, y2], 'k-', lw=3)
    
    ax.add_patch(Circle((x1, y1), pendulumR * 0.8,
                        facecolor='orange', edgecolor='k', zorder=3))
    ax.add_patch(Circle((x2, y2), pendulumR,
                        facecolor='red', edgecolor='k', zorder=3))

if __name__ == "__main__": # 테스트
    step_length = 0.05
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.ion()

    X = np.array([0.0, np.pi + 0.1, np.pi - 0.1, 0.0, 0.0, 0.0])
    U = np.array([0.0])
    
    for i in range(1000):
        xc = X[0]
        t1 = X[1]
        t2 = X[2]
        draw_double_cartpole(ax, xc, t1, t2)
        plt.pause(step_length)
        X = discrete_dynamics(X, U, step_length)

    plt.ioff()
    plt.show()