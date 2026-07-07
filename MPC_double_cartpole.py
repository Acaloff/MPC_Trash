import numpy as np
from numba import njit
from Dynamics_double import rk4_step_numba, draw_double_cartpole
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize
import matplotlib.pyplot as plt

@njit(cache=True, fastmath=True)
def wrap_angle_numba(err_array):
    err = err_array.copy()
    err[1] = (err[1] + np.pi) % (2 * np.pi) - np.pi
    err[2] = (err[2] + np.pi) % (2 * np.pi) - np.pi
    return err

@njit(cache=True, fastmath=True)
def compute_cost_numba(u_seq_flat, x_curr, x_ref, Pf, Q, R, N, dt, M, m1, m2, l1, l2, g):
    x = x_curr.copy()
    cost = 0.0
    for i in range(N):
        u = np.array([u_seq_flat[i]], dtype=np.float64)
        x = rk4_step_numba(x, u, dt, M, m1, m2, l1, l2, g)
        
        err = x - x_ref
        err = wrap_angle_numba(err)
        cost += np.dot(err, np.dot(Q, err)) + np.dot(u, np.dot(R, u))

    # V_fin
    err_N = x - x_ref
    err_N = wrap_angle_numba(err_N)
    cost += np.dot(err_N, np.dot(Pf, err_N))
    
    return cost

def get_jacobian(f, x0, u0, eps=1e-6):
    x_want = np.asarray(x0, dtype=float)
    u_want = np.atleast_1d(np.asarray(u0, dtype=float))
    n = len(x_want)
    m = len(u_want)
    
    A = np.zeros((n, n))
    
    B = np.zeros((n, m))
    
    for i in range(n):
        x_u, x_d = x_want.copy(), x_want.copy()
        x_u[i] += eps
        x_d[i] -= eps
        A[:, i] = (f(x_u, u_want) - f(x_d, u_want)) / (2 * eps)
    
    for i in range(m):
        u_u, u_d = u_want.copy(), u_want.copy()
        u_u[i] += eps
        u_d[i] -= eps
        B[:, i] = (f(x_want, u_u) - f(x_want, u_d)) / (2 * eps)
        
    return A, B
        
class StepSolver:
    def __init__(self, Q_, R_, N_, step_length, x_refs: list[np.ndarray], u_stable: np.ndarray, phys_params: tuple):
        self.Q = Q_
        self.R = R_
        self.N = N_
        self.step_length = step_length
        self.phys_params = phys_params
        M, m1, m2, l1, l2, g = self.phys_params

        disc = lambda x, u: rk4_step_numba(x, u, step_length, M, m1, m2, l1, l2, g)
        
        self.x_refs = x_refs
        self.P_fs = []
        for x_ref in self.x_refs:
            Af, Bf = get_jacobian(disc, x_ref, u_stable)
            Pf = solve_discrete_are(Af, Bf, self.Q, self.R)
            self.P_fs.append(Pf)
            
        self.n = np.atleast_1d(np.asarray(self.x_refs[0])).size
        self.m = np.atleast_1d(np.asarray(u_stable)).size
        self.u_prev_guess = np.zeros(self.N * self.m)
        
    def solve_step(self, x, target_mode: int = 0):
        N = self.N
        u_init = self.u_prev_guess 
        
        x_ref = self.x_refs[target_mode]
        Pf = self.P_fs[target_mode]
        M, m1, m2, l1, l2, g = self.phys_params
        args = (x, x_ref, Pf, self.Q, self.R, N, self.step_length, 
                M, m1, m2, l1, l2, g)

        res = minimize(compute_cost_numba, u_init, args=args, 
                       method='SLSQP', bounds=[(-40.0, 40.0)] * N, 
                       options={'maxiter': 50, 'ftol': 1e-3})
        
        u_opt = res.x
        
        self.u_prev_guess[:-1] = u_opt[1:]
        self.u_prev_guess[-1] = u_opt[-1]
        
        return np.atleast_1d(np.asarray(u_opt[0]))
    
if __name__ == "__main__":
    N = 30
    step_length = 0.05
    
    phys_params = (2.0, 1.0, 1.0, 2.0, 2.0, 9.81) 
    
    Q = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    R = np.array([[1.0]])
    
    x_refs = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),     # 1: Down-Down
        np.array([0.0, 0.0, np.pi, 0.0, 0.0, 0.0]),
        np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0]),    # 2 Up-Down
        np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0]), # 0: Up-Up
    ]
    us = np.array([0.0])
    
    solver = StepSolver(Q, R, N, step_length, x_refs, us, phys_params)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.ion()
    
    app_state = {'target_mode': 0}
    
    def on_key_press(event):
        if event.key in ['1']:
            app_state['target_mode'] = 0
        elif event.key in ['2']:
            app_state['target_mode'] = 1
        elif event.key in ['3']:
            app_state['target_mode'] = 2
        elif event.key in ['4']:
            app_state['target_mode'] = 3

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    X = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    M_p, m1_p, m2_p, l1_p, l2_p, g_p = phys_params

    while plt.fignum_exists(fig.number):
        draw_double_cartpole(ax, X[0], X[1], X[2])
        plt.pause(step_length)
        
        current_mode = app_state['target_mode']
        u = solver.solve_step(X, target_mode=current_mode)
        X = rk4_step_numba(X, u, step_length, M_p, m1_p, m2_p, l1_p, l2_p, g_p)
        
    plt.ioff()