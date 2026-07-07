"""
Headless demo recorder: runs the MPC and saves a GIF (no interactive window).
Usage:  python record_demo.py
Output: assets/demo.gif
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from Dynamics_double import rk4_step_numba, draw_double_cartpole
from MPC_double_cartpole import StepSolver


def main():
    N = 30
    dt = 0.05
    phys = (2.0, 1.0, 1.0, 2.0, 2.0, 9.81)
    Q = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    R = np.array([[1.0]])
    x_refs = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),      # 0: down-down
        np.array([0.0, 0.0, np.pi, 0.0, 0.0, 0.0]),    # 1: down-up
        np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0]),    # 2: up-down
        np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0]),  # 3: up-up
    ]
    solver = StepSolver(Q, R, N, dt, x_refs, np.array([0.0]), phys)
    M, m1, m2, l1, l2, g = phys

    # target schedule: (mode, number_of_steps). Shows stabilize -> switch targets.
    schedule = [(3, 120), (2, 90), (1, 90), (3, 100)]

    # start slightly perturbed from up-up so the balancing is visible immediately
    X = np.array([0.0, np.pi + 0.15, np.pi - 0.15, 0.0, 0.0, 0.0])
    traj = []
    for mode, nsteps in schedule:
        for _ in range(nsteps):
            traj.append(X.copy())
            u = solver.solve_step(X, target_mode=mode)
            X = rk4_step_numba(X, u, dt, M, m1, m2, l1, l2, g)
    print(f"simulated {len(traj)} steps")

    # subsample for a lighter GIF
    frames = traj[::2]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    def update(i):
        draw_double_cartpole(ax, frames[i][0], frames[i][1], frames[i][2])
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=50)
    anim.save("assets/demo.gif", writer=PillowWriter(fps=20))
    plt.close(fig)
    print("saved assets/demo.gif")


if __name__ == "__main__":
    main()
