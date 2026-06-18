import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

# Constants
X_C, X_C1, X_C2 = 2.0, 3.2, 4.0
A_PARAM, B_PARAM = 5.0, 1.0


# --- Optimal Velocity Models ---
def optimal_velocity(delta_x: float) -> float:
    if delta_x < X_C1:
        return np.tanh(delta_x - X_C) + np.tanh(X_C)
    elif delta_x < X_C2:
        return A_PARAM - delta_x
    else:
        return B_PARAM


def optimal_velocity_normal(delta_x: float) -> float:
    return np.tanh(delta_x - X_C) + np.tanh(X_C)


# --- Acceleration Models ---
def acceleration(v_i: float, v_im1: float, delta_x_i: float, 
                 kappa: float, lambda_: float) -> float:
    return kappa * (optimal_velocity(delta_x_i) - v_i) + lambda_ * (v_im1 - v_i)


def acceleration_normal(v_i: float, v_im1: float, delta_x_i: float, 
                         kappa: float, lambda_: float) -> float:
    return kappa * (optimal_velocity_normal(delta_x_i) - v_i) + lambda_ * (v_im1 - v_i)


# --- Initialization ---
def initialize(N: int, L: float) -> Tuple[np.ndarray, np.ndarray]:
    positions = np.linspace(0, L, N, endpoint=False)
    v0 = optimal_velocity(L / N)
    velocities = np.full(N, v0)
    return positions, velocities


# --- Euler Integration ---
def euler_step_template(accel_func, positions: np.ndarray, velocities: np.ndarray, 
                        kappa: float, lambda_: float, dt: float, L: float, 
                        noise_amplitude: float = 0.0, 
                        perturbation: Optional[dict] = None, t: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    N = velocities.shape[0]
    dx = (np.roll(positions, -1) - positions) % L
    v_lead = np.roll(velocities, -1)
    dv_dt = np.empty_like(velocities)

    for i in range(N):
        if perturbation and i in perturbation['idx_list'] and t < perturbation['duration']:
            dv_dt[i] = -1.0
        else:
            dv_dt[i] = accel_func(velocities[i], v_lead[i], dx[i], kappa, lambda_)

    v_next = velocities + dv_dt * dt

    if noise_amplitude > 0.0:
        v_next += (np.random.rand(N) - 0.5) * noise_amplitude

        v_max = optimal_velocity(X_C1)
        v_next = np.clip(v_next, 0.0, v_max)

        x_next = (positions + 0.5 * (velocities + v_next) * dt) % L

    else:
        v_next = np.maximum(v_next, 0.0)
        x_next = (positions + velocities * dt + 0.5 * dv_dt * dt**2) % L
        
    return x_next, v_next


def euler_step(*args, **kwargs):
    return euler_step_template(acceleration, *args, **kwargs)


def euler_step_normal(*args, **kwargs):
    return euler_step_template(acceleration_normal, *args, **kwargs)


# --- Raster Plot ---
def plot_raster(N: int, n_dec: int, kappa: float, lambda_: float, *, 
                dt=0.1, simulation_time=2000, noise_amplitude=0.0, 
                L=500, raster_range=300, figsize=(9, 3)) -> Tuple[plt.Figure, plt.Axes]:
    steps = int(simulation_time / dt)
    x, v = initialize(N, L)

    np.random.seed(42)
    perturbation = {'idx_list': np.random.choice(N, size=1, replace=False), 'duration': n_dec}
    trajectory = np.zeros((steps, N))

    for t in range(steps):
        trajectory[t] = x
        x, v = euler_step(x, v, kappa, lambda_, dt, L, noise_amplitude, perturbation=perturbation, t=t)

    traj = trajectory[-raster_range:]
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(N):
        ax.scatter(traj[:, i], np.arange(raster_range), s=1, color='gray', alpha=0.5, rasterized=True)
    ax.set(xlabel='Position', ylabel='Time step', xlim=(0, L), ylim=(0, raster_range), title='Car positions over time')
    fig.tight_layout()
    return fig, ax


# --- Fundamental Diagram Plot ---
def compute_fundamental_diagram(kappa: float, lambda_: float, n_dec: int, *, 
                             simulation_time=1000, dt=0.1, L=500, 
                             stop_warmup_steps=200, N_start=50, N_end=450, N_step=10, 
                             noise_amplitude=0.0, label="Night driving with perturbations"):
    steps = int(simulation_time / dt)
    warmup = steps - stop_warmup_steps
    N_values = np.arange(N_start, N_end + 1, N_step)
    densities = N_values / L
    flows = []

    for N in N_values:
        np.random.seed(42)
        perturbation = {'idx_list': np.random.choice(N, size=1, replace=False), 'duration': n_dec}
        x, v = initialize(N, L)
        v_record = []

        for t in range(steps):
            x, v = euler_step(x, v, kappa, lambda_, dt, L, noise_amplitude, perturbation, t=t)
            if t >= warmup:
                v_record.append(v.copy())

        flow = (N / L) * np.mean(v_record)
        flows.append(flow)

    return densities, flows

    plt.figure(figsize=(8, 5))
    plt.plot(densities, flows, 'o-', label=label)
    plt.xlabel("Density (ρ = N / L)")
    plt.ylabel("Flow (q = ρ · v̄)")
    plt.title("Fundamental Diagram: Flow vs Density")
    plt.grid(True)
    plt.legend()
    plt.show()