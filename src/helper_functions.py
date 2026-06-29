from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

L = 500.0
DT = 0.1

XC = 2.0
XC1 = 3.2
XC2 = 4.0
A_PARAM = 5.0
B_PARAM = 1.0

def _return_scalar_if_scalar(input_value, output):
    return float(output) if np.isscalar(input_value) else output

def V_normal(dx):
    """
    Normal optimal velocity function:
        V(dx) = tanh(dx - xc) + tanh(xc)
    """
    values = np.asarray(dx, dtype=float)
    result = np.tanh(values - XC) + np.tanh(XC)
    return _return_scalar_if_scalar(dx, result)


def V_night(dx):
    """
    Night-driving optimal velocity function:
        V(dx) = tanh(dx - xc) + tanh(xc),  dx < xc1
              = a - dx,                    xc1 <= dx < xc2
              = b,                         dx >= xc2
    """
    values = np.asarray(dx, dtype=float)
    result = np.where(
        values < XC1,
        np.tanh(values - XC) + np.tanh(XC),
        np.where(values < XC2, A_PARAM - values, B_PARAM),
    )
    return _return_scalar_if_scalar(dx, result)


def optimal_velocity(delta_x):
    return V_night(delta_x)


def optimal_velocity_normal(delta_x):
    return V_normal(delta_x)


def numerical_derivative(f_vals, x_vals):
    dx = x_vals[1] - x_vals[0]
    deriv = np.zeros_like(f_vals)
    deriv[1:-1] = (f_vals[2:] - f_vals[:-2]) / (2 * dx)
    deriv[0] = (f_vals[1] - f_vals[0]) / dx
    deriv[-1] = (f_vals[-1] - f_vals[-2]) / dx
    return deriv


def theoretical_curves(rho_min=0.01, rho_max=1.1, points=500):
    """
    Equilibrium fundamental diagrams:
        q = rho * V(1 / rho)
    """
    rho = np.linspace(rho_min, rho_max, points)
    headway = 1.0 / rho

    q_normal = rho * V_normal(headway)
    q_night = rho * V_night(headway)

    return rho, q_normal, q_night


def acceleration(
    v_i: float,
    v_im1: float,
    delta_x_i: float,
    kappa: float,
    lambda_: float,
) -> float:
    return kappa * (optimal_velocity(delta_x_i) - v_i) + lambda_ * (v_im1 - v_i)


def acceleration_normal(
    v_i: float,
    v_im1: float,
    delta_x_i: float,
    kappa: float,
    lambda_: float,
) -> float:
    return kappa * (optimal_velocity_normal(delta_x_i) - v_i) + lambda_ * (v_im1 - v_i)


def initialize(N: int, road_length: float = L) -> Tuple[np.ndarray, np.ndarray]:
    positions = np.linspace(0, road_length, N, endpoint=False)
    v0 = optimal_velocity(road_length / N)
    velocities = np.full(N, v0)
    return positions, velocities


def euler_step_template(
    accel_func: Callable,
    positions: np.ndarray,
    velocities: np.ndarray,
    kappa: float,
    lambda_: float,
    dt: float,
    road_length: float,
    noise_amplitude: float = 0.0,
    perturbation: Optional[dict] = None,
    t: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    n_cars = velocities.shape[0]
    dx = (np.roll(positions, -1) - positions) % road_length
    v_lead = np.roll(velocities, -1)
    dv_dt = np.empty_like(velocities)

    for i in range(n_cars):
        if perturbation and i in perturbation["idx_list"] and t < perturbation["duration"]:
            dv_dt[i] = -1.0
        else:
            dv_dt[i] = accel_func(velocities[i], v_lead[i], dx[i], kappa, lambda_)

    v_next = np.maximum(velocities + dv_dt * dt, 0.0)
    if noise_amplitude > 0.0:
        v_next += (np.random.rand(n_cars) - 0.5) * noise_amplitude

    x_next = (positions + velocities * dt + 0.5 * dv_dt * dt**2) % road_length

    return x_next, v_next


def euler_step(*args, **kwargs):
    return euler_step_template(acceleration, *args, **kwargs)


def euler_step_normal(*args, **kwargs):
    return euler_step_template(acceleration_normal, *args, **kwargs)


def plot_raster(
    N: int,
    n_dec: int,
    kappa: float,
    lambda_: float,
    *,
    dt=DT,
    simulation_time=2000,
    noise_amplitude=0.0,
    L=L,
    raster_range=300,
    figsize=(9, 3),
) -> Tuple[plt.Figure, plt.Axes]:
    steps = int(simulation_time / dt)
    x, v = initialize(N, L)

    np.random.seed(42)
    perturbation = {
        "idx_list": np.random.choice(N, size=1, replace=False),
        "duration": n_dec,
    }
    trajectory = np.zeros((steps, N))

    for t in range(steps):
        trajectory[t] = x
        x, v = euler_step(
            x,
            v,
            kappa,
            lambda_,
            dt,
            L,
            noise_amplitude,
            perturbation=perturbation,
            t=t,
        )

    traj = trajectory[-raster_range:]
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(N):
        ax.scatter(
            traj[:, i],
            np.arange(raster_range),
            s=1,
            color="gray",
            alpha=0.5,
            rasterized=True,
        )
    ax.set(
        xlabel="Position",
        ylabel="Time step",
        xlim=(0, L),
        ylim=(0, raster_range),
        title="Car positions over time",
    )
    fig.tight_layout()
    return fig, ax


def compute_fundamental_diagram(
    kappa: float = 1.0,
    lambda_: float = 0.2,
    n_dec: int = 1,
    N_values=None,
    total_steps: int = 8000,
    warmup_steps: int = 4000,
    verbose: bool = True,
    *,
    dt=DT,
    L=L,
    noise_amplitude=0.0,
    perturbation_index: Optional[int] = 0,
    perturbation_seed: Optional[int] = None,
):
    """
    Compute density-flow points.
    """
    if N_values is None:
        N_values = np.arange(5, 551, 5)

    densities = []
    flows = []

    for N in N_values:
        N = int(N)
        rng = np.random.RandomState(perturbation_seed) if perturbation_seed is not None else np.random
        if perturbation_seed is None:
            idx_list = [] if perturbation_index is None else [perturbation_index]
        else:
            idx_list = rng.choice(N, size=1, replace=False)

        x, v = initialize(N, L)
        velocity_sum = 0.0
        velocity_count = 0
        vehicle_indices = np.arange(N)
        leader_indices = np.roll(vehicle_indices, -1)

        for t in range(total_steps):
            dx = x[leader_indices] - x
            dx = np.where(dx <= 0, dx + L, dx)
            dv = v[leader_indices] - v
            acc = kappa * (V_night(dx) - v) + lambda_ * dv

            if t < n_dec:
                acc[idx_list] = -1.0

            v_next = np.maximum(v + acc * dt, 0.0)
            if noise_amplitude > 0.0:
                v_next += (rng.rand(N) - 0.5) * noise_amplitude

            x_next = (x + v * dt + 0.5 * acc * dt**2) % L

            x = x_next
            v = v_next

            if t >= warmup_steps:
                velocity_sum += np.mean(v)
                velocity_count += 1

        avg_v = velocity_sum / velocity_count if velocity_count > 0 else float(np.mean(v))
        rho = N / L
        flow = rho * avg_v

        densities.append(rho)
        flows.append(flow)

        if verbose:
            print(f"N={N:4d}, rho={rho:.3f}, avg_v={avg_v:.4f}, flow={flow:.4f}")

    return np.asarray(densities), np.asarray(flows)
