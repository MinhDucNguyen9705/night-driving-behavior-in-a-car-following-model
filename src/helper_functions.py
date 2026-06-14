import numpy as np

L = 500.0
DT = 0.1

XC = 2.0
XC1 = 3.2
XC2 = 4.0
A_PARAM = 5.0
B_PARAM = 1.0


def V_normal(dx):
    """
    Normal optimal velocity function:
        V(dx) = tanh(dx - xc) + tanh(xc)
    """
    dx = np.asarray(dx, dtype=float)
    return np.tanh(dx - XC) + np.tanh(XC)


def V_night(dx):
    """
    Night-driving optimal velocity function:
        V(dx) = tanh(dx - xc) + tanh(xc),  dx < xc1
              = a - dx,                    xc1 <= dx < xc2
              = b,                         dx >= xc2
    """
    dx = np.asarray(dx, dtype=float)
    v = np.zeros_like(dx, dtype=float)

    mask1 = dx < XC1
    mask2 = (dx >= XC1) & (dx < XC2)
    mask3 = dx >= XC2

    v[mask1] = np.tanh(dx[mask1] - XC) + np.tanh(XC)
    v[mask2] = A_PARAM - dx[mask2]
    v[mask3] = B_PARAM

    return v


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

def simulate_fvd(
    N: int,
    kappa: float = 1.0,
    lam: float = 0.2,
    ndec: int = 1,
    total_steps: int = 8000,
    warmup_steps: int = 4000,
    record_space_time: bool = False,
    record_start: int = 3000,
    record_interval: int = 2,
):
    """
    Simulate the FVD model under periodic boundary conditions.

    Parameters
    ----------
    N:
        Number of vehicles.
    kappa:
        Sensitivity to optimal velocity.
    lam:
        Sensitivity to velocity difference.
    ndec:
        Perturbation duration. Small perturbation in Figures 4-6 uses ndec=1.
    total_steps:
        Total simulation steps.
    warmup_steps:
        Steps ignored before calculating average velocity.
    record_space_time:
        Whether to record positions for a space-time plot.
    record_start:
        Simulation step at which recording starts.
    record_interval:
        Record every this many simulation steps.

    Returns
    -------
    avg_v:
        Average velocity after warmup.
    records:
        List of sorted vehicle positions for space-time plotting.
    """
    # Homogeneous initial spacing.
    h = L / N
    x = np.arange(N, dtype=float) * h
    v = V_night(np.full(N, h))

    velocity_sum = 0.0
    velocity_count = 0
    records = []

    vehicle_indices = np.arange(N)
    leader_indices = np.roll(vehicle_indices, -1)

    for step in range(total_steps):
        # Periodic headway from each vehicle to its leader.
        dx = x[leader_indices] - x
        dx = np.where(dx <= 0, dx + L, dx)

        # Velocity difference between leader and follower.
        dv = v[leader_indices] - v

        # FVD acceleration.
        acc = kappa * (V_night(dx) - v) + lam * dv

        # Small perturbation: vehicle 0 decelerates with constant deceleration 1.
        if step < ndec:
            acc[0] = -1.0

        # Euler-like update consistent with the equations used in the paper.
        v_new = v + acc * DT
        v_new = np.maximum(v_new, 0.0)

        x_new = x + v * DT + 0.5 * acc * DT**2
        x_new = np.mod(x_new, L)

        x = x_new
        v = v_new

        if step >= warmup_steps:
            velocity_sum += np.mean(v)
            velocity_count += 1

        if record_space_time and step >= record_start and step % record_interval == 0:
            records.append(np.sort(x.copy()))

    avg_v = velocity_sum / velocity_count if velocity_count > 0 else float(np.mean(v))
    return avg_v, records

def compute_fundamental_diagram(
    lam: float,
    kappa: float = 1.0,
    ndec: int = 1,
    N_values=None,
    total_steps: int = 8000,
    warmup_steps: int = 4000,
    verbose: bool = True,
):
    """
    Compute density-flow points by running one simulation for each N.
    """

    if N_values is None:
        N_values = np.arange(5, 551, 5)

    densities = []
    flows = []

    for N in N_values:
        avg_v, _ = simulate_fvd(
            N=int(N),
            kappa=kappa,
            lam=lam,
            ndec=ndec,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            record_space_time=False,
        )

        rho = N / L
        flow = rho * avg_v

        densities.append(rho)
        flows.append(flow)

        if verbose:
            print(f"N={int(N):4d}, rho={rho:.3f}, avg_v={avg_v:.4f}, flow={flow:.4f}")

    return np.asarray(densities), np.asarray(flows)