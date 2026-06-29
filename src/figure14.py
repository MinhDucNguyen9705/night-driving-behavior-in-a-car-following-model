import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from helper_functions import L, compute_fundamental_diagram, optimal_velocity, optimal_velocity_normal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../figures/figure_14_reproduction.png")
    parser.add_argument("--n-min", type=int, default=50)
    parser.add_argument("--n-max", type=int, default=450)
    parser.add_argument("--n-step", type=int, default=10)
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=9800)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    n_values = np.arange(args.n_min, args.n_max + 1, args.n_step)
    densities = n_values / L
    normal_flows = densities * np.array([optimal_velocity_normal(L / n) for n in n_values])
    night_flows = densities * np.array([optimal_velocity(L / n) for n in n_values])

    densities, flows = compute_fundamental_diagram(
        kappa=1.0,
        lambda_=0.1,
        n_dec=80,
        N_values=n_values,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        L=L,
        dt=0.1,
        perturbation_seed=42,
        verbose=args.verbose,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(densities, normal_flows, "-", label="Normal driving")
    plt.plot(densities, night_flows, "--", label="Night driving without perturbations")
    plt.plot(densities, flows, ":", linewidth=2.0, label="Night driving with perturbations")

    plt.xlabel("Density (rho = N / L)")
    plt.ylabel("Flow (q = rho * v_bar)")
    plt.title("Fig 14")
    plt.grid(True)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
