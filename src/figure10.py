import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from helper_functions import compute_fundamental_diagram, theoretical_curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../figures/figure_10_reproduction.png")
    parser.add_argument("--n-min", type=int, default=5)
    parser.add_argument("--n-max", type=int, default=550)
    parser.add_argument("--n-step", type=int, default=10)
    parser.add_argument("--total-steps", type=int, default=12000)
    parser.add_argument("--warmup-steps", type=int, default=11000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    N_values = np.arange(args.n_min, args.n_max + 1, args.n_step)
    rho_theory, q_normal, q_night = theoretical_curves()

    densities, flows = compute_fundamental_diagram(
        kappa=1.0,
        lambda_=0.5,
        n_dec=80,
        N_values=N_values,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        verbose=args.verbose,
    )

    plt.figure(figsize=(7, 5))
    plt.plot(rho_theory, q_normal, linestyle="-", label="Normal driving")
    plt.plot(rho_theory, q_night, linestyle="--", label="Night driving")
    plt.plot(
        densities,
        flows,
        linestyle=":",
        linewidth=2.0,
        label=r"Night driving with large perturbation, $n_{dec}=80$",
    )

    plt.xlabel("Density")
    plt.ylabel("Flow")
    plt.title(r"Figure 10 reproduction: kappa=1.0, lambda=0.5, $n_{dec}=80$")
    plt.xlim(0, 1.1)
    plt.ylim(0, 0.8)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
