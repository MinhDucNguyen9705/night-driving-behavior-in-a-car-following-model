import argparse
import numpy as np
import matplotlib.pyplot as plt

from helper_functions import compute_fundamental_diagram, theoretical_curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Use fewer densities and fewer steps for quick testing.")
    parser.add_argument("--output", default="figure_6_reproduction.png")
    args = parser.parse_args()

    if args.fast:
        N_values = np.arange(20, 551, 20)
        total_steps = 3000
        warmup_steps = 1500
    else:
        N_values = np.arange(5, 551, 5)
        total_steps = 10000
        warmup_steps = 9800

    rho_theory, q_normal, q_night = theoretical_curves()

    densities, flows = compute_fundamental_diagram(
        lam=0.1,
        kappa=1.0,
        ndec=1,
        N_values=N_values,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    plt.figure(figsize=(7, 5))
    plt.plot(rho_theory, q_normal, linestyle="-", label="Normal driving, theoretical")
    plt.plot(rho_theory, q_night, linestyle="--", label="Night driving, theoretical")
    plt.plot(densities, flows, linestyle=":", linewidth=2.0, label="Night driving, simulation")

    plt.xlabel("Density")
    plt.ylabel("Flow")
    plt.title("Figure 6 reproduction: kappa=1.0, lambda=0.1, ndec=1")
    plt.xlim(0, 1.1)
    plt.ylim(0, 0.8)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
