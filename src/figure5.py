import argparse
import numpy as np
import matplotlib.pyplot as plt

from helper_functions import L, simulate_fvd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figure_5_reproduction.png")
    parser.add_argument("--total_steps", type=int, default=8600)
    parser.add_argument("--record_start", type=int, default=7000)
    parser.add_argument("--record_interval", type=int, default=2)
    args = parser.parse_args()

    _, records = simulate_fvd(
        N=250,
        kappa=1.0,
        lam=0.2,
        ndec=1,
        total_steps=args.total_steps,
        warmup_steps=4000,
        record_space_time=True,
        record_start=args.record_start,
        record_interval=args.record_interval,
    )

    plt.figure(figsize=(6, 6))

    for t_idx, positions in enumerate(records):
        plt.scatter(
            positions,
            np.full_like(positions, t_idx),
            s=1,
            marker=".",
            # linewidths=0,
            rasterized=True,
            alpha=0.5,
            color='gray'
        )

    plt.xlabel("Position")
    plt.ylabel("Time")
    plt.title("Figure 5 reproduction: N=250, kappa=1.0, lambda=0.2")
    plt.xlim(0, L)
    plt.ylim(0, len(records))
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
