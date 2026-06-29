import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from helper_functions import plot_raster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-a", default="../figures/figure_8a_reproduction.png")
    parser.add_argument("--output-b", default="../figures/figure_8b_reproduction.png")
    parser.add_argument("--simulation-time", type=float, default=2000.0)
    parser.add_argument("--raster-range", type=int, default=300)
    args = parser.parse_args()

    output_a = Path(args.output_a)
    output_b = Path(args.output_b)
    output_a.parent.mkdir(parents=True, exist_ok=True)
    output_b.parent.mkdir(parents=True, exist_ok=True)

    fig_a, ax_a = plot_raster(
        N=300,
        n_dec=1,
        kappa=1.0,
        lambda_=0.1,
        simulation_time=args.simulation_time,
        raster_range=args.raster_range,
        figsize=(6, 6),
    )
    ax_a.set_title("Figure 8(a): FVD, kappa=1.0, lambda=0.1")
    fig_a.savefig(output_a, dpi=300)

    fig_b, ax_b = plot_raster(
        N=300,
        n_dec=1,
        kappa=1.2,
        lambda_=0.0,
        simulation_time=args.simulation_time,
        raster_range=args.raster_range,
        figsize=(6, 6),
    )
    ax_b.set_title("Figure 8(b): OV case, kappa=1.2, lambda=0.0")
    fig_b.savefig(output_b, dpi=300)

    plt.close(fig_a)
    plt.close(fig_b)


if __name__ == "__main__":
    main()
