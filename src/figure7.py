import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from helper_functions import plot_raster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../figures/figure_7_reproduction.png")
    parser.add_argument("--simulation-time", type=float, default=1200.0)
    parser.add_argument("--raster-range", type=int, default=700)
    args = parser.parse_args()

    fig, ax = plot_raster(
        N=230,
        n_dec=1,
        kappa=1.0,
        lambda_=0.1,
        simulation_time=args.simulation_time,
        raster_range=args.raster_range,
        figsize=(6, 6),
    )
    ax.set_title("Figure 7 reproduction: stable clusters")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
