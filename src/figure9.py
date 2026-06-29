import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from helper_functions import plot_raster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="../figures")
    parser.add_argument("--simulation-time", type=float, default=3000.0)
    parser.add_argument("--raster-range", type=int, default=300)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for A in [0.01, 0.05, 0.1]:
        fig, ax = plot_raster(
            N=300,
            n_dec=1,
            kappa=1.0,
            lambda_=0.1,
            simulation_time=args.simulation_time,
            noise_amplitude=A,
            raster_range=args.raster_range,
            figsize=(9, 3),
        )
        ax.set_title(f"Figure 9: randomness A={A:g}")
        fig.savefig(output_dir / f"figure_9_A_{A:g}.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
