import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from helper_functions import plot_raster

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../figures/figure_3_reproduction.png")
    parser.add_argument("--simulation-time", type=float, default=2000.0)
    parser.add_argument("--raster-range", type=int, default=300)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = plot_raster(
        N=150,
        n_dec=1,
        kappa=1.0,
        lambda_=0.5,
        simulation_time=args.simulation_time,
        raster_range=args.raster_range,
    )
    
    fig.savefig(output, dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()
