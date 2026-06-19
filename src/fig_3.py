import argparse
import matplotlib.pyplot as plt

from helper_functions import plot_raster

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Use fewer steps for quick testing.")
    parser.add_argument("--output", default="figure_3_reproduction.png")
    args = parser.parse_args()

    if args.fast:
        simulation_time = 200
        raster_range = 50
    else:
        simulation_time = 2000
        raster_range = 300

    fig, ax = plot_raster(
        N=150, 
        n_dec=1, 
        kappa=1, 
        lambda_=0.5,
        simulation_time=simulation_time,
        raster_range=raster_range
    )
    
    plt.savefig(args.output, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
