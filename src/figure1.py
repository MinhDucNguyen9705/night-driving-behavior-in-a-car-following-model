import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from helper_functions import optimal_velocity, optimal_velocity_normal

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../figures/figure_1_reproduction.png")
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=8.0)
    parser.add_argument("--num-points", type=int, default=400)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    x_vals = np.linspace(args.x_min, args.x_max, args.num_points)
    y1 = optimal_velocity(x_vals)
    y2 = optimal_velocity_normal(x_vals)

    plt.figure(figsize=(12, 6))
    plt.plot(x_vals, y1, 'r--', label='Night traffic optimal velocity function', zorder=2)
    plt.plot(x_vals, y2, 'b-', label='Normal optimal velocity function', zorder=1)
    plt.xlabel("delta_x")
    plt.ylabel("Velocity")
    plt.title("Comparison of Velocity Functions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
