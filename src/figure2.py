import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from helper_functions import numerical_derivative, optimal_velocity, optimal_velocity_normal

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../figures/figure_2_reproduction.png")
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=6.0)
    parser.add_argument("--num-points", type=int, default=1000)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    x_vals = np.linspace(args.x_min, args.x_max, args.num_points)
    y_piecewise = optimal_velocity(x_vals)
    y_tanh = optimal_velocity_normal(x_vals)

    dy_piecewise = numerical_derivative(y_piecewise, x_vals)
    dy_tanh = numerical_derivative(y_tanh, x_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, dy_piecewise, 'r--', label="d/dx optimal_velocity (piecewise)", zorder=2)
    plt.plot(x_vals, dy_tanh, 'b-', label="d/dx tanh(delta_x - X_C) + tanh(X_C)", zorder=1)
    plt.xlabel("delta_x")
    plt.ylabel("First Derivative")
    plt.title("First Derivative Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
