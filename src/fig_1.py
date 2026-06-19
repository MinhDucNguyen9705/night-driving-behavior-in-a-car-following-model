import argparse
import numpy as np
import matplotlib.pyplot as plt

from helper_functions import optimal_velocity

X_C, X_C1, X_C2 = 2.0, 3.2, 4.0
A_PARAM, B_PARAM = 5.0, 1.0

def tanh_formula(delta_x: np.ndarray) -> np.ndarray:
    return np.tanh(delta_x - X_C) + np.tanh(X_C)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Use fewer points for quick testing.")
    parser.add_argument("--output", default="figure_1_reproduction.png")
    args = parser.parse_args()

    vectorized_optimal_velocity = np.vectorize(optimal_velocity)

    if args.fast:
        num_points = 50
    else:
        num_points = 400

    x_vals = np.linspace(0, 8, num_points)
    y1 = vectorized_optimal_velocity(x_vals)
    y2 = tanh_formula(x_vals)

    plt.figure(figsize=(12, 6))
    plt.plot(x_vals, y1, 'r--', label='Night traffic optimal velocity function', zorder=2)
    plt.plot(x_vals, y2, 'b-', label='Normal optimal velocity function', zorder=1)
    plt.xlabel("delta_x")
    plt.ylabel("Velocity")
    plt.title("Comparison of Velocity Functions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
