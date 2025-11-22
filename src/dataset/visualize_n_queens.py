import argparse
import os
import numpy as np
import json


def print_board(flat_board, size, title="Board"):
    print(f"--- {title} ---")
    # Remap back: 0=PAD, 1=Empty, 2=Queen
    # Display: . = Empty, Q = Queen

    board = flat_board.reshape(size, size)

    print("  " + " ".join([str(i % 10) for i in range(size)]))
    for r in range(size):
        row_str = f"{r % 10} "
        for c in range(size):
            val = board[r, c]
            if val == 2:  # Queen
                row_str += "Q "
            elif val == 1:  # Empty
                row_str += ". "
            else:  # Pad?
                row_str += "? "
        print(row_str)
    print("")


def visualize(data_dir, num_samples=5):
    # Load dataset metadata to get size
    with open(os.path.join(data_dir, "dataset.json"), "r") as f:
        meta = json.load(f)

    seq_len = meta["seq_len"]
    n_size = int(seq_len**0.5)
    print(f"Detected Board Size: {n_size}x{n_size}")

    # Load data
    inputs = np.load(os.path.join(data_dir, "all__inputs.npy"))
    labels = np.load(os.path.join(data_dir, "all__labels.npy"))

    indices = np.random.choice(
        len(inputs), min(len(inputs), num_samples), replace=False
    )

    for i in indices:
        print(f"\nExample #{i}")
        print_board(inputs[i], n_size, title="Input (Problem)")
        print_board(labels[i], n_size, title="Label (Solution)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to dataset subset (e.g. data/n_queens_12/train)",
    )
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    visualize(args.dir, args.n)
