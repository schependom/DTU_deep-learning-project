import argparse
import os
import numpy as np
import json

def print_board(flat_board, size, title="Board"):
    print(f"--- {title} ---")
    # Data Values: 0=PAD, 1=Empty, 2=Queen
    # Visualization Codes: 3=Hidden Queen (Masked)
    
    board = flat_board.reshape(size, size)
    
    # Column headers
    print("  " + " ".join([str(i%10) for i in range(size)]))
    
    for r in range(size):
        row_str = f"{r%10} "
        for c in range(size):
            val = board[r, c]
            if val == 2:   # Given Queen
                row_str += "Q "
            elif val == 3: # Hidden Queen (Masked)
                row_str += "X "
            elif val == 1: # Empty
                row_str += ". "
            else:          # Padding or unknown
                row_str += "? "
        print(row_str)
    print("")

def visualize(data_dir, num_samples=5):
    # Load dataset metadata to get size
    meta_path = os.path.join(data_dir, "dataset.json")
    if not os.path.exists(meta_path):
        print(f"Error: Metadata not found at {meta_path}")
        return

    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    seq_len = meta["seq_len"]
    n_size = int(seq_len ** 0.5)
    print(f"Detected Board Size: {n_size}x{n_size}")
    
    # Load data
    try:
        inputs = np.load(os.path.join(data_dir, "all__inputs.npy"))
        labels = np.load(os.path.join(data_dir, "all__labels.npy"))
    except FileNotFoundError:
        print("Error: Data .npy files not found in directory.")
        return
    
    indices = np.random.choice(len(inputs), min(len(inputs), num_samples), replace=False)
    
    for i in indices:
        print(f"\nExample #{i}")
        inp = inputs[i]
        lbl = labels[i]
        
        # Create a visualization board combining hints and targets
        # Start with the Solution (Label)
        viz_board = lbl.copy()
        
        # Logic:
        # If Label has Queen (2) BUT Input had Empty (1), then it was a Hidden Queen.
        # We mark this as 3 for visualization.
        mask_indices = (lbl == 2) & (inp == 1)
        viz_board[mask_indices] = 3
        
        print_board(inp, n_size, title="Input (What Model Sees)")
        print_board(viz_board, n_size, title="Target (Q=Given, X=To Predict)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Path to dataset subset (e.g. data/n_queens_12/train)")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()
    
    visualize(args.dir, args.n)