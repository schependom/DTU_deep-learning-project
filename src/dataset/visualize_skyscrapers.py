import argparse
import numpy as np
import os

def visualize(data_dir, n_display=3):
    try:
        inputs = np.load(os.path.join(data_dir, "all__inputs.npy"))
        labels = np.load(os.path.join(data_dir, "all__labels.npy"))
    except:
        print(f"Could not load data from {data_dir}")
        return

    count = len(inputs)
    indices = np.random.choice(count, min(count, n_display), replace=False)
    
    # Infer N from seq len
    seq_len = inputs.shape[1]
    grid_dim = int(np.sqrt(seq_len))
    
    print(f"Visualizing Skyscrapers (Grid size including clues: {grid_dim}x{grid_dim})")
    print("-" * 30)

    for idx in indices:
        inp = inputs[idx].reshape(grid_dim, grid_dim)
        lab = labels[idx].reshape(grid_dim, grid_dim)
        
        # Combine side by side
        print(f"Example {idx}: Input vs Label")
        
        for r in range(grid_dim):
            # Input Row
            row_str_i = ""
            for c in range(grid_dim):
                val = inp[r, c]
                char = str(val) if val > 0 else "."
                # Highlight clues (edges)
                if r == 0 or r == grid_dim-1 or c == 0 or c == grid_dim-1:
                    row_str_i += f"[{char}]"
                else:
                    row_str_i += f" {char} "
            
            # Separator
            sep = "   |   "
            
            # Label Row
            row_str_l = ""
            for c in range(grid_dim):
                val = lab[r, c]
                char = str(val)
                if r == 0 or r == grid_dim-1 or c == 0 or c == grid_dim-1:
                    row_str_l += f"[{char}]"
                else:
                    row_str_l += f" {char} "
            
            print(row_str_i + sep + row_str_l)
        print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()
    visualize(args.dir, args.n)