import argparse
import os
import numpy as np
import json

def generate_latex(data_dir, num_samples=3, output_file="n_queens_vis.tex"):
    # Load dataset metadata to get board size
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

    # Select random samples
    total_samples = len(inputs)
    indices = np.random.choice(total_samples, min(total_samples, num_samples), replace=False)

    # Initialize LaTeX content
    latex_code = [
        r"\documentclass{article}",
        r"\usepackage{tikz}",
        r"\usepackage{subcaption}",
        r"\usepackage{geometry}",
        r"\usepackage{xcolor}",
        r"\geometry{a4paper, margin=1in}",
        r"\begin{document}",
        r"\section*{N-Queens Visualization}"
    ]

    for idx in indices:
        inp = inputs[idx].reshape(n_size, n_size)
        lbl = labels[idx].reshape(n_size, n_size)

        # Calculate Mask Percentage (Based on Queens)
        # Token 2 = Queen in Input/Label
        # Token 3 = Mask in Input
        total_queens = np.sum(lbl == 2)
        visible_queens = np.sum(inp == 2)
        
        if total_queens > 0:
            mask_pct = 100 * (1 - (visible_queens / total_queens))
        else:
            mask_pct = 0.0

        latex_code.append(r"\begin{figure}[h!]")
        latex_code.append(r"\centering")
        
        # --- Left Subfigure: Input ---
        latex_code.append(r"\begin{subfigure}{0.45\textwidth}")
        latex_code.append(r"\centering")
        latex_code.append(r"\begin{tikzpicture}[scale=0.5]")
        
        # Draw Grid
        latex_code.append(f"\\draw[step=1.0,gray,thin] (0,0) grid ({n_size},{n_size});")
        
        # Fill Cells
        for r in range(n_size):
            for c in range(n_size):
                val = inp[r, c]
                x = c
                y = n_size - 1 - r
                
                # STRICT CHECK: Only mask (3) gets black.
                # 0=Pad, 1=Empty (if any), 2=Queen, 3=Mask
                if val == 3: 
                    latex_code.append(f"\\fill[black] ({x},{y}) rectangle ++(1,1);")
                elif val == 2: # Given Queen
                    latex_code.append(f"\\node at ({x}+0.5, {y}+0.5) {{\\textbf{{Q}}}};")
        
        # Draw Border
        latex_code.append(r"\draw (0,0) rectangle (" + str(n_size) + "," + str(n_size) + ");")
        latex_code.append(r"\end{tikzpicture}")
        latex_code.append(f"\\caption{{Input ({mask_pct:.1f}\\% Queens Masked)}}")
        latex_code.append(r"\end{subfigure}")
        latex_code.append(r"\hfill")

        # --- Right Subfigure: Target ---
        latex_code.append(r"\begin{subfigure}{0.45\textwidth}")
        latex_code.append(r"\centering")
        latex_code.append(r"\begin{tikzpicture}[scale=0.5]")
        
        # Draw Grid
        latex_code.append(f"\\draw[step=1.0,gray,thin] (0,0) grid ({n_size},{n_size});")
        
        for r in range(n_size):
            for c in range(n_size):
                l_val = lbl[r, c]
                i_val = inp[r, c]
                x = c
                y = n_size - 1 - r
                
                if l_val == 2: # Queen in solution
                    if i_val == 2: # Was given in input
                        latex_code.append(f"\\node at ({x}+0.5, {y}+0.5) {{\\textbf{{Q}}}};")
                    else: # Was masked in input (Input was 3 or 0)
                        latex_code.append(f"\\node[red] at ({x}+0.5, {y}+0.5) {{\\textbf{{Q}}}};")
                        
        latex_code.append(r"\draw (0,0) rectangle (" + str(n_size) + "," + str(n_size) + ");")
        latex_code.append(r"\end{tikzpicture}")
        latex_code.append(r"\caption{Target (Red = Inferred)}")
        latex_code.append(r"\end{subfigure}")
        
        latex_code.append(r"\end{figure}")
        
        if idx != indices[-1]:
            latex_code.append(r"\vspace{0.5cm}")

    latex_code.append(r"\end{document}")

    with open(output_file, "w") as f:
        f.write("\n".join(latex_code))
    
    print(f"Successfully generated LaTeX file at: {output_file}")
    print(f"To compile: pdflatex {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Path to dataset subset")
    parser.add_argument("--out", type=str, default="n_queens_vis.tex", help="Output .tex filename")
    parser.add_argument("--n", type=int, default=3, help="Number of examples to generate")
    args = parser.parse_args()
    
    generate_latex(args.dir, args.n, args.out)