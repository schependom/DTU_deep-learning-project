"""
Visualize samples from the Tower of Hanoi state-to-state (alternative) dataset.
"""

import os
import json
import numpy as np
import argparse

# --- Constants (must match build_hanoi_trm.py) ---
PAD_ID = 0
PEG_A = 1
PEG_B = 2
PEG_C = 3
PEG_NAMES = {PAD_ID: "PAD", PEG_A: "A", PEG_B: "B", PEG_C: "C"}


def render_state_ascii(state_vector, num_disks):
    """
    Renders a single state vector as ASCII art.
    state_vector format: [Peg_D0, Peg_D1, ..., Peg_DN, PAD, ..., TARGET]
    """

    target_peg = state_vector[-1]
    disk_pegs = state_vector[:-1]  # Slice off the target token

    # Max width for a peg's column. Based on the largest disk.
    # Disk 'N' string is '=(N)=' -> len 2*N + 3
    if num_disks == 0:
        max_width = 5  # for "  |  "
    else:
        # Width based on largest possible disk in this puzzle
        max_disk_size = num_disks - 1
        max_width = (max_disk_size * 2) + 3

    # 1. Build the tower data structure
    towers = {PEG_A: [], PEG_B: [], PEG_C: []}
    for disk_size, peg_id in enumerate(disk_pegs):
        if peg_id != PAD_ID:
            towers[peg_id].append(disk_size)

    # 2. Sort disks on each peg (largest at bottom, index 0)
    for peg_id in towers:
        towers[peg_id].sort(reverse=True)  # e.g., [3, 2] -> Disk 3 is at index 0

    # 3. Build ASCII lines from top to bottom
    output_lines = []
    output_lines.append(f"  Target Peg: {PEG_NAMES[target_peg]}\n")

    # Iterate h from top-most row (num_disks - 1) down to bottom row (0)
    for h in range(num_disks - 1, -1, -1):
        row_str = ""
        for peg_id in [PEG_A, PEG_B, PEG_C]:
            peg_disks = towers[peg_id]  # e.g., [3, 2]

            # --- BUG FIX ---
            # Original logic was mapping h (top-down) to the wrong disk.
            # New logic: Check if row 'h' (from bottom) has a disk.
            # peg_disks[0] is bottom (h=0)
            # peg_disks[1] is next (h=1)

            if h < len(peg_disks):
                # This row has a disk.
                # h=0 maps to peg_disks[0] (largest)
                # h=1 maps to peg_disks[1]
                disk_size = peg_disks[h]
                disk_str = ("=" * disk_size) + f"({disk_size})" + ("=" * disk_size)
                row_str += disk_str.center(max_width)
            else:
                # This row is empty, draw pole
                row_str += "|".center(max_width)

            row_str += "  "  # Spacer between pegs

        output_lines.append(row_str)

    # 4. Add base
    base_line = ("-" * max_width).center(max_width)
    peg_a_line = f"Peg {PEG_NAMES[PEG_A]}".center(max_width)
    peg_b_line = f"Peg {PEG_NAMES[PEG_B]}".center(max_width)
    peg_c_line = f"Peg {PEG_NAMES[PEG_C]}".center(max_width)

    output_lines.append(base_line + "  " + base_line + "  " + base_line)
    output_lines.append(peg_a_line + "  " + peg_b_line + "  " + peg_c_line)

    return "\n".join(output_lines)


def find_move(input_state, label_state):
    """Finds which disk moved between two states."""

    # Compare only the disk positions, not the target token
    diff = np.where(input_state[:-1] != label_state[:-1])[0]

    if len(diff) == 0:
        return "  ACTION: No Move (Initial or Final State)"

    # In optimal Hanoi, only one disk moves at a time
    disk_moved = diff[0]

    from_peg_id = input_state[disk_moved]
    to_peg_id = label_state[disk_moved]

    return f"  ACTION: Moved Disk {disk_moved} from Peg {PEG_NAMES[from_peg_id]} to Peg {PEG_NAMES[to_peg_id]}"


def visualize(data_dir, split, num_samples):
    print(f"Loading data from: {data_dir}/{split}")

    inputs_path = os.path.join(data_dir, split, "inputs.npy")
    labels_path = os.path.join(data_dir, split, "labels.npy")
    meta_path = os.path.join(data_dir, split, "metadata.json")

    if not all(os.path.exists(p) for p in [inputs_path, labels_path, meta_path]):
        print(f"Error: Could not find required files in {data_dir}/{split}")
        print("Please run `build_hanoi_trm.py` first.")
        return

    # Load data
    inputs = np.load(inputs_path)
    labels = np.load(labels_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    seq_len = metadata["seq_len"]

    print(f"Total samples in '{split}' set: {len(inputs)}")
    print(f"Sequence Length (Max Disks + Target): {seq_len}\n")

    # Select random samples
    if len(inputs) < num_samples:
        print(
            f"Warning: Not enough samples to show {num_samples}, showing all {len(inputs)}"
        )
        indices = np.arange(len(inputs))
    else:
        indices = np.random.choice(len(inputs), num_samples, replace=False)

    for i, idx in enumerate(indices):
        inp_vec = inputs[idx]
        lbl_vec = labels[idx]

        # Find num_disks by counting non-pad tokens (excl. target)
        num_disks = np.sum(inp_vec[:-1] != PAD_ID)
        action = find_move(inp_vec, lbl_vec)

        print("=" * 80)
        print(f"SAMPLE {i + 1} (Index: {idx}) | Total Disks: {num_disks}")
        print("=" * 80)

        # --- Render INPUT (Current State) ---
        print("\n--- INPUT (Current State) ---")
        print(render_state_ascii(inp_vec, num_disks))

        # --- Render LABEL (Next State) ---
        print("\n--- LABEL (Next State) ---")
        print(render_state_ascii(lbl_vec, num_disks))

        # --- Render Action ---
        print(f"\n{action}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Tower of Hanoi TRM dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/hanoi_trm",
        help="Path to the root data directory (e.g., 'data/hanoi_trm')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which split to visualize",
    )
    parser.add_argument(
        "--samples", type=int, default=5, help="Number of random samples to display"
    )
    args = parser.parse_args()

    visualize(args.data_dir, args.split, args.samples)
