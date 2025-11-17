"""
Visualize samples from the Tower of Hanoi state-to-state dataset.
"""

import numpy as np
import os
import argparse
import time

# Constants matching your build script
PAD_ID = 0
PEG_A = 1
PEG_B = 2
PEG_C = 3
PEG_NAMES = {0: "PAD", 1: "A", 2: "B", 3: "C"}


def render_tower(state_array, num_disks, target_peg):
    """
    Visualizes the state of the towers in ASCII.
    state_array: [peg_disk0, peg_disk1, ..., peg_diskN, PAD..., TARGET]
    """
    # Filter out PADs to get actual disk positions
    # The last element is the target, so we slice it off first
    input_seq = state_array[:-1]
    target_val = state_array[-1]

    # In the dataset, index i is disk i (0 is smallest)
    towers = {1: [], 2: [], 3: []}

    max_disk_size = 0

    for disk_size, peg_id in enumerate(input_seq):
        if peg_id == PAD_ID:
            continue
        towers[peg_id].append(disk_size)
        max_disk_size = max(max_disk_size, disk_size)

    # Create the visual buffer
    # We need height = number of disks
    height = max_disk_size + 1
    output_lines = []

    # Header
    output_lines.append(f"Target Peg: {PEG_NAMES.get(target_val, '?')}")
    output_lines.append("-" * 40)

    # Build rows from top to bottom
    for h in range(height - 1, -1, -1):
        row_str = ""
        for peg in [1, 2, 3]:
            # Check if this peg has a disk at this height
            # towers[peg] stores disks. We need to stack them.
            # Since larger disks are usually at the bottom, we sort reversed
            # BUT: In Hanoi, if we push to list, index 0 is bottom.
            # Let's organize: Sort disks on peg descending (largest bottom)
            current_peg_disks = sorted(towers[peg], reverse=True)

            if h < len(current_peg_disks):
                d_size = current_peg_disks[h]
                # Draw disk
                width = (d_size + 1) * 2
                disk_str = f"[{d_size}]".center(12)
            else:
                # Empty air
                disk_str = "|".center(12)

            row_str += disk_str
        output_lines.append(row_str)

    output_lines.append("-" * 40)
    output_lines.append("    Peg A       Peg B       Peg C    ")

    return "\n".join(output_lines)


def visualize_dataset(data_dir, split="train", num_samples=5):
    inputs_path = os.path.join(data_dir, split, "all__inputs.npy")
    labels_path = os.path.join(data_dir, split, "all__labels.npy")

    if not os.path.exists(inputs_path):
        print(f"Error: Could not find {inputs_path}")
        return

    print(f"Loading {split} data...")
    inputs = np.load(inputs_path)
    labels = np.load(labels_path)

    total_samples = inputs.shape[0]
    print(f"Total samples: {total_samples}")

    # Pick random indices
    indices = np.random.choice(total_samples, num_samples, replace=False)

    for idx in indices:
        inp = inputs[idx]
        lbl = labels[idx]

        # Extract target from input (last token)
        target_peg = inp[-1]

        # Calculate num disks (count non-pads excluding target)
        n_disks = np.sum(inp[:-1] != PAD_ID)

        print(f"\n\n=== Sample {idx} (Disks: {n_disks}) ===")

        # Render Input
        print("--- CURRENT STATE ---")
        print(render_tower(inp, n_disks, target_peg))

        # Render Label (Prediction)
        print("\n--- NEXT STEP (LABEL) ---")
        print(render_tower(lbl, n_disks, target_peg))

        # Check what moved
        diff = inp[:-1] != lbl[:-1]
        if np.any(diff):
            moved_disk = np.where(diff)[0][0]
            from_peg = inp[moved_disk]
            to_peg = lbl[moved_disk]
            print(
                f"\n>>> ACTION: Moved Disk {moved_disk} from {PEG_NAMES[from_peg]} to {PEG_NAMES[to_peg]}"
            )
        else:
            print("\n>>> ACTION: No Move (End state or error?)")

        print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="data/hanoi", help="Path to data folder"
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument(
        "--samples", type=int, default=5, help="Number of random samples to show"
    )

    args = parser.parse_args()

    visualize_dataset(args.data_dir, args.split, args.samples)
