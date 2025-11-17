import os
import json
import numpy as np
import argparse

# --- Constants ---
PAD_ID = 0
PEG_A = 1
PEG_B = 2
PEG_C = 3
PEG_NAMES = {PAD_ID: "PAD", PEG_A: "A", PEG_B: "B", PEG_C: "C"}
ACTION_DISK_OFFSET = 4


def load_metadata(data_dir):
    """Loads dataset.json (Standard format)"""
    path = os.path.join(data_dir, "dataset.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata not found at {path}")
    
    with open(path, "r") as f:
        return json.load(f)


def infer_config(meta):
    """
    Infers encoding type and actual number of disks from metadata.
    """
    vocab_size = meta.get("vocab_size", 0)
    
    # Heuristic:
    # Action encoding: vocab_size = 4 + max_disks
    # State encoding:  vocab_size = 4
    
    if vocab_size > 4:
        encoding = "action"
        max_disks = vocab_size - 4
    else:
        encoding = "state"
        # If state encoding, we can't easily guess max_disks from vocab.
        # We'll default to a reasonable guess or try to scan the data if needed.
        # For visualization purposes, 10 is a safe upper bound for rendering width.
        max_disks = 10 

    return encoding, max_disks


def strip_padding(vec):
    """
    Finds the target peg (last non-zero element before padding) and the state.
    Input format: [disk_0, disk_1, ..., disk_N, target, PAD, PAD...]
    """
    # Find the last non-zero index. 
    # However, disks are 1,2,3. Target is 1,2,3.
    # PAD is 0.
    
    # Get indices where value != 0
    valid_indices = np.where(vec != 0)[0]
    
    if len(valid_indices) == 0:
        return np.array([]), 0 # Empty
        
    last_valid_idx = valid_indices[-1]
    
    # The target is the very last valid token.
    target_peg = vec[last_valid_idx]
    
    # The state is everything before that.
    state_vec = vec[:last_valid_idx]
    
    return state_vec, target_peg


def render_tower_ascii(state_vec, target_peg, max_disks_render, title="State"):
    """
    Generates a visual ASCII representation of the Hanoi Board.
    """
    state_vec = state_vec.astype(int)
    
    # Organize disks onto pegs
    towers = {PEG_A: [], PEG_B: [], PEG_C: []}

    for disk_size, peg_id in enumerate(state_vec):
        if peg_id in towers:
            towers[peg_id].append(disk_size)

    # Sort disks: largest at bottom
    for p in towers:
        towers[p].sort(reverse=True)

    # Dimensions
    col_width = (max_disks_render * 2) + 6
    height = max_disks_render + 1

    lines = []
    lines.append(f" {title} ".center(col_width * 3, "═"))
    lines.append(
        f" Goal: Move all to Peg {PEG_NAMES.get(target_peg, '?')} ".center(
            col_width * 3
        )
    )
    lines.append("")  # spacer

    # Build rows from top to bottom
    for h in range(height - 1, -1, -1):
        row_str = ""
        for peg in [PEG_A, PEG_B, PEG_C]:
            disk_stack = towers[peg]

            if h < len(disk_stack):
                # Draw Disk
                disk_val = disk_stack[h]
                width_val = disk_val + 1
                bar = "=" * width_val
                disk_str = f"{bar}({disk_val}){bar}"
                row_str += disk_str.center(col_width)
            else:
                # Draw Pole
                row_str += "║".center(col_width)
        lines.append(row_str)

    # Base
    base_chunk = "╩".center(col_width, "═")
    lines.append(base_chunk + base_chunk + base_chunk)

    # Labels
    labels = (
        f"Peg A".center(col_width)
        + f"Peg B".center(col_width)
        + f"Peg C".center(col_width)
    )
    lines.append(labels)

    return "\n".join(lines)


def decode_action(label_vec):
    """Decodes [disk_token, dest_peg, ..., target, PAD...]"""
    # In the new format, the first two tokens are always action
    disk_token = int(label_vec[0])
    dest_peg = int(label_vec[1])

    if disk_token < ACTION_DISK_OFFSET:
        return "Invalid Action Token"

    disk_id = disk_token - ACTION_DISK_OFFSET
    dest_name = PEG_NAMES.get(dest_peg, "?")
    return f"👉 ACTION: Move Disk {disk_id} ➜ Peg {dest_name}"


def visualize(data_dir, num_samples=3):
    print(f"\n📂 Opening Dataset: {data_dir}")
    
    # 1. Load Metadata
    try:
        meta = load_metadata(data_dir)
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return

    # 2. Infer Config
    encoding, max_disks = infer_config(meta)
    print(f"   Inferred Encoding: {encoding.upper()}")
    print(f"   Inferred Max Disks (from vocab): {max_disks}")

    # 3. Load Data (Standard Format)
    inputs_path = os.path.join(data_dir, "all__inputs.npy")
    labels_path = os.path.join(data_dir, "all__labels.npy")
    
    try:
        inputs = np.load(inputs_path)
        labels = np.load(labels_path)
    except Exception as e:
        print(f"❌ Error loading numpy arrays: {e}")
        return

    print(f"   Samples Found: {len(inputs)}")
    print("-" * 60)

    # Pick random samples
    indices = np.random.choice(
        len(inputs), min(num_samples, len(inputs)), replace=False
    )

    for i, idx in enumerate(indices):
        inp = inputs[idx]
        lbl = labels[idx]

        print(f"\n🔎 SAMPLE #{idx}")

        # Strip Padding to get State and Target
        state_vec, target_peg = strip_padding(inp)
        
        # Because max_disks is inferred from vocab, it might be larger than the current sample's disk count.
        # We use the state_vec length to determine current disk count for rendering nicely.
        current_disks = len(state_vec)
        
        print(render_tower_ascii(state_vec, target_peg, max_disks_render=max(current_disks, 3), title="Current State"))

        if encoding == "action":
            # Render Action
            action_text = decode_action(lbl)
            print(f"\n{action_text}")
            print("=" * 60)

        elif encoding == "state":
            # Render Next State
            next_state_vec, _ = strip_padding(lbl)
            print("\n      ⬇ ⬇ ⬇ PREDICTED NEXT STATE ⬇ ⬇ ⬇\n")
            print(render_tower_ascii(next_state_vec, target_peg, max_disks_render=max(current_disks, 3), title="Target Next State"))
            print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Dataset directory (e.g. data/hanoi/train)",
    )
    parser.add_argument("--n", type=int, default=3, help="Number of samples to show")
    args = parser.parse_args()

    visualize(args.dir, args.n)