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
ACTION_DISK_OFFSET = 4  # Matches build script


def load_metadata(data_dir):
    """Loads dataset.json (new format)"""
    path = os.path.join(data_dir, "dataset.json")
    if not os.path.exists(path):
        # Fallback for backward compatibility or error
        old_path = os.path.join(data_dir, "metadata.json")
        if os.path.exists(old_path):
            print(f"⚠️  Warning: Found 'metadata.json' instead of 'dataset.json'. Using legacy format.")
            with open(old_path, "r") as f:
                return json.load(f)
        raise FileNotFoundError(f"Metadata not found at {path}")
    
    with open(path, "r") as f:
        return json.load(f)


def infer_config(meta):
    """
    Infers encoding type and max_disks from standard metadata fields 
    (vocab_size, seq_len) since specific fields are no longer stored.
    """
    # If legacy metadata with explicit fields
    if "encoding_type" in meta:
        return meta["encoding_type"], meta.get("max_disks", 5)

    vocab_size = meta.get("vocab_size", 0)
    seq_len = meta.get("seq_len", 0)

    # Heuristic based on build_hanoi_dataset.py logic:
    # Action encoding: vocab_size = 4 + max_disks
    # State encoding:  vocab_size = 4
    
    if vocab_size > 4:
        # Must be action encoding
        max_disks = vocab_size - 4
        return "action", max_disks
    else:
        # State encoding (vocab is just PEG_A..C + PAD)
        # Seq len = max_disks + 1 (state + target)
        max_disks = seq_len - 1
        return "state", max_disks


def render_tower_ascii(state_vec, max_disks_global, title="State"):
    """
    Generates a visual ASCII representation of the Hanoi Board.
    state_vec: [disk_0_peg, disk_1_peg, ..., target_peg]
    """
    # Ensure we are working with integers
    state_vec = state_vec.astype(int)
    
    target_peg = state_vec[-1]
    disks = state_vec[:-1]

    # 1. Organize disks onto pegs
    # towers[peg_id] = [list of disk sizes (0=smallest)]
    towers = {PEG_A: [], PEG_B: [], PEG_C: []}

    for disk_size, peg_id in enumerate(disks):
        if peg_id in towers:  # Ignore PAD (0)
            towers[peg_id].append(disk_size)

    # Sort disks: largest at bottom (index 0)
    for p in towers:
        towers[p].sort(reverse=True)

    # 2. Dimensions
    col_width = (max_disks_global * 2) + 6
    height = max_disks_global + 1

    lines = []
    lines.append(f" {title} ".center(col_width * 3, "═"))
    lines.append(
        f" Goal: Move all to Peg {PEG_NAMES.get(target_peg, '?')} ".center(
            col_width * 3
        )
    )
    lines.append("")  # spacer

    # 3. Build rows from top to bottom
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

    # 4. Base
    base_chunk = "╩".center(col_width, "═")
    lines.append(base_chunk + base_chunk + base_chunk)

    # 5. Labels
    labels = (
        f"Peg A".center(col_width)
        + f"Peg B".center(col_width)
        + f"Peg C".center(col_width)
    )
    lines.append(labels)

    return "\n".join(lines)


def decode_action(label_vec, target_peg):
    """Decodes [disk_token, dest_peg, ..., target]"""
    disk_token = int(label_vec[0])
    dest_peg = int(label_vec[1])

    if disk_token == PAD_ID:
        return "No Action / Padding"

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

    # 2. Infer Configuration
    encoding, max_disks = infer_config(meta)
    print(f"   Inferred Encoding: {encoding.upper()}")
    print(f"   Inferred Max Disks: {max_disks}")

    # 3. Load Data (Try new name first, then old)
    inputs_path = os.path.join(data_dir, "all__inputs.npy")
    labels_path = os.path.join(data_dir, "all__labels.npy")
    
    if not os.path.exists(inputs_path):
        inputs_path = os.path.join(data_dir, "inputs.npy")
        labels_path = os.path.join(data_dir, "labels.npy")

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

        # Render Input State
        print(render_tower_ascii(inp, max_disks, title="Current State"))

        if encoding == "action":
            # Render Action Text
            target = inp[-1]
            action_text = decode_action(lbl, target)
            print(f"\n{action_text}")
            print("=" * 60)

        elif encoding == "state":
            # Render Next State below
            print("\n      ⬇ ⬇ ⬇ PREDICTED NEXT STATE ⬇ ⬇ ⬇\n")
            print(render_tower_ascii(lbl, max_disks, title="Target Next State"))
            print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Dataset directory (e.g. data/hanoi/train) containing dataset.json and npy files",
    )
    parser.add_argument("--n", type=int, default=3, help="Number of samples to show")
    args = parser.parse_args()

    visualize(args.dir, args.n)