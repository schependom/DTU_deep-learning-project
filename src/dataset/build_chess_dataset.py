import os
import csv
import json
import argparse
import numpy as np
from tqdm import tqdm

from common import PuzzleDatasetMetadata

"""
This file uses the chess position / move csv created from the lichess data-base. Build the csv first with create_chess_csv.py.
"""


PAD = 0
IGNORE_LABEL_ID = 0

# FEN TOKENIZATION

FEN_CHARS = list("prnbqkPRNBQK12345678/ wb-abcdefgh")
FEN_CHARS = sorted(set(FEN_CHARS), key=lambda x: FEN_CHARS.index(x))
itos_fen = ["<PAD>"] + FEN_CHARS
stoi_fen = {ch: i for i, ch in enumerate(itos_fen)}
FEN_VOCAB_SIZE = len(itos_fen)

MAX_FEN_LEN = 80


def encode_fen_to_ids(fen: str, max_len: int = MAX_FEN_LEN) -> np.ndarray:
    ids = [stoi_fen[ch] for ch in fen if ch in stoi_fen]
    ids = ids[:max_len]
    ids = ids + [PAD] * (max_len - len(ids))
    return np.array(ids, dtype=np.uint16)



# MOVE TOKENIZATION

def square_to_idx(sq: str) -> int:
    file = "abcdefgh".index(sq[0])
    rank = int(sq[1]) - 1
    return rank * 8 + file


MOVE_BASE_SIZE = 64 * 64
PROMO_EXTRA = 4
MOVE_VOCAB_SIZE = MOVE_BASE_SIZE + PROMO_EXTRA


def uci_to_move_id(uci: str) -> int:
    if len(uci) < 4:
        return 0
    from_idx = square_to_idx(uci[:2])
    to_idx   = square_to_idx(uci[2:4])
    base = from_idx * 64 + to_idx
    return base  # promotions ignored for simplicity


LABEL_OFFSET = FEN_VOCAB_SIZE
MODEL_VOCAB_SIZE = FEN_VOCAB_SIZE + MOVE_VOCAB_SIZE


# DATASET CREATION 

def load_chess_rows(csv_path: str, fen_col: str = "FEN", move_col: str = "Best_Move"):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if move_col not in reader.fieldnames:
            raise ValueError(f"CSV must contain '{move_col}' column.")
        for r in reader:
            fen = r[fen_col].strip()
            move = r[move_col].strip()
            if fen == "" or move == "":
                continue
            rows.append((fen, move))
    return rows


def process_subset(set_name: str, samples, output_dir: str):
    """
    EXACT same output structure as N-Queens:
    - inputs:   npy matrix [N, seq_len]
    - labels:   npy matrix [N, seq_len]
    - puzzle_indices: boundaries for puzzle groups
    - group_indices: same
    """
    results = {
        "inputs": [],
        "labels": [],
        "puzzle_identifiers": [],
        "puzzle_indices": [0],
        "group_indices": [0],
    }

    example_id = 0
    puzzle_id = 0

    seq_len = 1 + MAX_FEN_LEN

    for fen, move in tqdm(samples, desc=f"Processing {set_name}"):
        # Encode inputs
        fen_ids = encode_fen_to_ids(fen)
        move_id = uci_to_move_id(move)
        label_token = LABEL_OFFSET + move_id

        inp = np.zeros(seq_len, dtype=np.uint16)
        lab = np.zeros(seq_len, dtype=np.int32)

        inp[0] = PAD
        inp[1:] = fen_ids

        lab[0] = label_token
        lab[1:] = IGNORE_LABEL_ID

        results["inputs"].append(inp)
        results["labels"].append(lab)
        results["puzzle_identifiers"].append(0)

        example_id += 1
        puzzle_id += 1

        results["puzzle_indices"].append(example_id)
        results["group_indices"].append(puzzle_id)

    final = {
        "inputs": np.stack(results["inputs"]),
        "labels": np.stack(results["labels"]),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
    }

    os.makedirs(os.path.join(output_dir, set_name), exist_ok=True)
    for k, v in final.items():
        np.save(os.path.join(output_dir, set_name, f"all__{k}.npy"), v)

    metadata = PuzzleDatasetMetadata(
        seq_len=seq_len,
        vocab_size=MODEL_VOCAB_SIZE,
        pad_id=PAD,
        ignore_label_id=IGNORE_LABEL_ID,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(final["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(final["group_indices"]) - 1,
        sets=["all"],
    )

    with open(os.path.join(output_dir, set_name, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    with open(os.path.join(output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


def build_dataset(csv_path: str, output_dir: str, train_ratio=0.9):
    rows = load_chess_rows(csv_path)

    np.random.shuffle(rows)
    split = int(len(rows) * train_ratio)
    train_rows = rows[:split]
    test_rows = rows[split:]

    process_subset("train", train_rows, output_dir)
    process_subset("test", test_rows, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to chess positions csv")
    parser.add_argument("--out", type=str, default="data/chess_trm")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)
    build_dataset(args.csv, args.out)
