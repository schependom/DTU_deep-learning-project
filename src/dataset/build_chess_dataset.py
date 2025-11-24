import csv
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


PAD = 0
IGNORE_LABEL_ID = -100

# --- FEN token vocabulary (simple char-level mapping) ---
FEN_CHARS = list("prnbqkPRNBQK12345678/ wb-abcdefgh")  # includes file letters for en-passant
# make deterministic ordering
FEN_CHARS = sorted(set(FEN_CHARS), key=lambda x: FEN_CHARS.index(x) if x in FEN_CHARS else 0)
itos_fen = ["<PAD>"] + FEN_CHARS
stoi_fen = {ch: i for i, ch in enumerate(itos_fen)}
FEN_VOCAB_SIZE = len(itos_fen)

# --- Move vocabulary (from->to) if available in CSV 'best_move' column ---
# 64 squares => encode square index 0..63 as a1=0 .. h8=63 (we'll use simple mapping)
def square_to_idx(sq: str) -> int:
    # sq is like 'e2' or '-' (we won't see '-' for moves)
    if len(sq) != 2:
        return 0
    file = "abcdefgh".index(sq[0])
    rank = int(sq[1]) - 1
    return rank * 8 + file

# make move id as from*64 + to (0..4095)
MOVE_BASE_SIZE = 64 * 64  # 4096
# keep a small promotion extension if needed
PROMO_EXTRA = 4
MOVE_VOCAB_SIZE = MOVE_BASE_SIZE + PROMO_EXTRA

# --- Eval bucketing fallback ---
# If no best_move column exists, we'll bucket evals into discrete tokens:
EVAL_MIN = -1200.0
EVAL_MAX = 1200.0
EVAL_BUCKETS = 256  # number of discrete buckets for evaluation
def eval_to_bucket(x: float) -> int:
    x = max(min(x, EVAL_MAX), EVAL_MIN)
    # normalize to [0, 1]
    u = (x - EVAL_MIN) / (EVAL_MAX - EVAL_MIN)
    b = int(u * (EVAL_BUCKETS - 1))
    return b

EVAL_VOCAB_SIZE = EVAL_BUCKETS

# We'll choose label_vocab_size = max(MOVE_VOCAB_SIZE, EVAL_VOCAB_SIZE)
LABEL_VOCAB_SIZE = max(MOVE_VOCAB_SIZE, EVAL_VOCAB_SIZE)

# Combined model vocab = fen_vocab + label_vocab (labels occupy indices [fen_vocab .. fen_vocab+label_vocab-1])
LABEL_OFFSET = FEN_VOCAB_SIZE
MODEL_VOCAB_SIZE = FEN_VOCAB_SIZE + LABEL_VOCAB_SIZE

# ---------------------------
# FEN encoding utilities
MAX_FEN_LEN = 80  # choose safe upper bound (position 0 will be label token)

def encode_fen_to_ids(fen: str, max_len: int = MAX_FEN_LEN) -> List[int]:
    ids = []
    for ch in fen:
        if ch in stoi_fen:
            ids.append(stoi_fen[ch])
        else:
            # unknown char: try to split or skip
            # (rare)
            continue
    # truncate/pad (we will later prefix label token at position 0)
    ids = ids[:max_len]
    pad_len = max_len - len(ids)
    ids = ids + [PAD] * pad_len
    return ids

def uci_to_move_id(uci: str) -> int:
    # uci example: "e2e4", "e7e8q" (promotion)
    if len(uci) < 4:
        return 0
    from_sq = uci[0:2]
    to_sq = uci[2:4]
    from_idx = square_to_idx(from_sq)
    to_idx = square_to_idx(to_sq)
    base_id = from_idx * 64 + to_idx
    # promotions: if present, map to extra ids in last PROMO_EXTRA slots
    if len(uci) > 4:
        promo_char = uci[4].lower()
        promo_map = {"n": 0, "b": 1, "r": 2, "q": 3}
        promo_idx = promo_map.get(promo_char, 0)
        return base_id  # for simplicity we don't expand into big space; keep promotions collapsed
    return base_id

# ---------------------------
# Dataset that yields the TRM-style batch (inputs with label token at pos 0)
class ChessFenDataset(Dataset):
    def __init__(self, csv_path: str, fen_col: str = "FEN", eval_col: str = "Evaluation", move_col: Optional[str] = "best_move"):
        self.rows = []
        self.has_move_col = False

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)

            # detect columns
            header = reader.fieldnames or []
            self.has_move_col = (move_col in header)

            for r in reader:
                fen = r[fen_col].strip()

                # --- TRY PARSING EVAL AS FLOAT ---
                eval_raw = r.get(eval_col, "").strip()
                try:
                    eval_v = float(eval_raw)
                except ValueError:
                    # ignore rows that don't contain a valid float
                    continue

                move = r.get(move_col, "").strip() if self.has_move_col else ""

                self.rows.append({
                    "fen": fen,
                    "eval": eval_v,
                    "move": move
                })

        self.fen_len = MAX_FEN_LEN
        self.seq_len = 1 + self.fen_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        fen_ids = encode_fen_to_ids(r["fen"], max_len=self.fen_len)  # length fen_len
        # choose label token id (in combined vocab)
        if self.has_move_col and r["move"]:
            move_id = uci_to_move_id(r["move"])
            label_token = LABEL_OFFSET + (move_id % LABEL_VOCAB_SIZE)
        else:
            b = eval_to_bucket(r["eval"])
            label_token = LABEL_OFFSET + (b % LABEL_VOCAB_SIZE)

        # Build inputs: length seq_len: position 0 is label token, remainder are fen IDs
        inputs = [label_token] + fen_ids
        # labels: set IGNORE everywhere except pos 0 where the target is the same label token
        labels = [label_token] + [IGNORE_LABEL_ID] * self.fen_len

        # puzzle_identifiers: zeros (unused). Keep dtype torch.long
        return {
            "inputs": torch.tensor(inputs, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "puzzle_identifiers": torch.tensor(0, dtype=torch.long)
        }

# collate that stacks dict entries into batch dict expected by TRM
def collate_batch(batch_list):
    B = len(batch_list)
    seq_len = batch_list[0]["inputs"].shape[0]
    inputs = torch.stack([b["inputs"] for b in batch_list], dim=0)
    labels = torch.stack([b["labels"] for b in batch_list], dim=0)
    puzzle_ids = torch.stack([b["puzzle_identifiers"] for b in batch_list], dim=0)
    return {"inputs": inputs, "labels": labels, "puzzle_identifiers": puzzle_ids}


if __name__ == "__main__":
    dataset = ChessFenDataset(csv_path = r"C:\Users\malth\Documents\DTU\Master\Andet Semester\Deep Learning\DTU_deep-learning-project\src\dataset\chessData.csv")
    print(dataset[0])