import csv
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader

PAD = 0
IGNORE_LABEL_ID = -100

# ---------------------------
# FEN TOKENIZATION
# ---------------------------

FEN_CHARS = list("prnbqkPRNBQK12345678/ wb-abcdefgh")
FEN_CHARS = sorted(set(FEN_CHARS), key=lambda x: FEN_CHARS.index(x))
itos_fen = ["<PAD>"] + FEN_CHARS
stoi_fen = {ch: i for i, ch in enumerate(itos_fen)}
FEN_VOCAB_SIZE = len(itos_fen)

MAX_FEN_LEN = 80   # Safe max length of FEN string

def encode_fen_to_ids(fen: str, max_len: int = MAX_FEN_LEN) -> List[int]:
    ids = [stoi_fen[ch] for ch in fen if ch in stoi_fen]
    ids = ids[:max_len]
    return ids + [PAD] * (max_len - len(ids))


# ---------------------------
# MOVE TOKENIZATION (uci)
# ---------------------------

def square_to_idx(sq: str) -> int:
    file = "abcdefgh".index(sq[0])
    rank = int(sq[1]) - 1
    return rank * 8 + file

MOVE_BASE_SIZE = 64 * 64         # 4096 possible from→to
PROMO_EXTRA = 4                  # optional (not used here but room)
MOVE_VOCAB_SIZE = MOVE_BASE_SIZE + PROMO_EXTRA

def uci_to_move_id(uci: str) -> int:
    """ Encode UCI like 'e2e4', 'e7e8q' into integer ID. """
    if len(uci) < 4:
        return 0
    from_idx = square_to_idx(uci[0:2])
    to_idx   = square_to_idx(uci[2:4])
    base_id  = from_idx * 64 + to_idx

    # Collapse promotions (optional)
    if len(uci) > 4:
        promo = uci[4].lower()
        promo_map = {"n": 0, "b": 1, "r": 2, "q": 3}
        return base_id  # (simplified – no +promo)

    return base_id


# ---------------------------
# COMBINED VOCAB
# ---------------------------

LABEL_OFFSET = FEN_VOCAB_SIZE
MODEL_VOCAB_SIZE = FEN_VOCAB_SIZE + MOVE_VOCAB_SIZE


# ---------------------------
# DATASET (MOVE PREDICTION ONLY)
# ---------------------------

class ChessFenMoveDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        fen_col: str = "FEN",
        move_col: str = "Best_Move",
    ):
        self.rows = []

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []

            if move_col not in header:
                raise ValueError(f"CSV must contain a '{move_col}' column for move prediction.")

            for r in reader:
                fen = r[fen_col].strip()
                move = r[move_col].strip()

                # skip rows with no move
                if move == "":
                    continue

                self.rows.append({
                    "fen": fen,
                    "move": move
                })

        self.fen_len = MAX_FEN_LEN
        self.seq_len = 1 + self.fen_len   # label token + FEN tokens

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]

        fen_ids = encode_fen_to_ids(r["fen"], max_len=self.fen_len)

        # ----------------------
        # MOVE LABEL TOKEN
        # ----------------------
        move_id = uci_to_move_id(r["move"])
        label_token = LABEL_OFFSET + move_id  # convert into model-wide vocab id

        # ----------------------
        # INPUTS / LABELS
        # ----------------------
        inputs = [label_token] + fen_ids
        labels = [label_token] + [IGNORE_LABEL_ID] * self.fen_len

        return {
            "inputs": torch.tensor(inputs, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "puzzle_identifiers": torch.tensor(0, dtype=torch.long)
        }


# ---------------------------
# COLLATE (TRM EXPECTS THIS SHAPE)
# ---------------------------

def collate_batch(batch_list):
    inputs = torch.stack([b["inputs"] for b in batch_list])
    labels = torch.stack([b["labels"] for b in batch_list])
    puzzle_ids = torch.stack([b["puzzle_identifiers"] for b in batch_list])
    return {
        "inputs": inputs,
        "labels": labels,
        "puzzle_identifiers": puzzle_ids
    }


# -------------------------------------------------------------------------
# TEST
# -------------------------------------------------------------------------

if __name__ == "__main__":
    dataset = ChessFenMoveDataset(csv_path = r"C:\Users\malth\Documents\DTU\Master\Andet Semester\Deep Learning\DTU_deep-learning-project\src\dataset\chess_moves.csv")
    print(dataset[0])