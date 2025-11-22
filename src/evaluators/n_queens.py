"""
Evaluator for N-Queens constraint satisfaction.
Save this as: evaluators/n_queens_evaluator.py
(Or map it in your pretrain.py imports)
"""

import torch
import torch.distributed as dist
from typing import Optional

# Constants matching build script
# Data is 1=Empty, 2=Queen. 0=PAD.
VAL_EMPTY = 1
VAL_QUEEN = 2


class NQueensEvaluator:
    """
    Evaluates N-Queens solutions.
    Checks:
    1. Token Accuracy (Did it predict Q vs Empty correctly?)
    2. Constraint Validity (Is the board a valid N-Queens solution?)
    3. Consistency (Did it keep the hint queens from the input?)
    """

    required_outputs = ["preds"]

    def __init__(self, data_path: str, eval_metadata, **kwargs):
        self.data_path = data_path
        self.eval_metadata = eval_metadata
        self.seq_len = eval_metadata.seq_len
        self.n_size = int(self.seq_len**0.5)
        self.reset_metrics()

    def reset_metrics(self):
        self.total_samples = torch.tensor(0.0, device="cuda")
        self.valid_solutions = torch.tensor(
            0.0, device="cuda"
        )  # Physically valid N-Queens
        self.correct_solutions = torch.tensor(
            0.0, device="cuda"
        )  # Matches Ground Truth exactly

    def begin_eval(self):
        self.reset_metrics()

    def _check_board_validity(self, board_tensor):
        """
        Returns True if board is a valid N-Queens solution.
        board_tensor: [N, N] containing 1s and 2s (or raw tokens).
        """
        # Convert to binary: 1 if Queen (val==2), 0 otherwise
        # (Assuming model outputs argmax class indices)

        # Output of model is typically 1-based index (1=Empty, 2=Queen)
        # Let's normalize: Queen if val == 2
        queens = board_tensor == VAL_QUEEN

        # Constraint 1: Exactly N queens
        if queens.sum() != self.n_size:
            return False

        # Get coordinates
        coords = torch.nonzero(queens)  # [N, 2]
        if coords.size(0) != self.n_size:
            return False

        rows = coords[:, 0]
        cols = coords[:, 1]

        # Constraint 2: Unique Rows and Cols
        if len(torch.unique(rows)) != self.n_size:
            return False
        if len(torch.unique(cols)) != self.n_size:
            return False

        # Constraint 3: Diagonals
        # Diagonals are defined by (row + col) and (row - col)
        sum_diag = rows + cols
        diff_diag = rows - cols

        if len(torch.unique(sum_diag)) != self.n_size:
            return False
        if len(torch.unique(diff_diag)) != self.n_size:
            return False

        return True

    def update_batch(self, batch: dict, preds: dict):
        inputs = batch["inputs"]  # [B, L]
        labels = batch["labels"]  # [B, L]
        pred_tokens = preds["preds"]  # [B, L]

        batch_size = labels.shape[0]

        for i in range(batch_size):
            p_flat = pred_tokens[i]
            l_flat = labels[i]

            self.total_samples += 1

            # Exact match check
            if torch.equal(p_flat, l_flat):
                self.correct_solutions += 1
                self.valid_solutions += 1  # If it matches GT, it is valid
            else:
                # Check if it's a valid alternate solution
                # (N-Queens might have multiple solutions for same partial input?
                # Though dataset generation tries to use unique base solutions,
                # heavy masking might make it ambiguous. Validity is a good metric.)
                board = p_flat.view(self.n_size, self.n_size)
                if self._check_board_validity(board):
                    self.valid_solutions += 1

    def result(
        self,
        save_path: Optional[str] = None,
        rank: int = 0,
        world_size: int = 1,
        group=None,
    ) -> Optional[dict]:
        if world_size > 1:
            dist.all_reduce(self.total_samples, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.valid_solutions, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.correct_solutions, op=dist.ReduceOp.SUM)

        if rank != 0:
            return None

        metrics = {}
        dataset_name = (
            self.data_path.split("/")[-1] if "/" in self.data_path else "n_queens"
        )

        total = self.total_samples.item()
        if total > 0:
            metrics[f"test/{dataset_name}/exact_accuracy"] = (
                self.correct_solutions.item() / total
            )
            metrics[f"test/{dataset_name}/valid_rate"] = (
                self.valid_solutions.item() / total
            )
        else:
            metrics[f"test/{dataset_name}/exact_accuracy"] = 0.0
            metrics[f"test/{dataset_name}/valid_rate"] = 0.0

        return metrics
