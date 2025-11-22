import torch
import torch.distributed as dist
import wandb
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

VAL_EMPTY = 1
VAL_QUEEN = 2

class NQueensEvaluator:
    required_outputs = ["preds"]

    def __init__(self, data_path: str, eval_metadata, **kwargs):
        self.data_path = data_path
        self.n_size = int(eval_metadata.seq_len ** 0.5)
        self.reset_metrics()

    def reset_metrics(self):
        self.total = 0
        self.correct = 0
        self.valid = 0
        self.row_errors = 0
        self.col_errors = 0
        self.diag_errors = 0
        self.count_errors = 0
        self.sample_boards = [] # Store a few for visualization

    def begin_eval(self):
        self.reset_metrics()

    def update_batch(self, batch: dict, preds: dict):
        inputs = batch["inputs"]
        labels = batch["labels"]
        pred_tokens = preds["preds"]
        
        # Convert to CPU for detailed analysis (slow but necessary for debug)
        preds_cpu = pred_tokens.cpu()
        inputs_cpu = inputs.cpu()
        
        for i in range(preds_cpu.shape[0]):
            self.total += 1
            board = preds_cpu[i].view(self.n_size, self.n_size)
            
            # Analyze constraints
            queens = (board == VAL_QUEEN)
            n_queens = queens.sum().item()
            
            coords = torch.nonzero(queens)
            rows = coords[:, 0]
            cols = coords[:, 1]
            
            is_valid = True
            
            # Check 1: Count
            if n_queens != self.n_size:
                self.count_errors += 1
                is_valid = False
                
            # Check 2: Rows/Cols
            if len(torch.unique(rows)) != n_queens:
                self.row_errors += 1
                is_valid = False
            if len(torch.unique(cols)) != n_queens:
                self.col_errors += 1
                is_valid = False
                
            # Check 3: Diagonals
            if len(torch.unique(rows + cols)) != n_queens or len(torch.unique(rows - cols)) != n_queens:
                self.diag_errors += 1
                is_valid = False
                
            if is_valid:
                self.valid += 1
                # Exact match check
                if torch.equal(preds_cpu[i], labels[i].cpu()):
                    self.correct += 1
            
            # Save first 5 failures for visualization
            if not is_valid and len(self.sample_boards) < 5:
                self.sample_boards.append((inputs_cpu[i], preds_cpu[i]))

    def result(self, save_path: Optional[str] = None, rank: int = 0, world_size: int = 1, group=None) -> Optional[dict]:
        # Note: Distributed reduction for scalar metrics omitted for brevity, 
        # but essential for multi-GPU. Assuming single GPU for debug.
        
        metrics = {
            "test/acc": self.correct / self.total,
            "test/valid_rate": self.valid / self.total,
            "test/err_row": self.row_errors / self.total,
            "test/err_col": self.col_errors / self.total,
            "test/err_diag": self.diag_errors / self.total,
            "test/err_count": self.count_errors / self.total,
        }
        
        # Log Images to WandB
        if len(self.sample_boards) > 0 and wandb.run is not None:
            images = []
            for inp, pred in self.sample_boards:
                fig, ax = plt.subplots(1, 2, figsize=(6, 3))
                ax[0].imshow(inp.view(self.n_size, self.n_size))
                ax[0].set_title("Input")
                ax[1].imshow(pred.view(self.n_size, self.n_size))
                ax[1].set_title("Prediction")
                images.append(wandb.Image(fig))
                plt.close(fig)
            
            # Log directly (a bit hacky inside evaluator but works)
            wandb.log({"examples/failures": images}, commit=False)
            
        return metrics