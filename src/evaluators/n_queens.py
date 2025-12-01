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
        # Create tensors for reduction
        metrics_tensor = torch.tensor([
            self.correct,
            self.valid,
            self.row_errors,
            self.col_errors,
            self.diag_errors,
            self.count_errors,
            self.total
        ], dtype=torch.float64, device="cuda")

        # Distributed reduction
        if world_size > 1:
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

        # Only Rank 0 computes and returns the dictionary
        if rank == 0:
            total = metrics_tensor[6].item()
            if total == 0: total = 1 # Avoid div by zero

            metrics = {
                "test/acc": metrics_tensor[0].item() / total,
                "test/valid_rate": metrics_tensor[1].item() / total,
                "test/err_row": metrics_tensor[2].item() / total,
                "test/err_col": metrics_tensor[3].item() / total,
                "test/err_diag": metrics_tensor[4].item() / total,
                "test/err_count": metrics_tensor[5].item() / total,
            }
            
            # --- IMAGE LOGGING FIX ---
            # We log images directly here, separate from the return dict
            # This bypasses the scalar reduction logic in pretrain.py
            if len(self.sample_boards) > 0 and wandb.run is not None:
                images = []
                for inp, pred in self.sample_boards:
                    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
                    
                    # Input (1=Empty, 2=Queen)
                    ax[0].imshow(inp.view(self.n_size, self.n_size))
                    ax[0].set_title("Input Hints")
                    ax[0].axis('off')
                    
                    # Pred
                    ax[1].imshow(pred.view(self.n_size, self.n_size))
                    ax[1].set_title("Model Guess")
                    ax[1].axis('off')
                    
                    images.append(wandb.Image(fig))
                    plt.close(fig)
                
                # Use commit=False so it attaches to the current step
                wandb.log({"examples/failures": images}, commit=False)

            return metrics
            
        return None