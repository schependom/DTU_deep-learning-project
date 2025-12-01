"""
Evaluator for Tower of Hanoi datasets.
Save this as: evaluators/hanoi_evaluator.py
"""

import torch
import torch.distributed as dist
import numpy as np
from typing import Optional


class HanoiEvaluator:
    """Evaluates Hanoi prediction accuracy."""

    required_outputs = ["preds"]

    def __init__(self, data_path: str, eval_metadata, **kwargs):
        self.data_path = data_path
        self.eval_metadata = eval_metadata
        self.reset_metrics()

    def reset_metrics(self):
        """Reset accumulated metrics."""
        self.total_token_correct = torch.tensor(0.0, device="cuda")
        self.total_tokens = torch.tensor(0.0, device="cuda")
        self.total_sequences = torch.tensor(0.0, device="cuda")
        self.exact_match_sequences = torch.tensor(0.0, device="cuda")

    def begin_eval(self):
        """Called at the start of evaluation."""
        self.reset_metrics()

    def update_batch(self, batch: dict, preds: dict):
        """Update metrics with a batch of predictions."""
        labels = batch["labels"]  # [B, L]
        pred_tokens = preds["preds"]  # [B, L]

        # Mask for non-padding positions
        pad_id = self.eval_metadata.pad_id
        ignore_id = (
            self.eval_metadata.ignore_label_id
            if self.eval_metadata.ignore_label_id is not None
            else pad_id
        )

        valid_mask = (labels != pad_id) & (labels != ignore_id)

        # Token-level accuracy
        correct_tokens = (pred_tokens == labels) & valid_mask
        self.total_token_correct += correct_tokens.sum()
        self.total_tokens += valid_mask.sum()

        # Sequence-level accuracy (all non-padding tokens must be correct)
        # For each sequence, check if all valid tokens match
        for i in range(labels.shape[0]):
            seq_valid_mask = valid_mask[i]
            if (
                seq_valid_mask.any()
            ):  # Only count sequences with at least one valid token
                seq_correct = (
                    pred_tokens[i][seq_valid_mask] == labels[i][seq_valid_mask]
                ).all()
                self.total_sequences += 1
                if seq_correct:
                    self.exact_match_sequences += 1

    def result(
        self,
        save_path: Optional[str] = None,
        rank: int = 0,
        world_size: int = 1,
        group=None,
    ) -> Optional[dict]:
        """Compute and return final metrics."""
        # Aggregate metrics across all ranks
        if world_size > 1:
            dist.all_reduce(self.total_token_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.total_tokens, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.total_sequences, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.exact_match_sequences, op=dist.ReduceOp.SUM)

        if rank != 0:
            return None

        metrics = {}

        # Get dataset name from path
        dataset_name = (
            self.data_path.split("/")[-1] if "/" in self.data_path else self.data_path
        )

        # Token accuracy
        if self.total_tokens > 0:
            token_accuracy = self.total_token_correct.item() / self.total_tokens.item()
            metrics[f"test/{dataset_name}/token_accuracy"] = token_accuracy
        else:
            metrics[f"test/{dataset_name}/token_accuracy"] = 0.0

        # Sequence accuracy
        if self.total_sequences > 0:
            exact_accuracy = (
                self.exact_match_sequences.item() / self.total_sequences.item()
            )
            metrics[f"test/{dataset_name}/exact_accuracy"] = exact_accuracy
        else:
            metrics[f"test/{dataset_name}/exact_accuracy"] = 0.0

        # Count metrics for debugging
        metrics[f"test/{dataset_name}/total_sequences"] = self.total_sequences.item()
        metrics[f"test/{dataset_name}/total_tokens"] = self.total_tokens.item()

        return metrics
