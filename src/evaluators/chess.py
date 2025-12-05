import torch
import torch.distributed as dist
import wandb
from typing import Optional

def idx_to_square(idx: int) -> str:
    file = idx % 8
    rank = idx // 8
    return "abcdefgh"[file] + str(rank + 1)

def move_id_to_uci(move_id: int) -> str:
    from_idx = move_id // 64
    to_idx = move_id % 64
    return idx_to_square(from_idx) + idx_to_square(to_idx)


class ChessEvaluator:
    required_outputs = ["preds", "logits"]

    def __init__(self, data_path: str, eval_metadata, **kwargs):
        self.data_path = data_path
        self.reset_metrics()

    def reset_metrics(self):
        self.total = 0
        self.correct_top1 = 0
        self.correct_top3 = 0
        self.correct_top5 = 0

        self.sample_errors = []

    def begin_eval(self):
        self.reset_metrics()

    def update_batch(self, batch: dict, preds: dict):
        """
        batch["inputs"]    shape: (B, seq_len)
        batch["labels"]    shape: (B, seq_len)  label[0] is true label
        preds["preds"]     shape: (B, vocab)    logits
        """
        inputs = batch["inputs"]
        labels = batch["labels"]
        logits = preds["preds"]

        label_ids = labels[:, 0].cpu()           # (B,)
        logits_cpu = logits.cpu()                # (B, V)


        if "logits" not in preds:
            print("Warning: 'logits' missing from evaluator input. Check return_keys.")
            return

        full_logits = preds["logits"]

        move_logits = full_logits[:, 0, :]
        
        logits_cpu = move_logits.cpu().float() 

        label_ids = labels[:, 0].cpu()           # (B,)
        inputs_cpu = inputs.cpu()

        for i in range(label_ids.shape[0]):
            self.total += 1
            true_id = label_ids[i].item()

            topk_vals, topk_ids = torch.topk(logits_cpu[i], k=5)
            topk_ids = topk_ids.tolist()

            # top-1
            if topk_ids[0] == true_id:
                self.correct_top1 += 1

            # top-3
            if true_id in topk_ids[:3]:
                self.correct_top3 += 1

            # top-5
            if true_id in topk_ids[:5]:
                self.correct_top5 += 1

            if topk_ids[0] != true_id and len(self.sample_errors) < 5:
                fen_tokens = inputs_cpu[i][1:].tolist()   # skip label token
                self.sample_errors.append({
                    "fen": fen_tokens,
                    "true": true_id,
                    "pred": topk_ids[0],
                    "top5": topk_ids[:5],
                })

    def result(self, save_path: Optional[str] = None,
               rank: int = 0, world_size: int = 1, group=None):

        tensor = torch.tensor([
            self.correct_top1,
            self.correct_top3,
            self.correct_top5,
            self.total
        ], dtype=torch.float64, device="cuda")

        if world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        if rank != 0:
            return None

        # compute metrics
        total = max(tensor[3].item(), 1)

        metrics = {
            "test/top1": tensor[0].item() / total,
            "test/top3": tensor[1].item() / total,
            "test/top5": tensor[2].item() / total,
            "test/total": total,
        }

        # ------- W&B error logging -------
        if wandb.run is not None and len(self.sample_errors) > 0:
            table = wandb.Table(columns=["FEN_tokens", "true_move", "pred_move", "top5"])

            for e in self.sample_errors:
                true_uci = move_id_to_uci(e["true"])
                pred_uci = move_id_to_uci(e["pred"])
                top5_uci = [move_id_to_uci(x) for x in e["top5"]]

                table.add_data(
                    str(e["fen"]),
                    true_uci,
                    pred_uci,
                    str(top5_uci)
                )

            wandb.log({"examples/chess_move_errors": table}, commit=False)

        return metrics
