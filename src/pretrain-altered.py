from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy
import warnings  # added

# Suppress verbose pydantic UnsupportedFieldAttributeWarning (repr/frozen misuse in external configs)
try:
    from pydantic._internal._generate_schema import UnsupportedFieldAttributeWarning  # type: ignore

    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except Exception:
    warnings.filterwarnings("ignore", message="UnsupportedFieldAttributeWarning")

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper

from torch.optim import Adam as adam
from device_helper import get_device

device = get_device()


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0  # when to start eval
    eval_save_outputs: List[str] = []

    ema: bool = False  # use Exponential-Moving-Average
    ema_rate: float = 0.999  # EMA-rate
    freeze_weights: bool = (
        False  # If True, freeze weights and only learn the embeddings
    )


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(
    config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs
):
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=config.data_paths_test
            if len(config.data_paths_test) > 0 and split == "test"
            else config.data_paths,
            rank=rank,
            num_replicas=world_size,
            **kwargs,
        ),
        split=split,
    )
    # CPU-friendly defaults; override with env if needed
    num_workers_env = os.environ.get("DL_NUM_WORKERS")
    num_workers = (
        int(num_workers_env)
        if num_workers_env is not None
        else (2 if device.type == "cuda" else 0)
    )
    loader_kwargs = dict(
        batch_size=None,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        prefetch = int(os.environ.get("DL_PREFETCH_FACTOR", "2"))
        loader_kwargs["prefetch_factor"] = prefetch

    dataloader = DataLoader(dataset, **loader_kwargs)

    # If the requested split produced zero groups, fall back to "all"
    try:
        empty_split = dataset.metadata.total_groups == 0
    except Exception:
        empty_split = False

    if empty_split:
        if rank == 0:
            print(
                f"WARNING: Split '{split}' yielded zero groups. Falling back to split='all'."
            )
        dataset = PuzzleDataset(
            PuzzleDatasetConfig(
                seed=config.seed,
                dataset_paths=config.data_paths_test
                if len(config.data_paths_test) > 0 and split == "test"
                else config.data_paths,
                rank=rank,
                num_replicas=world_size,
                **kwargs,
            ),
            split="all",
        )
        dataloader = DataLoader(dataset, **loader_kwargs)

    if rank == 0:
        msg = f"DataLoader[{split}]: num_workers={num_workers}, persistent_workers={num_workers > 0}"
        if num_workers > 0:
            msg += f", prefetch_factor={loader_kwargs.get('prefetch_factor')}"
        sets = getattr(dataset.metadata, "sets", [])
        total_groups = getattr(dataset.metadata, "total_groups", None)
        print(msg)
        print(f"Dataset[{split}] sets={list(sets)} total_groups={total_groups}")

    return dataloader, dataset.metadata


def create_model(
    config: PretrainConfig,
    train_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,
):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False,  # Non-autoregressive
    )

    # Load model class
    model_cls = load_model_class(config.arch.name)

    # Create a ACTLossHead instance based on trm.yaml to compute the losses
    loss_head_cls = load_model_class(config.arch.loss.name)

    # Create the actual model
    model: nn.Module = model_cls(model_cfg).to(device)
    # print(model)

    model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore

    model.to(device)

    # Only compile on CUDA; fallback to eager if it fails
    can_compile = ("DISABLE_COMPILE" not in os.environ) and (device.type == "cuda")
    if can_compile:
        try:
            model = torch.compile(model)  # type: ignore
            if rank == 0:
                print(f"torch.compile enabled ({device.type}).")
        except Exception as e:
            if rank == 0:
                print(
                    f"torch.compile failed on {device.type}: {e}. Falling back to eager mode."
                )
    else:
        if rank == 0:
            reason = (
                "env flag DISABLE_COMPILE is set"
                if "DISABLE_COMPILE" in os.environ
                else f"device is {device.type}, skipping compile"
            )
            print(f"Skipping torch.compile: {reason}.")

    # Load checkpoint (only if provided)
    if rank == 0:
        load_checkpoint(model, config)

    # Broadcast parameters from rank 0
    if world_size > 1:
        with torch.no_grad():
            for param in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(param, src=0)

    # Optimizers and lr
    # Helper to fetch exactly 3 tensors for sparse embedding optimizer
    def _get_sparse_embedding_params(puzzle_emb):
        # Prefer explicit method if provided by module
        if hasattr(puzzle_emb, "get_optimizer_buffers"):
            params = list(puzzle_emb.get_optimizer_buffers())
            if len(params) == 3:
                return params
        # Collect named buffers
        named_bufs = list(puzzle_emb.named_buffers())
        if len(named_bufs) == 3:
            return [b for _, b in named_bufs]
        # Try heuristic ordering by common names
        priority_order = [
            "weights",
            "weight",
            "embeddings",
            "exp_avg",
            "m",
            "ema",
            "avg",
            "exp_avg_sq",
            "v",
            "var",
        ]
        chosen = []
        seen = set()
        name_map = {n: b for n, b in named_bufs}
        for key in priority_order:
            if key in name_map and id(name_map[key]) not in seen:
                chosen.append(name_map[key])
                seen.add(id(name_map[key]))
            if len(chosen) == 3:
                return chosen
        # Fallback: take first up to 3
        raw_bufs = [b for _, b in named_bufs]
        if len(raw_bufs) >= 3:
            return raw_bufs[:3]
        # If fewer than 3, pad with zeros like shapes of first
        if len(raw_bufs) == 0:
            raise RuntimeError(
                "Sparse embedding has no buffers; cannot build optimizer params."
            )
        first = raw_bufs[0]
        while len(raw_bufs) < 3:
            raw_bufs.append(torch.zeros_like(first))
        if rank == 0:
            print("WARNING: Padded sparse embedding optimizer params to length 3.")
        return raw_bufs[:3]

    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            adam(
                model.parameters(),
                lr=0,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2),
            )
        ]
        optimizer_lrs = [config.lr]
    elif config.freeze_weights:
        sparse_params = _get_sparse_embedding_params(model.model.puzzle_emb)  # type: ignore
        if rank == 0 and len(sparse_params) != 3:
            print(
                f"WARNING: Adjusted sparse embedding param count to 3 (found {len(sparse_params)})."
            )
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                sparse_params,
                lr=0,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            )
        ]
        optimizer_lrs = [config.puzzle_emb_lr]
    else:
        sparse_params = _get_sparse_embedding_params(model.model.puzzle_emb)  # type: ignore
        if rank == 0 and len(sparse_params) != 3:
            print(
                f"WARNING: Adjusted sparse embedding param count to 3 (found {len(sparse_params)})."
            )
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                sparse_params,
                lr=0,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            ),
            adam(
                model.parameters(),
                lr=0,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2),
            ),
        ]
        optimizer_lrs = [config.puzzle_emb_lr, config.lr]

    return model, optimizers, optimizer_lrs


def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0] * sd[0][k].to(device)
        for i in range(1, len(nets)):
            comb_net += alpha[i] * sd[i][k].to(device)
        sd_alpha[k] = comb_net
    net.load_state_dict(sd_alpha)
    return net


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0,
    num_cycles: float = 0.5,
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return base_lr * (
        min_ratio
        + max(
            0.0,
            (1 - min_ratio)
            * 0.5
            * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )
    )


def init_train_state(
    config: PretrainConfig,
    train_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,
):
    # Estimated total training steps
    total_steps = int(
        config.epochs
        * train_metadata.total_groups
        * train_metadata.mean_puzzle_examples
        / config.global_batch_size
    )

    # Model
    model, optimizers, optimizer_lrs = create_model(
        config, train_metadata, rank=rank, world_size=world_size
    )

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(
        train_state.model.state_dict(),
        os.path.join(config.checkpoint_path, f"step_{train_state.step}"),
    )


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        # Load state dict
        state_dict = torch.load(config.load_checkpoint, map_location=device)

        # Resize and reset puzzle emb if needed
        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(
                    f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}"
                )
                # Re-initialize using mean
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True)
                    .expand(expected_shape)
                    .contiguous()
                )
        model.load_state_dict(state_dict, assign=True)


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio,
    )


def create_evaluators(
    config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata
) -> List[Any]:
    data_paths = (
        config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    )
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path,
                eval_metadata=eval_metadata,
                **cfg.__pydantic_extra__,
            )  # type: ignore
            evaluators.append(cls)

    return evaluators


def train_batch(
    config: PretrainConfig,
    train_state: TrainState,
    batch: Any,
    global_batch_size: int,
    rank: int,
    world_size: int,
):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.to(device) for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device(device):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(
        carry=train_state.carry, batch=batch, return_keys=[]
    )

    ((1 / global_batch_size) * loss).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    # Apply optimizer
    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group["lr"] = lr_this_step

        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        # Sort keys to guarantee all processes use the same order.
        metric_keys = list(sorted(metrics.keys()))
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {
                f"train/{k}": v / (global_batch_size if k.endswith("loss") else count)
                for k, v in reduced_metrics.items()
            }

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0

        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")

            # To device
            batch = {k: v.to(device) for k, v in batch.items()}
            carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        # Move to CPU for saving GPU memory
                        save_preds[k].append(v.cpu())

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())),
                    dtype=torch.float32,
                    device=device,
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds,
                os.path.join(
                    config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"
                ),
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")

        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(
                    f"Running evaluator {i + 1}/{len(evaluators)}: {evaluator.__class__.__name__}"
                )

            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(
                evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group
            )
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")

        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics


def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name),
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(
    hydra_config: DictConfig, rank: int, world_size: int
) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = (
                f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
            )
        if config.run_name is None:
            config.run_name = (
                f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
            )
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join(
                "checkpoints", config.project_name, config.run_name
            )

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    """
    Launch the training process.
    """
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])

        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        elif device.type == "mps":
            backend = "gloo"  # MPS doesn't support NCCL
        else:
            backend = "gloo"  # CPU

        dist.init_process_group(backend=backend)
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        # Optional CPU process group if you really need it
        if backend == "gloo":
            CPU_PROCESS_GROUP = dist.new_group(backend="gloo")

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = (
        config.eval_interval if config.eval_interval is not None else config.epochs
    )
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, (
        "Eval interval must be a divisor of total epochs."
    )

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    # Debug: dataset / expected steps summary
    if RANK == 0:
        expected_examples = (
            train_metadata.total_groups * train_metadata.mean_puzzle_examples
        )
        expected_steps_total = int(
            config.epochs * expected_examples / config.global_batch_size
        )
        expected_steps_per_iter = int(
            train_epochs_per_iter * expected_examples / config.global_batch_size
        )
        print(
            f"Train metadata: total_groups={train_metadata.total_groups}, mean_examples_per_group={train_metadata.mean_puzzle_examples:.2f}"
        )
        print(
            f"Expected steps total={expected_steps_total}, per_iter={expected_steps_per_iter}, global_batch_size={config.global_batch_size}"
        )
    try:
        # Eval dataloader
        # This is the test set
        # The test set is used for evaluation during training
        eval_loader, eval_metadata = create_dataloader(
            config,
            "test",
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
            rank=RANK,
            world_size=WORLD_SIZE,
        )
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(
        config, train_metadata, rank=RANK, world_size=WORLD_SIZE
    )

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump(),
            settings=wandb.Settings(_disable_stats=True),
        )  # type: ignore
        wandb.log(
            {"num_params": sum(x.numel() for x in train_state.model.parameters())},
            step=0,
        )
        save_code_and_config(config)
    if config.ema:
        print("Setup EMA")
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Training Loop
    for _iter_id in range(total_iters):
        # The rank is a unique identifier for each process (distributed training on multiple GPUs)
        # The world size is the total number of processes
        print(
            f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}"
        )

        # Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        first_batch_seen = False
        for set_name, batch, global_batch_size in train_loader:
            if not first_batch_seen and RANK == 0:
                print("First batch received.")
                first_batch_seen = True
            metrics = train_batch(
                config,
                train_state,
                batch,
                global_batch_size,
                rank=RANK,
                world_size=WORLD_SIZE,
            )

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
            if config.ema:
                ema_helper.update(train_state.model)

        if not first_batch_seen and RANK == 0:
            print(
                "WARNING: No training batches yielded this iteration. Check dataset path/split or global_batch_size."
            )

        # If enough steps have been taken, run evaluation.
        # The evaluation evaluates the model on a validation set and saves checkpoints.
        if _iter_id >= config.min_eval_interval:
            # Evaluation
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            metrics = evaluate(
                config,
                train_state_eval,
                eval_loader,
                eval_metadata,
                evaluators,
                rank=RANK,
                world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP,
            )

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)

            # Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (
                config.checkpoint_every_eval or (_iter_id == total_iters - 1)
            ):
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
