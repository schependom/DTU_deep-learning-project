from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy

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
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=(device.type == "cuda"),  # avoid MPS warning
        persistent_workers=True,
    )
    return dataloader, dataset.metadata


def create_model(
    config: PretrainConfig,
    train_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,  # world size is the number of processes/nodes used in distributed training
    # in CPU training, world_size=1, because only one process is used
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
    if rank == 0:
        print("--- Model Architecture ---")
        print(model)
        print("--------------------------")

    model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore

    model.to(device)

    # Only compile on CUDA; fallback to eager if it fails
    if ("DISABLE_COMPILE" not in os.environ) and (device.type != "mps"):
        try:
            model = torch.compile(model)  # type: ignore
            if rank == 0:
                print("torch.compile enabled (CUDA).")
        except Exception as e:
            if rank == 0:
                print(f"torch.compile failed: {e}. Falling back to eager mode.")
    else:
        if rank == 0:
            reason = (
                "env flag DISABLE_COMPILE is set"
                if "DISABLE_COMPILE" in os.environ
                else f"device is {device.type}, skipping compile"
            )
            print(f"Skipping torch.compile: {reason}.")

    # Load checkpoint
    if rank == 0:
        load_checkpoint(model, config)

    # Broadcast parameters from rank 0
    if world_size > 1:
        if rank == 0:
            print(f"Broadcasting model parameters to {world_size} processes.")
        with torch.no_grad():
            for param in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(param, src=0)

    # Optimizers and lr
    if config.arch.puzzle_emb_ndim == 0:
        if rank == 0:
            print("Using Adam optimizer for all parameters.")
        optimizers = [
            adam(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2),
            )
        ]
        optimizer_lrs = [config.lr]
    elif config.freeze_weights:
        if rank == 0:
            print(
                "Using CastedSparseEmbeddingSignSGD_Distributed for puzzle_emb (weights frozen)."
            )
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            )
        ]
        optimizer_lrs = [config.puzzle_emb_lr]
    else:
        if rank == 0:
            print(
                "Using CastedSparseEmbeddingSignSGD_Distributed for puzzle_emb and Adam for rest."
            )

        # --- NEW FIX: Separate params/buffers for sparse optimizer ---

        # 1. Get the 1 parameter AND 2 buffers for the sparse optimizer (total=3)
        sparse_tensors = list(model.model.puzzle_emb.parameters()) + list(
            model.model.puzzle_emb.buffers()
        )

        # 2. Get the *names* of the sparse parameters to exclude them from Adam
        sparse_param_names = {n for n, p in model.model.puzzle_emb.named_parameters()}

        # 3. Get all *other* parameters for Adam
        adam_params = [
            p for n, p in model.named_parameters() if n not in sparse_param_names
        ]

        if rank == 0:
            print(f"  > Sparse optimizer tensors: {len(sparse_tensors)} (Expected 3)")
            print(f"  > Adam optimizer parameters: {len(adam_params)}")

        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                sparse_tensors,  # <-- Pass all 3 tensors here
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            ),
            adam(
                adam_params,  # <-- Pass only the *other* params here
                lr=0,  # Needs to be set by scheduler
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
    save_path = os.path.join(config.checkpoint_path, f"step_{train_state.step}.pth")
    print(f"Saving checkpoint to {save_path}")
    torch.save(
        train_state.model.state_dict(),
        save_path,
    )


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

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
                print(f"  Evaluating batch {processed_batches}: {set_name}")

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
                print(f"    Completed inference in {inference_steps} steps")

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
        if len(evaluators) > 0:
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
                    evaluator_save_path,
                    rank=rank,
                    world_size=world_size,
                    group=cpu_group,
                )
                if rank == 0 and metrics is not None:
                    if reduced_metrics is None:
                        reduced_metrics = {}

                    reduced_metrics.update(metrics)
                    print(f"  Completed {evaluator.__class__.__name__}")

            if rank == 0:
                print("All evaluators completed!")
        else:
            if rank == 0:
                print("No evaluators to run.")

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
        print(f"Rank {rank}: Broadcasting config from Rank 0.")
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
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
            backend = "gloo"  # MPS doesn’t support NCCL
        else:
            backend = "gloo"  # CPU

        print(f"Initializing distributed process group with backend '{backend}'...")
        dist.init_process_group(backend=backend)
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        # Optional CPU process group if you really need it
        if backend == "gloo":
            CPU_PROCESS_GROUP = dist.new_group(backend="gloo")

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # --- ADDED: Start of new logging ---
    if RANK == 0:
        print("\n" + "=" * 50)
        print("--- 🚀 PRETRAIN SCRIPT STARTED 🚀 ---")
        print(f"Run Name:         {config.run_name}")
        print(f"Device:           {device}")
        print(
            f"Distributed Mode: {WORLD_SIZE > 1} (Rank: {RANK}, World Size: {WORLD_SIZE})"
        )
        print(f"Checkpoint Path:  {config.checkpoint_path}")
        print(f"Data Path:        {config.data_paths}")
        print(f"Epochs:           {config.epochs} (Eval every {config.eval_interval})")
        print("=" * 50 + "\n")
    # --- ADDED: End of new logging ---

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

    if RANK == 0:
        print("--- 💾 Loading Data... ---")
    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    try:
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
        if RANK == 0:
            print("WARNING: No evaluation data found.")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        if RANK == 0:
            print("No evaluators found.")
        evaluators = []

    # --- ADDED: Logging for dataset ---
    if RANK == 0:
        print("--- Data Loaded ---")
        print(
            f"Train Metadata: {train_metadata.vocab_size} vocab, {train_metadata.seq_len} seq len, {train_metadata.total_groups} groups"
        )
        if eval_metadata:
            print(
                f"Eval Metadata: {eval_metadata.vocab_size} vocab, {eval_metadata.seq_len} seq len, {eval_metadata.total_groups} groups"
            )
        print(
            f"Total Iterations: {total_iters} ({train_epochs_per_iter} epochs per iter)"
        )
        print("-" * 50 + "\n")
    # --- ADDED: End of new logging ---

    # Train state
    if RANK == 0:
        print("--- 🧠 Initializing Model & Train State... ---")
    train_state = init_train_state(
        config, train_metadata, rank=RANK, world_size=WORLD_SIZE
    )

    # --- ADDED: Logging for model ---
    if RANK == 0:
        print(f"--- Model Initialized ---")
        print(f"Total training steps estimated: {train_state.total_steps}")
        print(f"Using EMA (Exponential Moving Average): {config.ema}")
        print(
            f"Total Parameters: {sum(x.numel() for x in train_state.model.parameters())}"
        )
        print("-" * 50 + "\n")
    # --- ADDED: End of new logging ---

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps, desc="Training")
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
        if RANK == 0:
            print("Setting up EMA Helper...")
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Training Loop
    if RANK == 0:
        print("--- 🔥 Starting Training Loop ---")

    # 1 iteration = train_epochs_per_iter epochs
    for _iter_id in range(total_iters):
        current_epoch = _iter_id * train_epochs_per_iter
        if RANK == 0:
            print(
                f"\n--- Iteration {_iter_id + 1}/{total_iters} (Starting Epoch {current_epoch}) ---"
            )

        # Train Iter
        if RANK == 0:
            print("Mode: 🏋️ TRAINING")
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
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

                # --- MODIFIED: Add metrics to progress bar ---
                postfix_metrics = {
                    k.replace("train/", ""): f"{v:.4f}" for k, v in metrics.items()
                }
                progress_bar.set_postfix(postfix_metrics)
                # --- END MODIFICATION ---

            if config.ema:
                ema_helper.update(train_state.model)

        if _iter_id >= config.min_eval_interval and eval_loader is not None:
            # Evaluation
            if RANK == 0:
                print(
                    f"\nMode: 📊 EVALUATING (After Epoch {current_epoch + train_epochs_per_iter - 1})"
                )

            if config.ema:
                if RANK == 0:
                    print("Switching to EMA model for evaluation...")
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
                print(f"Evaluation Metrics: {metrics}")
                wandb.log(metrics, step=train_state.step)

            # Checkpointing
            if RANK == 0:
                print("Mode: 💾 CHECKPOINTING")
            if RANK == 0 and (
                config.checkpoint_every_eval or (_iter_id == total_iters - 1)
            ):
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

        elif eval_loader is None and RANK == 0:
            print("\nSkipping evaluation as no eval_loader is available.")

    # finalize
    if RANK == 0:
        print("\n--- ✅ Training Complete ---")
        if progress_bar:
            progress_bar.close()
    if dist.is_initialized():
        dist.destroy_process_group()
    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    launch()
