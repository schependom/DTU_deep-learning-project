"""
Tiny Recursive Reasoning Model with ACT (Adaptive Computation Time) and two levels of recursion (z_L, z_H).

    -   z_H in the HRM is simply the current (embedded) solution (y in the TRM)
        The embedding is reversed by applying the output head and rounding to the nearest token using the argmax operation

    -   z_L in the HRM (z in the TRM) is a latent feature that does not directly correspond to a solution, 
        but it can be transformed into a solution by applying z_H <- f_H(x, z_L, z_H), or just y <- f(x, y, z)
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int  # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False  # use mlp on L instead of transformer
    puzzle_emb_len: int = 16  # if non-zero, its specified to this value
    # No continue ACT loss, only use the sigmoid of the halt which makes much more sense
    no_ACT_continue: bool = True


# ---------------------------------------------------------------------------- #
#          1 'SEQUENCE' OR 'BLOCK' OF SA/MLP->ADD&NORM->MLP->ADD&NORM          #
# ---------------------------------------------------------------------------- #


class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config

        # Use MLP on z_L instead of self-attention
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -
                                    self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len,  # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(
                hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(
                cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(
            hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):

    # Layers is 4x dense layer (transformer)
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    # ---------------------------------------------------------------------------- #
    #                               LATENT RECURSION                               #
    # ---------------------------------------------------------------------------- #

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:

        # z_L = z_L + (z_H + x)
        # z = z + (y + x)
        hidden_states = hidden_states + input_injection

        # Each layer is a TinyRecursiveReasoningModel_ACTV1Block
        # -> this corresponds to the 4x transformer on page 1
        for layer in self.layers:
            # Call the network with the new hidden_states calculated above as z+(y+z)
            # "z = net(x,y,z)"
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Embed the input x to hidden size
        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)

        # Output heads
        self.lm_head = CastedLinear(
            self.config.hidden_size, self.config.vocab_size, bias=False)

        # Q-head for halting
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        # puzzle_emb_ndim is the total embedding dimension for the puzzle identifiers
        # puzzle_emb_len is the length in number of tokens (each of hidden_size dimension)
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -
                                self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div

        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len,
                                             self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers (latent recursion)
        #   self.L_layers is 4 in the figure on page 1 (4 x dense layers)
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
            layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states z_H, z_L
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(
            self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(
            self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * \
                self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * \
                (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len,
                            self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len,
                            self.config.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    # ---------------------------------------------------------------------------- #
    #                               DEEP SUPERVISION                               #
    # ---------------------------------------------------------------------------- #

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(
            batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L

        with torch.no_grad():

            # H_cycles-1 = N_sup-1 without grad
            for _H_step in range(self.config.H_cycles-1):

                # n-1 steps
                # update latent z given x+y+z
                for _L_step in range(self.config.L_cycles):
                    # z <- f(x + y + z)
                    # z_H <- f_L(x + z_H + z_L)
                    # latent_recursion
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                    # L_level is the TinyRecursiveReasoningModel_ACTV1ReasoningModule
                    # The forward() is called here
                    # forward(z_L, input_injection = z_H+x, ...)

                # update solution y given x+z
                # y <- f(y + z)
                # z_H <- f_H(z_L + z_H)
                z_H = self.L_level(z_H, z_L, **seq_info)  # latent_recursion

        # 1 with gradients:
        #   z_L = f_L(z_L + z_H + input_embeddings)
        #   z_H = f_H(z_H, z_L)
        for _L_step in range(self.config.L_cycles):

            # z <- f(x + y + z)
            # z_H <- f_L(z_H + z_L + x)
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

        # y <- f(y + z)
        # z_H <- f_H(z_L + z_H)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # LM Outputs
        # (y,z) = (z_H, z_L) is the carry that is input to each new deep supervision step
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
            # detach the carry for the new supervision step
            z_H=z_H.detach(), z_L=z_L.detach()
        )

        # Use Language Model head to generate the y_hat and calculate loss
        # y_hat = ...
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q-head to generate q, the chance of halting
        # uses the first puzzle_emb position
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        # (x,y), y_hat, q
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_ACTV1Carry(
            # Empty is expected, it will be reseted in first pass as all sequences are halted.
            inner_carry=self.inner.empty_carry(batch_size),

            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            # Default to halted
            halted=torch.ones((batch_size, ), dtype=torch.bool),

            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(
            carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view(
            (-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes

                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * \
                    torch.randint_like(
                        new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(
                        new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(
                        is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
