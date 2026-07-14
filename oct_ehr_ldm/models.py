from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler

from .data import safe_torch_load


def build_cxr_autoencoder(use_checkpointing: bool = False) -> AutoencoderKL:
    return AutoencoderKL(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        latent_channels=3,
        num_channels=(64, 128, 128, 128),
        num_res_blocks=2,
        norm_num_groups=32,
        norm_eps=1e-6,
        attention_levels=(False, False, False, False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        use_checkpointing=use_checkpointing,
    )


def build_cxr_diffusion() -> DiffusionModelUNet:
    return DiffusionModelUNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        num_channels=(256, 512, 768),
        num_res_blocks=2,
        attention_levels=(False, True, True),
        norm_num_groups=32,
        norm_eps=1e-6,
        resblock_updown=False,
        num_head_channels=(0, 512, 768),
        with_conditioning=True,
        transformer_num_layers=1,
        cross_attention_dim=1024,
    )


def build_cxr_scheduler() -> DDIMScheduler:
    return DDIMScheduler(
        beta_start=0.0015,
        beta_end=0.0205,
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        prediction_type="v_prediction",
        clip_sample=False,
    )


def extract_state_dict(payload: Any, preferred_keys: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload must be a dictionary, got {type(payload)!r}")
    for key in preferred_keys + ("state_dict", "model_state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, dict) and candidate and all(torch.is_tensor(value) for value in candidate.values()):
            return candidate
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise KeyError(f"Could not find a tensor state_dict; checked keys {preferred_keys + ('state_dict', 'model_state_dict')}")


def load_module_checkpoint(
    module: nn.Module,
    path: str | Path,
    preferred_keys: tuple[str, ...] = (),
    strict: bool = True,
) -> Any:
    payload = safe_torch_load(path)
    state = extract_state_dict(payload, preferred_keys)
    return module.load_state_dict(state, strict=strict)


class EHRConditionProjector(nn.Module):
    """Map one patient EHR latent and an optional OCT view to cross-attention tokens."""

    def __init__(
        self,
        ehr_dim: int,
        cross_attention_dim: int = 1024,
        num_tokens: int = 8,
        hidden_dim: int | None = None,
        num_views: int = 1,
        view_embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        self.ehr_dim = int(ehr_dim)
        self.cross_attention_dim = int(cross_attention_dim)
        self.num_tokens = int(num_tokens)
        self.num_views = int(num_views)
        self.view_embedding = (
            nn.Embedding(self.num_views, view_embedding_dim) if self.num_views > 1 and view_embedding_dim > 0 else None
        )
        input_dim = self.ehr_dim + (view_embedding_dim if self.view_embedding is not None else 0)
        hidden = int(hidden_dim or min(max(input_dim * 2, 1024), 4096))
        self.projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.num_tokens * self.cross_attention_dim),
        )
        self.null_context = nn.Parameter(torch.zeros(1, self.num_tokens, self.cross_attention_dim))
        nn.init.normal_(self.null_context, std=0.02)
        self.register_buffer("ehr_mean", torch.zeros(self.ehr_dim), persistent=True)
        self.register_buffer("ehr_std", torch.ones(self.ehr_dim), persistent=True)

    @torch.no_grad()
    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (self.ehr_dim,) or std.shape != (self.ehr_dim,):
            raise ValueError(f"Expected EHR statistics with shape {(self.ehr_dim,)}, got {mean.shape} and {std.shape}")
        self.ehr_mean.copy_(mean.float())
        self.ehr_std.copy_(std.float().clamp_min(1e-6))

    def conditional_context(self, ehr: torch.Tensor, view_ids: torch.Tensor | None = None) -> torch.Tensor:
        if ehr.ndim != 2 or ehr.shape[1] != self.ehr_dim:
            raise ValueError(f"Expected EHR input [B,{self.ehr_dim}], got {tuple(ehr.shape)}")
        features = (ehr.float() - self.ehr_mean) / self.ehr_std
        if self.view_embedding is not None:
            if view_ids is None:
                view_ids = torch.zeros(ehr.shape[0], dtype=torch.long, device=ehr.device)
            features = torch.cat([features, self.view_embedding(view_ids.long())], dim=-1)
        context = self.projector(features)
        return context.view(ehr.shape[0], self.num_tokens, self.cross_attention_dim)

    def unconditional_context(self, batch_size: int) -> torch.Tensor:
        return self.null_context.expand(batch_size, -1, -1)

    def forward(
        self,
        ehr: torch.Tensor,
        view_ids: torch.Tensor | None = None,
        condition_dropout: float = 0.0,
    ) -> torch.Tensor:
        conditional = self.conditional_context(ehr, view_ids)
        if not self.training or condition_dropout <= 0:
            return conditional
        drop = torch.rand(ehr.shape[0], 1, 1, device=ehr.device) < condition_dropout
        return torch.where(drop, self.unconditional_context(ehr.shape[0]), conditional)


def configure_diffusion_phase(model: nn.Module, phase: str) -> list[str]:
    if phase not in {"alignment", "full"}:
        raise ValueError(f"Unknown diffusion phase: {phase}")
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = phase == "full" or ".attn2." in name
        if parameter.requires_grad:
            trainable_names.append(name)
    if phase == "alignment" and not trainable_names:
        raise RuntimeError("No cross-attention parameters matched '.attn2.' in DiffusionModelUNet")
    return trainable_names


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad or not trainable_only)
