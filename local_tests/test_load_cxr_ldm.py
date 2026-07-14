from pathlib import Path

import torch

from generative.networks.nets import AutoencoderKL, DiffusionModelUNet


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIR = (
    ROOT
    / "model-zoo"
    / "models"
    / "cxr_image_synthesis_latent_diffusion_model"
    / "models"
)

AUTOENCODER_CKPT = CHECKPOINT_DIR / "autoencoder.pth"
DIFFUSION_CKPT = CHECKPOINT_DIR / "diffusion_model.pth"


def load_checkpoint(path: Path):
    """Load a state_dict checkpoint safely onto CPU."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")

    # Defensive handling in case a future checkpoint is wrapped.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected checkpoint to contain a state_dict, "
            f"but got {type(checkpoint)}"
        )

    return checkpoint


def build_autoencoder() -> AutoencoderKL:
    """Exact architecture used by the official CXR checkpoint."""
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
    )


def build_diffusion_model() -> DiffusionModelUNet:
    """Exact architecture used by the official CXR checkpoint."""
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


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main():
    print("=" * 80)
    print("MONAI GenerativeModels checkpoint loading test")
    print("=" * 80)

    print(f"\nAutoencoder checkpoint:\n  {AUTOENCODER_CKPT}")
    print(f"Diffusion checkpoint:\n  {DIFFUSION_CKPT}")

    # ------------------------------------------------------------------
    # Autoencoder
    # ------------------------------------------------------------------
    print("\n[1/2] Building AutoencoderKL...")
    autoencoder = build_autoencoder()

    print("Loading AutoencoderKL checkpoint with strict=True...")
    autoencoder_state = load_checkpoint(AUTOENCODER_CKPT)

    result = autoencoder.load_state_dict(
        autoencoder_state,
        strict=True,
    )

    print("Autoencoder loaded successfully.")
    print(f"Load result: {result}")
    print(
        f"Parameters: "
        f"{count_parameters(autoencoder) / 1_000_000:.2f} M"
    )

    # ------------------------------------------------------------------
    # Diffusion U-Net
    # ------------------------------------------------------------------
    print("\n[2/2] Building DiffusionModelUNet...")
    diffusion_model = build_diffusion_model()

    print("Loading DiffusionModelUNet checkpoint with strict=True...")
    diffusion_state = load_checkpoint(DIFFUSION_CKPT)

    result = diffusion_model.load_state_dict(
        diffusion_state,
        strict=True,
    )

    print("Diffusion model loaded successfully.")
    print(f"Load result: {result}")
    print(
        f"Parameters: "
        f"{count_parameters(diffusion_model) / 1_000_000:.2f} M"
    )

    print("\n" + "=" * 80)
    print("SUCCESS: both official checkpoints loaded with strict=True.")
    print("=" * 80)


if __name__ == "__main__":
    main()
