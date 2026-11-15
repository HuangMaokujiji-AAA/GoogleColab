"""Configuration, reproducibility and local device selection."""
import json
import random
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_config(path=None):
    with (PROJECT_DIR / "configs/local.json").open() as f:
        cfg = json.load(f)
    if path and Path(path).resolve() != PROJECT_DIR / "configs/local.json":
        with Path(path).expanduser().open() as f:
            overrides = json.load(f)
        unknown = overrides.keys() - cfg.keys()
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        cfg.update(overrides)
    return cfg


def validate_config(cfg):
    positive = ("n_top_genes", "d_cell", "d_niche", "n_prototypes",
                "max_cells_per_tile", "max_context_cells", "batch_size",
                "inference_batch_size", "niche_context_cells", "niche_query_cells",
                "max_vae_steps_per_epoch", "max_fm_steps_per_epoch",
                "max_niche_steps_per_epoch", "vae_epochs", "niche_epochs",
                "fm_epochs", "n_ode_steps", "lr", "rare_gamma", "rare_w_max",
                "num_threads")
    for key in positive:
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 < cfg["rare_freq_thresh"] < 1:
        raise ValueError("rare_freq_thresh must be between 0 and 1")
    if cfg["max_cells_per_tile"] < 6:
        raise ValueError("max_cells_per_tile must be at least 6 for flow training")
    for key in ("max_sections", "max_cells_per_section"):
        if cfg[key] is not None and cfg[key] < (2 if key == "max_sections" else 12):
            raise ValueError(f"{key} is too small")
    if cfg["tile_halo"] < 0:
        raise ValueError("tile_halo must be non-negative")
    if cfg["expression_mode"] not in ("auto", "counts", "normalized"):
        raise ValueError("expression_mode must be auto, counts or normalized")
    return cfg


def select_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu or auto")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use --device cpu or auto")
    return torch.device(name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def smoke_config(cfg):
    """A real-data wiring check, not a trained scientific result."""
    cfg.update(max_sections=3, max_cells_per_section=256, n_top_genes=64,
               n_prototypes=4, d_cell=8, d_niche=8, vae_epochs=1,
               niche_epochs=1, fm_epochs=1, max_vae_steps_per_epoch=2,
               max_niche_steps_per_epoch=2, max_fm_steps_per_epoch=2,
               batch_size=128, max_cells_per_tile=128, max_context_cells=256,
               niche_context_cells=256, niche_query_cells=128,
               n_ode_steps=2)
    return cfg
