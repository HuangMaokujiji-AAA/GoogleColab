"""Portable model weights, atomic saving and reconstruction of trained networks."""
from pathlib import Path

import torch

from .config import select_device
from .models import IntrinsicDualVAE, HeterophilyNicheEncoder, NicheIntensityPredictor, RareFlowNet


def cpu_state_dict(model):
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def save_checkpoint(checkpoint, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.pt")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    print(f"[Saved model] {path}", flush=True)
    return path


def load_models(path, device="cpu"):
    """Load the complete models.pt bundle for inference, without training."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    models = {
        "vae": IntrinsicDualVAE(len(checkpoint["gene_names"]), cfg["d_cell"], checkpoint["is_count"]),
        "niche_encoder": HeterophilyNicheEncoder(cfg["d_cell"], cfg["d_niche"]),
        "niche_predictor": NicheIntensityPredictor(cfg["d_cell"], cfg["d_niche"], len(checkpoint["centers"])),
        "flow": RareFlowNet(cfg["d_cell"]),
    }
    target = select_device(device)
    for name, model in models.items():
        model.load_state_dict(checkpoint[name], strict=True)
        model.to(target).eval()
    return models, checkpoint
