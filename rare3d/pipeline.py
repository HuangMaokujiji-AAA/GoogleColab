"""Explicit orchestration of loading, training, validation and export."""
import json
import os
from functools import partial
from pathlib import Path

import torch

from .config import set_seed, select_device, validate_config
from .checkpoints import cpu_state_dict, save_checkpoint
from .data import find_h5ad_files, select_files, load_zhuang_sections, prepare_geometry
from .training import train_vae, discover_prototypes, build_train_triplets, train_niche, train_flow
from .reconstruction import reconstruct_between, export_volume


def run(cfg):
    validate_config(cfg)
    set_seed(cfg["seed"])
    torch.set_num_threads(cfg["num_threads"])
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(cfg["num_threads"]))
    device = select_device(cfg["device"])
    out = Path(cfg["output_dir"]).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = {**cfg, "output_dir": str(out), "data_dir": str(Path(cfg["data_dir"]).expanduser().resolve())}
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    print(f"[Device] {device}\n[Phase 0/4] Loading {cfg['data_dir']}", flush=True)
    files = select_files(find_h5ad_files(cfg["data_dir"]), cfg["max_sections"])
    sections, genes, is_count, depth_axis, plane_axes = load_zhuang_sections(files, cfg)
    center, scale, gene_weights, median_library = prepare_geometry(sections)
    validation = len(sections) // 2 if len(sections) >= 3 else -1
    triplets = build_train_triplets(len(sections), validation)
    vae = train_vae(sections, len(genes), is_count, gene_weights, cfg, device)
    centers, rare, major = discover_prototypes(sections, cfg)
    niche_enc, niche_pred = train_niche(sections, triplets, len(centers), cfg, device)
    flow = train_flow(sections, triplets, centers, rare, major, niche_pred, cfg, device)
    # Persist weights before reconstruction, which is the expensive final stage.
    checkpoint = {"config": cfg, "vae": cpu_state_dict(vae), "niche_encoder": cpu_state_dict(niche_enc),
                  "niche_predictor": cpu_state_dict(niche_pred), "flow": cpu_state_dict(flow),
                  "gene_names": genes.tolist(), "is_count": is_count,
                  "centers": torch.from_numpy(centers), "rare_prototypes": sorted(rare),
                  "major_prototypes": sorted(major), "xy_center": torch.from_numpy(center),
                  "xy_scale": scale, "median_library": median_library,
                  "depth_axis": depth_axis, "plane_axes": plane_axes,
                  "section_files": [s["file"] for s in sections],
                  "validation_index": validation, "gene_weights": torch.from_numpy(gene_weights)}
    save_checkpoint(checkpoint, out / "models.pt")
    summary = {"sections": len(sections), "cells": sum(s["X_csr"].shape[0] for s in sections),
               "genes": len(genes), "device": str(device), "expression_mode": "counts" if is_count else "normalized",
               "depth_axis": sections[0]["coord_names"][depth_axis],
               "section_depths": {s["file"]: s["z_val"] for s in sections},
               "validation_index": validation, "rare_prototypes": sorted(rare),
               "validation_protocol": "transductive: all sections inform VAE/prototypes; validation excluded from niche/flow training"}
    (out / "data_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    reconstruct = partial(reconstruct_between, sections, vae=vae, flow=flow, niche_pred=niche_pred,
                          centers=centers, rare_protos=rare, median_library=median_library, cfg=cfg, device=device)
    print("[Phase 4/4] Validating and exporting reconstruction", flush=True)
    metrics = {"status": "skipped", "reason": "Only two sections; no middle validation slice"}
    if validation >= 0:
        from .evaluation import evaluate_rare_recovery
        a, b = validation - 1, validation + 1
        t = (sections[validation]["z_val"] - sections[a]["z_val"]) / (sections[b]["z_val"] - sections[a]["z_val"])
        s, e, x = reconstruct(a, b, t)
        truth = sections[validation]
        metrics = evaluate_rare_recovery(s, e, x, truth["s"].numpy(), truth["e"].numpy(), truth["X_csr"],
                                        centers, rare, major, gene_weights, true_bio_labels=truth["labels"],
                                        save_fig_path=out / "rare3d_val_comparison.png")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    if cfg["export_reconstruction"]:
        export_volume(sections, genes, center, scale, depth_axis, plane_axes, reconstruct, out)
    print(f"[Done] Results: {out}", flush=True)
    return out
