"""Read local sections in bounded batches and preserve physical depth."""
import re
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch


def natural_sort_key(path):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", Path(path).name)]


def find_h5ad_files(data_dir):
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if (root / "Zhuang-ABCA-2").is_dir():
        root /= "Zhuang-ABCA-2"
    files = sorted((p for p in root.glob("*.h5ad") if not p.name.startswith("._")),
                   key=natural_sort_key)
    if len(files) < 2:
        raise ValueError(f"Need at least two section .h5ad files in {root}")
    return files


def select_files(files, limit=None):
    if limit and limit < len(files):
        return [files[i] for i in np.linspace(0, len(files) - 1, limit, dtype=int)]
    return files


def inspect_dataset(files):
    rows = []
    for path in files:
        a = ad.read_h5ad(path, backed="r")
        try:
            rows.append({"file": path.name, "cells": a.n_obs, "genes": a.n_vars,
                         "obsm": list(a.obsm.keys()),
                         "coordinates": [c for c in ("x", "y", "z", "x_ccf", "y_ccf", "z_ccf")
                                         if c in a.obs]})
        finally:
            a.file.close()
    return rows


def get_coordinates(a, file_index):
    for names in (("x_ccf", "y_ccf", "z_ccf"), ("x", "y", "z")):
        if set(names).issubset(a.obs.columns):
            return a.obs[list(names)].to_numpy(dtype=np.float32), list(names)
    for key in ("aligned_spatial", "spatial"):
        if key in a.obsm and a.obsm[key].shape[1] >= 3:
            return np.asarray(a.obsm[key][:, :3], dtype=np.float32), ["dim0", "dim1", "dim2"]
    if "spatial" in a.obsm:
        xy = np.asarray(a.obsm["spatial"][:, :2], dtype=np.float32)
    elif {"x", "y"}.issubset(a.obs):
        xy = a.obs[["x", "y"]].to_numpy(dtype=np.float32)
    else:
        raise ValueError("Section has no usable spatial coordinates")
    if xy.shape[1] != 2:
        raise ValueError("spatial must contain at least two coordinate columns")
    # Use the section number rather than its position among selected files.
    numbers = re.findall(r"\d+", Path(a.filename).stem)
    depth = float(numbers[-1]) if numbers else float(file_index)
    return np.column_stack([xy, np.full(len(xy), depth)]).astype(np.float32), ["x", "y", "section_index"]


def compute_hvg_plus_gini_sparse(matrices, n_top_genes):
    g = matrices[0].shape[1]
    if g <= n_top_genes:
        return np.arange(g)
    subs = []
    per_sec = max(1, 6000 // len(matrices))
    for x in matrices:
        idx = np.random.choice(x.shape[0], min(x.shape[0], per_sec), replace=False)
        subs.append(x[idx].toarray())
    x = np.concatenate(subs).astype(np.float32)
    dispersion = x.var(axis=0) / (x.mean(axis=0) + 1e-8)
    n = len(x)
    weights = 2 * np.arange(1, n + 1, dtype=np.float32) - n - 1
    gini = np.empty(g, dtype=np.float32)
    for start in range(0, g, 500):
        block = np.sort(x[:, start:start + 500], axis=0)
        gini[start:start + 500] = (weights[:, None] * block).sum(axis=0) / (n * block.sum(axis=0) + 1e-8)
    valid = x.sum(axis=0) > 0
    dispersion[~valid], gini[~valid] = -1, -1
    candidates = [*np.argsort(-dispersion)[:n_top_genes // 2], *np.argsort(-gini)]
    return np.asarray(list(dict.fromkeys(candidates))[:n_top_genes], dtype=np.int64)


def load_zhuang_sections(files, cfg):
    sections, common_genes = [], None
    for i, path in enumerate(files):
        a = ad.read_h5ad(path, backed="r")
        try:
            if not a.n_obs or not a.n_vars or not a.var_names.is_unique:
                raise ValueError(f"{path}: empty section or duplicate gene names")
            xyz, names = get_coordinates(a, i)
            if not np.isfinite(xyz).all():
                raise ValueError(f"{path}: coordinates contain NaN/Inf")
            limit = cfg["max_cells_per_section"]
            rows = np.arange(a.n_obs) if not limit or a.n_obs <= limit else np.sort(
                np.random.choice(a.n_obs, limit, replace=False))
            label_col = next((c for c in ("subclass", "supertype", "cluster", "class", "cell_type") if c in a.obs), None)
            labels = a.obs[label_col].astype(str).to_numpy()[rows] if label_col else None
            blocks = []
            for start in range(0, len(rows), cfg["inference_batch_size"]):
                block = sp.csr_matrix(a.X[rows[start:start + cfg["inference_batch_size"]]], dtype=np.float32)
                if not np.isfinite(block.data).all() or (block.data < 0).any():
                    raise ValueError(f"{path}: expression must be finite and non-negative")
                block.eliminate_zeros()
                blocks.append(block)
            genes = a.var_names.copy()
            sections.append(dict(file=path.name, xyz=xyz[rows], coord_names=names,
                                 var_names=genes, X_csr=sp.vstack(blocks, format="csr"),
                                 labels=labels, obs_names=a.obs_names.to_numpy()[rows]))
            common_genes = genes if common_genes is None else common_genes.intersection(genes, sort=False)
        finally:
            a.file.close()
        print(f"  Loaded {path.name}: {len(rows):,} cells", flush=True)
    if common_genes.empty:
        raise ValueError("No genes are shared by all sections")
    if any(s["coord_names"] != sections[0]["coord_names"] for s in sections):
        raise ValueError("Coordinate fields differ between sections")
    depth_axis = int(np.argmin(np.mean([s["xyz"].std(axis=0) for s in sections], axis=0)))
    plane_axes = [i for i in range(3) if i != depth_axis]
    if sections[0]["coord_names"] == ["x_ccf", "y_ccf", "z_ccf"] and depth_axis == 0:
        plane_axes = [2, 1]
    for s in sections:
        s["X_csr"] = s["X_csr"][:, s["var_names"].get_indexer(common_genes)].tocsr()
        s["z_val"] = float(np.median(s["xyz"][:, depth_axis]))
        s["xy"] = s.pop("xyz")[:, plane_axes]
        del s["var_names"]
    sections.sort(key=lambda s: s["z_val"])
    if any(a["z_val"] >= b["z_val"] for a, b in zip(sections, sections[1:])):
        raise ValueError("Section depths must be distinct; check coordinate fields")
    selected = compute_hvg_plus_gini_sparse([s["X_csr"] for s in sections], cfg["n_top_genes"])
    gene_names = common_genes[selected].to_numpy()
    sampled = []
    for s in sections:
        s["X_csr"] = s["X_csr"][:, selected].tocsr()
        sampled.append(s["X_csr"][:min(1000, s["X_csr"].shape[0])].data[:10000])
    values = np.concatenate(sampled)
    is_count = bool(values.size and np.allclose(values, np.round(values)))
    if cfg["expression_mode"] != "auto":
        is_count = cfg["expression_mode"] == "counts"
    print(f"[Data] {len(sections)} sections | {sum(s['X_csr'].shape[0] for s in sections):,} cells | "
          f"{len(gene_names)} genes | depth={sections[0]['coord_names'][depth_axis]} | "
          f"{'ZINB' if is_count else 'Hurdle-Gaussian'}", flush=True)
    return sections, gene_names, is_count, depth_axis, plane_axes


def prepare_geometry(sections):
    lo = np.min([s["xy"].min(axis=0) for s in sections], axis=0)
    hi = np.max([s["xy"].max(axis=0) for s in sections], axis=0)
    center = (lo + hi) / 2
    scale = float(max(hi - lo) / 2 + 1e-6)
    z_min, z_max = sections[0]["z_val"], sections[-1]["z_val"]
    for s in sections:
        s["s"] = torch.from_numpy(((s["xy"] - center) / scale).astype(np.float32))
        s["z_norm"] = (s["z_val"] - z_min) / (z_max - z_min)
    sub = np.concatenate([s["X_csr"][:min(500, s["X_csr"].shape[0])].toarray() for s in sections])
    weights = np.clip((0.1 / ((sub > 0).mean(axis=0) + 1e-3)) ** 0.35, 1, 4).astype(np.float32)
    return center, scale, weights, max(float(np.median(sub.sum(axis=1))), 1.0)
