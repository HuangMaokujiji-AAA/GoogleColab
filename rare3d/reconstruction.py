"""Tiled Heun integration and incremental sparse 3D export."""
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from anndata.io import sparse_dataset, write_elem

from .transport import build_adaptive_quadtree_tiles, sample_rare_anchored_prior
from .training import crop_mask, get_context, make_flow_context

@torch.no_grad()
def heun_ode_reconstruct(flow_net, t_rel, s0, e0, p0, ctx_s, ctx_e, ctx_dz, rare_protos, n_steps=30, cfg_base=1.2, cfg_rare=2.2):
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    N = s0.shape[0]
    if N == 0:
        return s0, e0
    dev = s0.device
    s_cur, e_cur = s0.clone(), e0.clone()
    q_m = torch.ones(1, N, dtype=torch.bool, device=dev)
    c_m = torch.ones(1, ctx_s.shape[0], dtype=torch.bool, device=dev)
    t_rel_t = torch.tensor([t_rel], device=dev, dtype=torch.float32)
    is_r = torch.tensor([int(p) in rare_protos for p in p0.cpu()], device=dev)
    omega = torch.where(is_r, torch.full((N,), cfg_rare, device=dev), torch.full((N,), cfg_base, device=dev)).unsqueeze(-1)

    def eval_v(tau_val, si, ei):
        tt = torch.tensor([tau_val], device=dev, dtype=torch.float32)
        vsc, vec = flow_net(tt, t_rel_t, si.unsqueeze(0), ei.unsqueeze(0), q_m, ctx_s.unsqueeze(0), ctx_e.unsqueeze(0), ctx_dz.unsqueeze(0), c_m, torch.tensor([False], device=dev))
        vsu, veu = flow_net(tt, t_rel_t, si.unsqueeze(0), ei.unsqueeze(0), q_m, ctx_s.unsqueeze(0), ctx_e.unsqueeze(0), ctx_dz.unsqueeze(0), c_m, torch.tensor([True], device=dev))
        return vsu[0] + omega * (vsc[0] - vsu[0]), veu[0] + omega * (vec[0] - veu[0])

    dt = 1.0 / n_steps
    for step in range(n_steps):
        vs0, ve0 = eval_v(step * dt, s_cur, e_cur)
        sp, ep = s_cur + dt * vs0, e_cur + dt * ve0
        vs1, ve1 = eval_v((step + 1) * dt, sp, ep)
        s_cur = s_cur + 0.5 * dt * (vs0 + vs1)
        e_cur = e_cur + 0.5 * dt * (ve0 + ve1)
    return s_cur, e_cur


@torch.no_grad()
def reconstruct_between(sections, ka, kb, t_rel, vae, flow, niche_pred,
                        centers, rare_protos, median_library, cfg, device):
    if not 0 <= t_rel <= 1 or ka == kb:
        raise ValueError("Reconstruction requires two distinct slices and 0 <= t_rel <= 1")
    a, b = sections[ka], sections[kb]
    tiles = build_adaptive_quadtree_tiles(np.concatenate([a["s"].numpy(), b["s"].numpy()]), cfg["max_cells_per_tile"])
    centers_t = torch.as_tensor(centers, device=device)
    s_out, e_out, x_out = [], [], []
    for tile in tiles:
        core_a, core_b = crop_mask(a, tile), crop_mask(b, tile)
        n_est = round((1 - t_rel) * int(core_a.sum()) + t_rel * int(core_b.sum()))
        if not n_est:
            continue
        ca = get_context(a, crop_mask(a, tile, cfg["tile_halo"]), cfg["max_context_cells"], device)
        cb = get_context(b, crop_mask(b, tile, cfg["tile_halo"]), cfg["max_context_cells"], device)
        sa, ea = ca[:2]; sb, eb = cb[:2]
        s0, e0, p0 = sample_rare_anchored_prior(*ca, *cb, t_rel, n_est, centers_t, rare_protos, niche_pred)
        cs, ce, dz = make_flow_context(sa, ea, sb, eb, t_rel, cfg["max_context_cells"])
        for start in range(0, n_est, cfg["max_cells_per_tile"]):
            end = start + cfg["max_cells_per_tile"]
            sr, er = heun_ode_reconstruct(flow, t_rel, s0[start:end], e0[start:end], p0[start:end],
                                          cs, ce, dz, rare_protos, cfg["n_ode_steps"], cfg["cfg_base"], cfg["cfg_rare"])
            if not torch.isfinite(sr).all() or not torch.isfinite(er).all():
                raise FloatingPointError("Non-finite reconstructed coordinates or embeddings")
            x0, x1, y0, y1 = tile
            keep = (sr[:, 0] >= x0) & (sr[:, 0] < x1) & (sr[:, 1] >= y0) & (sr[:, 1] < y1)
            sr, er = sr[keep], er[keep]
            if not len(sr):
                continue
            lib = er.new_full((len(er), 1), median_library if vae.is_count else 1.0)
            mu, _, drop = vae.decode(er, lib)
            x = (1 - torch.sigmoid(drop)) * mu
            x[x < 1e-3] = 0
            if not torch.isfinite(x).all():
                raise FloatingPointError("Non-finite decoded expression")
            s_out.append(sr.cpu().numpy()); e_out.append(er.cpu().numpy())
            x_out.append(sp.csr_matrix(x.cpu().numpy()))
    if not s_out:
        return (np.empty((0, 2), dtype=np.float32), np.empty((0, cfg["d_cell"]), dtype=np.float32),
                sp.csr_matrix((0, vae.out_mean.out_features), dtype=np.float32))
    return np.concatenate(s_out), np.concatenate(e_out), sp.vstack(x_out, format="csr")


def export_volume(sections, gene_names, xy_center, xy_scale, depth_axis, plane_axes,
                  reconstruct, output_dir):
    """Append X section by section; never concatenate the full expression volume."""
    path = Path(output_dir) / "zhuang_abca2_rare3d_reconstructed.h5ad"
    partial = path.with_suffix(".partial.h5ad")
    empty = sp.csr_matrix((0, len(gene_names)), dtype=np.float32)
    empty.indptr = empty.indptr.astype(np.int64)
    initial = ad.AnnData(X=empty, var=pd.DataFrame(index=pd.Index(gene_names)))
    initial.write_h5ad(partial, compression="gzip")
    coords, depths, observations = [], [], []
    with h5py.File(partial, "a") as f:
        matrix = sparse_dataset(f["X"])

        def append(x, xy, z, source, section_name, ids, labels=None):
            if x.shape[0] == 0:
                return
            matrix.append(x.tocsr())
            coords.append(np.asarray(xy, dtype=np.float32))
            depths.append(np.full(len(xy), z, dtype=np.float32))
            obs = pd.DataFrame({"source": source, "section": section_name}, index=ids)
            obs["cell_type"] = labels if labels is not None else "unknown"
            observations.append(obs)

        for k, section in enumerate(sections):
            append(section["X_csr"], section["xy"], section["z_val"], "observed", section["file"],
                   [f"{section['file']}:{i}" for i in section["obs_names"]], section["labels"])
            if k + 1 < len(sections):
                s, _, x = reconstruct(k, k + 1, 0.5)
                append(x, s * xy_scale + xy_center, (section["z_val"] + sections[k + 1]["z_val"]) / 2,
                       "reconstructed_FM", f"mid_{k:03d}_{k + 1:03d}",
                       [f"mid_{k:03d}_{k + 1:03d}:{i}" for i in range(len(s))])
                print(f"  Exported interval {k + 1}/{len(sections) - 1}: {len(s):,} generated cells", flush=True)
        xy, z = np.concatenate(coords), np.concatenate(depths)
        xyz = np.empty((len(z), 3), dtype=np.float32)
        xyz[:, depth_axis] = z
        xyz[:, plane_axes] = xy
        obs = pd.concat(observations)
        obs["section_depth"] = z
        names = sections[0]["coord_names"]
        if names in (["x_ccf", "y_ccf", "z_ccf"], ["x", "y", "z"]):
            for i, name in enumerate(names):
                obs[name] = xyz[:, i]
        for col in ("source", "section", "cell_type"):
            obs[col] = obs[col].astype("category")
        write_elem(f, "obs", obs)
        write_elem(f, "obsm", {"spatial": xy, "spatial_3d": xyz})
        write_elem(f, "uns", {"coordinate_names": names, "depth_axis": depth_axis,
                               "plane_axes": np.asarray(plane_axes),
                               "method": "Rare3D-FM"})
    partial.replace(path)
    print(f"[Saved] {path}", flush=True)
    return path
