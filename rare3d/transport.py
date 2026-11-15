"""Rare-weighted transport, spatial tiling and anchored priors."""
import math

import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from .models import NicheIntensityPredictor

@torch.no_grad()
def solve_rw_uot(s_a, e_a, n_a, w_a, s_b, e_b, n_b, w_b, eps=0.05, n_iters=35, cos_thresh=0.35):
    Na, Nb = s_a.shape[0], s_b.shape[0]
    if Na == 0 or Nb == 0:
        return torch.zeros(Na, Nb, device=s_a.device)

    d_sp = torch.cdist(s_a, s_b).pow(2)
    d_ex = torch.cdist(e_a, e_b).pow(2) / max(1.0, e_a.shape[1] * 0.25)
    d_ni = torch.cdist(n_a, n_b).pow(2) / max(1.0, n_a.shape[1] * 0.25)

    cos_sim = F.normalize(e_a, dim=-1) @ F.normalize(e_b, dim=-1).T
    barrier = torch.where(cos_sim < cos_thresh, torch.full_like(cos_sim, 50.0), torch.zeros_like(cos_sim))
    C = (1.5 * d_sp + 1.0 * d_ex + 0.3 * d_ni + barrier).clamp(max=80.0)

    log_a = torch.full((Na,), -math.log(Na), device=s_a.device)
    log_b = torch.full((Nb,), -math.log(Nb), device=s_a.device)

    compat_a = ((cos_sim >= cos_thresh) & (d_sp < 0.25)).any(dim=1).float()
    compat_b = ((cos_sim >= cos_thresh) & (d_sp < 0.25)).any(dim=0).float()
    lam_a = 0.5 + 1.5 * compat_a * torch.log1p(w_a)
    lam_b = 0.5 + 1.5 * compat_b * torch.log1p(w_b)
    tau_a, tau_b = lam_a / (lam_a + eps), lam_b / (lam_b + eps)

    f, g = torch.zeros(Na, device=s_a.device), torch.zeros(Nb, device=s_a.device)
    for _ in range(n_iters):
        f = tau_a * (eps * log_a - eps * torch.logsumexp((g[None, :] - C) / eps, dim=1))
        g = tau_b * (eps * log_b - eps * torch.logsumexp((f[:, None] - C) / eps, dim=0))
    return torch.exp(((f[:, None] + g[None, :] - C) / eps).clamp(max=0.0))

def build_adaptive_quadtree_tiles(xy_ref: np.ndarray, max_cells: int = 900, min_size: float = 0.08):
    """基于单层典型细胞分布构建四叉树分块，自动忽略脑外空白且均衡密集区"""
    if max_cells < 1:
        raise ValueError("max_cells must be positive")
    if len(xy_ref) == 0:
        return []
    tiles = []
    def _split(x0, x1, y0, y1):
        m = (xy_ref[:, 0] >= x0) & (xy_ref[:, 0] < x1) & (xy_ref[:, 1] >= y0) & (xy_ref[:, 1] < y1)
        cnt = int(m.sum())
        if cnt == 0:
            return
        if cnt <= max_cells or (x1 - x0) <= min_size or (y1 - y0) <= min_size:
            tiles.append((float(x0), float(x1), float(y0), float(y1)))
            return
        xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        _split(x0, xm, y0, ym); _split(xm, x1, y0, ym)
        _split(x0, xm, ym, y1); _split(xm, x1, ym, y1)
    lo, hi = xy_ref.min(axis=0), xy_ref.max(axis=0)
    pad = np.maximum((hi - lo) * 1e-5, 1e-5)
    _split(lo[0] - pad[0], hi[0] + pad[0], lo[1] - pad[1], hi[1] + pad[1])
    return tiles

@torch.no_grad()
def sample_rare_anchored_prior(
    sa, ea, na, wa, pa, sb, eb, nb, wb, pb,
    t_rel: float, n_target: int, centers_t: torch.Tensor, rare_protos: set,
    niche_pred: NicheIntensityPredictor, sigma_s: float = 0.025, sigma_e: float = 0.15
):
    dev = sa.device
    d_c = ea.shape[1]
    if n_target == 0:
        return torch.empty(0, 2, device=dev), torch.empty(0, d_c, device=dev), torch.empty(0, dtype=torch.long, device=dev)

    anchor_s, anchor_e, anchor_p, anchor_w = [], [], [], []
    if sa.shape[0] > 0 and sb.shape[0] > 0:
        T = solve_rw_uot(sa, ea, na, wa, sb, eb, nb, wb)
        max_v, max_idx = T.max(dim=1)
        paired = max_v > (1.0 / (max(sa.shape[0], sb.shape[0]) * 5.0))
        if paired.any():
            ia = torch.where(paired)[0]
            ib = max_idx[ia]
            anchor_s.append((1.0 - t_rel) * sa[ia] + t_rel * sb[ib])
            anchor_e.append((1.0 - t_rel) * ea[ia] + t_rel * eb[ib])
            anchor_p.append(pa[ia] if t_rel < 0.5 else pb[ib])
            anchor_w.append(max_v[ia] * (1.0 + 0.3 * torch.log1p(wa[ia])))

        row_m, col_m = T.sum(dim=1), T.sum(dim=0)
        rare_a = torch.tensor([int(p) in rare_protos for p in pa.cpu()], device=dev)
        rare_b = torch.tensor([int(p) in rare_protos for p in pb.cpu()], device=dev)
        unm_a = (row_m < 0.2 / max(1, sa.shape[0])) & rare_a
        unm_b = (col_m < 0.2 / max(1, sb.shape[0])) & rare_b
        if unm_a.any():
            anchor_s.append(sa[unm_a]); anchor_e.append(ea[unm_a]); anchor_p.append(pa[unm_a])
            anchor_w.append(torch.full((unm_a.sum(),), math.exp(-t_rel * 1.2) * 0.05, device=dev))
        if unm_b.any():
            anchor_s.append(sb[unm_b]); anchor_e.append(eb[unm_b]); anchor_p.append(pb[unm_b])
            anchor_w.append(torch.full((unm_b.sum(),), math.exp(-(1.0 - t_rel) * 1.2) * 0.05, device=dev))

        if len(rare_protos) > 0:
            base_pts = torch.cat([sa, sb], dim=0)
            cand_xy = base_pts + torch.randn_like(base_pts) * sigma_s
            probs = torch.softmax(niche_pred(cand_xy, sa, ea, na, sb, eb, nb, t_rel), dim=-1)
            rare_idx_t = torch.tensor(list(rare_protos), device=dev, dtype=torch.long)
            rare_prob_sum = probs[:, rare_idx_t].sum(dim=-1)
            sel = rare_prob_sum > 0.15
            if sel.any():
                picked = rare_idx_t[torch.multinomial(probs[sel][:, rare_idx_t], num_samples=1).squeeze(-1)]
                anchor_s.append(cand_xy[sel]); anchor_e.append(centers_t[picked]); anchor_p.append(picked)
                anchor_w.append(rare_prob_sum[sel] * 0.15)

    if not anchor_s and len(sa) + len(sb):
        anchor_s.append(torch.cat([sa, sb]))
        anchor_e.append(torch.cat([ea, eb]))
        anchor_p.append(torch.cat([pa, pb]))
        anchor_w.append(torch.cat([wa, wb]).clamp(min=1e-6))

    if not anchor_s:
        s0 = (torch.rand(n_target, 2, device=dev) * 2.0 - 1.0)
        p0 = torch.randint(0, centers_t.shape[0], (n_target,), device=dev)
        return s0, centers_t[p0] + torch.randn(n_target, d_c, device=dev) * sigma_e, p0

    all_s, all_e = torch.cat(anchor_s, 0), torch.cat(anchor_e, 0)
    all_p, all_w = torch.cat(anchor_p, 0), torch.cat(anchor_w, 0).clamp(min=1e-6)
    chosen = torch.multinomial(all_w / all_w.sum(), num_samples=n_target, replacement=True)
    return (all_s[chosen] + torch.randn(n_target, 2, device=dev) * sigma_s,
            all_e[chosen] + torch.randn(n_target, d_c, device=dev) * sigma_e,
            all_p[chosen])

@torch.no_grad()
def prototype_constrained_ot_coupling(s0, e0, p0, s1, e1, p1, rare_protos: set):
    N = s1.shape[0]
    if N <= 1:
        idx = torch.arange(N, device=s1.device)
        return idx, idx
    cost = torch.cdist(s0, s1).pow(2) + 0.5 * torch.cdist(e0, e1).pow(2)
    is_rare_1 = torch.tensor([int(p) in rare_protos for p in p1.cpu()], device=s1.device)
    cost = cost + ((p0[:, None] != p1[None, :]) & is_rare_1[None, :]).float() * 50.0
    r_ind, c_ind = linear_sum_assignment(cost.cpu().numpy())
    return torch.tensor(r_ind, device=s1.device, dtype=torch.long), torch.tensor(c_ind, device=s1.device, dtype=torch.long)
