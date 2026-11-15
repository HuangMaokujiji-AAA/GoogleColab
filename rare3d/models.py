"""VAE, heterophily niche encoder and conditional flow network."""
import math
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial import cKDTree

class IntrinsicDualVAE(nn.Module):
    def __init__(self, n_genes: int, d_cell: int, is_count: bool, hidden: int = 256):
        super().__init__()
        self.is_count = is_count
        self.enc = nn.Sequential(
            nn.Linear(n_genes, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.fc_mu = nn.Linear(hidden, d_cell)
        self.fc_logvar = nn.Linear(hidden, d_cell)
        self.dec = nn.Sequential(
            nn.Linear(d_cell, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.out_mean = nn.Linear(hidden, n_genes)
        self.out_disp = nn.Linear(hidden, n_genes)
        self.out_drop = nn.Linear(hidden, n_genes)

    def encode(self, x: torch.Tensor, lib_size: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_in = torch.log1p(x / (lib_size + 1e-8) * 1e4) if self.is_count else x
        h = self.enc(x_in)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-8.0, 4.0)

    def decode(self, z: torch.Tensor, lib_size: torch.Tensor):
        h = self.dec(z)
        drop_logits = self.out_drop(h)
        if self.is_count:
            mu = F.softmax(self.out_mean(h), dim=-1) * lib_size
            theta = torch.exp(self.out_disp(h).clamp(-4.0, 4.0))
            return mu, theta, drop_logits
        mu = F.softplus(self.out_mean(h))
        log_var = self.out_disp(h).clamp(-6.0, 3.0)
        return mu, log_var, drop_logits

    def loss_fn(self, x: torch.Tensor, lib_size: torch.Tensor, gene_weights: torch.Tensor):
        mu_z, logvar_z = self.encode(x, lib_size)
        std = torch.exp(0.5 * logvar_z)
        z = mu_z + torch.randn_like(std) * std if self.training else mu_z
        eps = 1e-8

        if self.is_count:
            mu, theta, drop_logits = self.decode(z, lib_size)
            tpm = theta + mu + eps
            nb_case = (
                torch.lgamma(x + theta + eps) - torch.lgamma(theta + eps) - torch.lgamma(x + 1.0)
                + theta * torch.log(theta / tpm + eps) + x * torch.log(mu / tpm + eps)
            )
            nb_zero = theta * torch.log(theta / tpm + eps)
            log_pi = -F.softplus(-drop_logits)
            log_1mp = -F.softplus(drop_logits)
            log_prob = torch.where(x < 0.5, torch.logaddexp(log_pi, log_1mp + nb_zero), log_1mp + nb_case)
        else:
            mu, log_var, drop_logits = self.decode(z, lib_size)
            log_pi = -F.softplus(-drop_logits)
            log_1mp = -F.softplus(drop_logits)
            gauss_lp = -0.5 * (math.log(2 * math.pi) + log_var + ((x - mu) ** 2) / (torch.exp(log_var) + eps))
            log_prob = torch.where(x <= 1e-5, log_pi, log_1mp + gauss_lp)

        recon_nll = -(log_prob * gene_weights[None, :]).sum(dim=-1)
        kl = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=-1)
        return (recon_nll + 0.5 * kl).mean()

@torch.no_grad()
def chunked_knn(query_xy, ref_xy, k, exclude_self=False):
    """Exact spatial neighbours without an N-by-M distance matrix.

    exclude_self assumes query and reference have the same row order.
    """
    n, m = len(query_xy), len(ref_xy)
    if exclude_self and (n != m or not torch.equal(query_xy, ref_xy)):
        raise ValueError("exclude_self requires identical query/reference rows")
    k = min(k, m - int(exclude_self))
    if k < 1:
        return (query_xy.new_empty((n, 0)),
                torch.empty((n, 0), device=query_xy.device, dtype=torch.long))
    if n == 0:
        return (query_xy.new_empty((0, k)),
                torch.empty((0, k), device=query_xy.device, dtype=torch.long))
    tree = cKDTree(ref_xy.detach().cpu().numpy())
    dists, inds = tree.query(query_xy.detach().cpu().numpy(), k=k + int(exclude_self))
    dists, inds = np.asarray(dists).reshape(n, -1), np.asarray(inds).reshape(n, -1)
    if exclude_self:
        # Remove the row itself even when coordinates contain exact duplicates.
        dists[inds == np.arange(n)[:, None]] = np.inf
        order = np.argsort(dists, axis=1, kind="stable")[:, :k]
        inds = np.take_along_axis(inds, order, axis=1)
        dists = np.take_along_axis(dists, order, axis=1)
    return (torch.as_tensor(dists, dtype=query_xy.dtype, device=query_xy.device),
            torch.as_tensor(inds, dtype=torch.long, device=query_xy.device))


class HeterophilyNicheEncoder(nn.Module):
    def __init__(self, d_cell: int, d_niche: int):
        super().__init__()
        self.q_proj = nn.Linear(d_cell, d_niche)
        self.k_proj = nn.Linear(d_cell + 3, d_niche)
        self.v_proj = nn.Linear(d_cell + 3, d_niche)
        self.gate = nn.Sequential(nn.Linear(d_cell + d_niche, d_niche), nn.Sigmoid())
        self.out = nn.Sequential(nn.Linear(d_niche, d_niche), nn.LayerNorm(d_niche), nn.SiLU())

    def forward(self, xy: torch.Tensor, e_cell: torch.Tensor, k: int = 12) -> torch.Tensor:
        N = xy.shape[0]
        if N <= 1:
            return xy.new_zeros((N, self.q_proj.out_features))
        dists, nbr_idx = chunked_knn(xy, xy, k=min(k, N - 1), exclude_self=True)
        rel_xy = xy[nbr_idx] - xy[:, None, :]
        return self.aggregate(e_cell, e_cell[nbr_idx], rel_xy, dists)

    def aggregate(self, e_cell, neighbour_e, rel_xy, dists):
        """Shared attention step for full-section and chunked inference."""
        feat = torch.cat([neighbour_e, rel_xy, dists.unsqueeze(-1)], dim=-1)
        q = self.q_proj(e_cell).unsqueeze(1)
        k_m = self.k_proj(feat)
        v_m = self.v_proj(feat)
        attn = torch.softmax((q * k_m).sum(dim=-1) / math.sqrt(k_m.shape[-1]), dim=-1)
        agg = (attn.unsqueeze(-1) * v_m).sum(dim=1)
        return self.out(self.gate(torch.cat([e_cell, agg], dim=-1)) * agg)

class NicheIntensityPredictor(nn.Module):
    def __init__(self, d_cell: int, d_niche: int, n_protos: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_cell + d_niche + 3, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
        )
        self.proto_head = nn.Linear(128, n_protos)

    def _ctx(self, q_xy, ctx_xy, ctx_e, ctx_n, k=8):
        if len(ctx_xy) == 0:
            return (q_xy.new_zeros((len(q_xy), ctx_e.shape[1])),
                    q_xy.new_zeros((len(q_xy), ctx_n.shape[1])),
                    q_xy.new_ones((len(q_xy), 1)))
        dists, inds = chunked_knn(q_xy, ctx_xy, k=min(k, ctx_xy.shape[0]))
        w = torch.softmax(-dists * 8.0, dim=-1).unsqueeze(-1)
        return (w * ctx_e[inds]).sum(dim=1), (w * ctx_n[inds]).sum(dim=1), dists.mean(dim=-1, keepdim=True)

    def forward(self, q_xy, sa, ea, na, sb, eb, nb, t_rel: float):
        ea_g, na_g, da = self._ctx(q_xy, sa, ea, na)
        eb_g, nb_g, db = self._ctx(q_xy, sb, eb, nb)
        wa, wb = 1.0 - t_rel, t_rel
        feat = torch.cat([wa * ea_g + wb * eb_g, wa * na_g + wb * nb_g,
                          torch.full((q_xy.shape[0], 1), t_rel, device=q_xy.device), da, db], dim=-1)
        return self.proto_head(self.net(feat))

class SinusoidalEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(1, half - 1))
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class SpatialBiasedAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, is_cross: bool = False):
        super().__init__()
        self.is_cross, self.n_heads, self.head_dim = is_cross, n_heads, d_model // n_heads
        self.norm_q = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm_kv = nn.LayerNorm(d_model) if is_cross else None
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.log_gamma = nn.Parameter(torch.linspace(1.0, 4.0, n_heads))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 3 * d_model))
        self.mlp_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model))
        self.ada_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 3 * d_model))

    def forward(self, h, s_q, t_emb, kv=None, s_kv=None, q_mask=None, kv_mask=None):
        B, N, D = h.shape
        sh1, sc1, g1 = self.ada(t_emb).unsqueeze(1).chunk(3, dim=-1)
        q_in = self.norm_q(h) * (1.0 + sc1) + sh1
        kv_in, s_k = (self.norm_kv(kv), s_kv) if self.is_cross else (q_in, s_q)
        M = kv_in.shape[1]
        q = self.q_proj(q_in).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_in).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logits = logits - torch.exp(self.log_gamma).view(1, self.n_heads, 1, 1) * torch.cdist(s_q, s_k).pow(2).unsqueeze(1)
        if kv_mask is not None:
            logits = logits.masked_fill(~kv_mask[:, None, None, :], -1e4)
        out = (torch.softmax(logits, dim=-1) @ v).transpose(1, 2).contiguous().view(B, N, D)
        h = h + g1 * self.out_proj(out)
        sh2, sc2, g2 = self.ada_mlp(t_emb).unsqueeze(1).chunk(3, dim=-1)
        h = h + g2 * self.mlp(self.mlp_norm(h) * (1.0 + sc2) + sh2)
        return h * q_mask.unsqueeze(-1).float() if q_mask is not None else h

class RareFlowNet(nn.Module):
    def __init__(self, d_cell: int, d_model: int = 128, n_layers: int = 3):
        super().__init__()
        self.t_sin, self.z_sin = SinusoidalEmb(64), SinusoidalEmb(64)
        self.cond_proj = nn.Sequential(nn.Linear(128, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.in_proj = nn.Sequential(nn.Linear(2 + d_cell, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.ctx_proj = nn.Sequential(nn.Linear(2 + d_cell + 1, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.null_ctx = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({"cross": SpatialBiasedAttentionBlock(d_model, 4, True),
                           "self": SpatialBiasedAttentionBlock(d_model, 4, False)})
            for _ in range(n_layers)
        ])
        self.head_sp = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.SiLU(), nn.Linear(d_model // 2, 2))
        self.expr_gate = nn.Linear(d_model + d_cell, d_model)
        self.head_ex = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_cell))

    def forward(self, tau, t_rel, s_tau, e_tau, q_mask, ctx_s, ctx_e, ctx_dz, ctx_mask, uncond_drop=None):
        B = s_tau.shape[0]
        t_emb = self.cond_proj(torch.cat([self.t_sin(tau), self.z_sin(t_rel)], dim=-1))
        h = self.in_proj(torch.cat([s_tau, e_tau], dim=-1))
        ctx_h = self.ctx_proj(torch.cat([ctx_s, ctx_e, ctx_dz], dim=-1))
        eff_mask = torch.where(uncond_drop[:, None], torch.zeros_like(ctx_mask), ctx_mask) if (uncond_drop is not None and uncond_drop.any()) else ctx_mask
        full_kv = torch.cat([self.null_ctx.expand(B, 1, -1), ctx_h], dim=1)
        full_s = torch.cat([torch.zeros(B, 1, 2, device=s_tau.device), ctx_s], dim=1)
        full_m = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=s_tau.device), eff_mask], dim=1)
        for blk in self.blocks:
            h = blk["cross"](h, s_tau, t_emb, kv=full_kv, s_kv=full_s, q_mask=q_mask, kv_mask=full_m)
            h = blk["self"](h, s_tau, t_emb, q_mask=q_mask, kv_mask=q_mask)
        v_sp = self.head_sp(h)
        v_ex = self.head_ex(h * torch.sigmoid(self.expr_gate(torch.cat([h, e_tau], dim=-1))))
        return v_sp, v_ex
