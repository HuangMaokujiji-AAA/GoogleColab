"""Train the three model stages; section arrays stay on CPU between batches."""
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from torch.nn import functional as F

from .models import IntrinsicDualVAE, HeterophilyNicheEncoder, NicheIntensityPredictor, RareFlowNet
from .checkpoints import cpu_state_dict, save_checkpoint
from .transport import build_adaptive_quadtree_tiles, sample_rare_anchored_prior, prototype_constrained_ot_coupling

CONTEXT_KEYS = ("s", "e", "n", "w", "p")


def checked_step(loss, optimizer, parameters, max_norm=5.0):
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite training loss; inspect expression values and learning rate")
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    optimizer.step()


def train_vae(sections, n_genes, is_count, gene_weights, cfg, device):
    print("[Phase 1/4] Training intrinsic VAE", flush=True)
    vae = IntrinsicDualVAE(n_genes, cfg["d_cell"], is_count).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=1e-3, weight_decay=1e-4)
    weights = torch.as_tensor(gene_weights, device=device)
    for ep in range(cfg["vae_epochs"]):
        vae.train()
        total = 0.0
        for _ in range(cfg["max_vae_steps_per_epoch"]):
            x = random.choice(sections)["X_csr"]
            idx = np.random.randint(x.shape[0], size=min(cfg["batch_size"], x.shape[0]))
            xb = torch.as_tensor(x[idx].toarray(), device=device)
            loss = vae.loss_fn(xb, xb.sum(-1, keepdim=True).clamp(min=1), weights)
            checked_step(loss, opt, vae.parameters())
            total += loss.item()
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"  VAE {ep + 1}/{cfg['vae_epochs']} | loss={total / cfg['max_vae_steps_per_epoch']:.4f}", flush=True)
    save_checkpoint(cpu_state_dict(vae), Path(cfg["output_dir"]) / "checkpoints" / "vae.pt")
    vae.eval()
    with torch.no_grad():
        for s in sections:
            x, parts = s["X_csr"], []
            for start in range(0, x.shape[0], cfg["inference_batch_size"]):
                xb = torch.as_tensor(x[start:start + cfg["inference_batch_size"]].toarray(), device=device)
                mu, _ = vae.encode(xb, xb.sum(-1, keepdim=True).clamp(min=1))
                parts.append(mu.cpu())
            s["e"] = torch.cat(parts)
    return vae


def discover_prototypes(sections, cfg):
    e = np.concatenate([s["e"].numpy() for s in sections])
    n = len(e)
    ref_idx = np.random.choice(n, min(n, 8000), replace=False)
    nn = NearestNeighbors(n_neighbors=min(15, len(ref_idx)), algorithm="brute", n_jobs=1).fit(e[ref_idx])
    sample = e[np.random.choice(n, min(n, 4000), replace=False)]
    bw = max(float(np.median(nn.kneighbors(sample)[0][:, -1])), 1e-3)
    rho = np.empty(n, dtype=np.float32)
    for start in range(0, n, 2048):
        dists = nn.kneighbors(e[start:start + 2048])[0]
        rho[start:start + 2048] = np.exp(-0.5 * (dists / bw) ** 2).mean(axis=1)
    weights = np.clip((max(float(np.median(rho)), 1e-8) / (rho + 1e-8)) ** cfg["rare_gamma"], 1, cfg["rare_w_max"])
    fit_n = min(n, 120000)
    p = np.sqrt(weights)
    idx = np.random.choice(n, fit_n, replace=False, p=p / p.sum()) if n > fit_n else np.arange(n)
    count = min(cfg["n_prototypes"], n, max(4, n // 15))
    km = MiniBatchKMeans(n_clusters=count, random_state=cfg["seed"], n_init=8, batch_size=2048)
    km.fit(e[idx], sample_weight=np.sqrt(weights[idx]))
    ids = np.concatenate([km.predict(e[i:i + 50000]) for i in range(0, n, 50000)])
    counts = np.bincount(ids, minlength=count)
    rare = (counts / n < cfg["rare_freq_thresh"]) & (counts > 0)
    if not rare.any():
        proto_w = np.array([weights[ids == i].mean() if counts[i] else 0 for i in range(count)])
        rare = (proto_w >= np.quantile(proto_w[counts > 0], 0.85)) & (counts > 0)
    offset = 0
    for s in sections:
        end = offset + len(s["s"])
        s["p"] = torch.from_numpy(ids[offset:end].astype(np.int64))
        s["w"] = torch.from_numpy(weights[offset:end].astype(np.float32))
        offset = end
    rare_ids = set(np.flatnonzero(rare).tolist())
    major_ids = set(np.flatnonzero(~rare & (counts > 0)).tolist())
    print(f"[Prototypes] {count} total, {len(rare_ids)} rare", flush=True)
    return km.cluster_centers_.astype(np.float32), rare_ids, major_ids


def build_train_triplets(n_sections, validation_index):
    available = [k for k in range(n_sections) if k != validation_index]
    triplets = []
    for stride in (1, 2):
        for i in range(len(available) - stride):
            a, b = available[i], available[i + stride]
            triplets.extend((a, m, b) for m in available if a < m < b)
    if not triplets:
        a, b = available[0], available[-1]
        triplets = [(a, a, b), (a, b, b)]
    return triplets


def random_rows(n, limit):
    return torch.randperm(n)[:limit] if n > limit else torch.arange(n)


def crop_mask(section, tile, halo=0):
    x0, x1, y0, y1 = tile
    s = section["s"]
    return ((s[:, 0] >= x0 - halo) & (s[:, 0] < x1 + halo)
            & (s[:, 1] >= y0 - halo) & (s[:, 1] < y1 + halo))


def get_context(section, mask, limit, device, exclude=None):
    idx = torch.where(mask)[0]
    if exclude is not None:
        idx = idx[~torch.isin(idx, exclude)]
    idx = idx[random_rows(len(idx), limit)]
    return tuple(section[key][idx].to(device) for key in CONTEXT_KEYS)


def make_flow_context(sa, ea, sb, eb, t_rel, limit):
    s, e = torch.cat([sa, sb]), torch.cat([ea, eb])
    dz = torch.cat([sa.new_full((len(sa), 1), -t_rel), sb.new_full((len(sb), 1), 1 - t_rel)])
    idx = random_rows(len(s), limit).to(s.device)
    return s[idx], e[idx], dz[idx]


def train_niche(sections, triplets, n_protos, cfg, device):
    print("[Phase 2/4] Training niche encoder and intensity predictor", flush=True)
    enc = HeterophilyNicheEncoder(cfg["d_cell"], cfg["d_niche"]).to(device)
    pred = NicheIntensityPredictor(cfg["d_cell"], cfg["d_niche"], n_protos).to(device)
    params = list(enc.parameters()) + list(pred.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    for ep in range(cfg["niche_epochs"]):
        enc.train(); pred.train()
        order = random.sample(triplets, min(len(triplets), cfg["max_niche_steps_per_epoch"]))
        total = 0.0
        for a, m, b in order:
            target = sections[m]
            q = random_rows(len(target["s"]), min(cfg["niche_query_cells"], max(1, len(target["s"]) // 3)))
            contexts = []
            for k in (a, b):
                idx = torch.arange(len(sections[k]["s"]))
                if k == m:
                    idx = idx[~torch.isin(idx, q)]
                idx = idx[random_rows(len(idx), cfg["niche_context_cells"])]
                s, e = sections[k]["s"][idx].to(device), sections[k]["e"][idx].to(device)
                contexts.extend((s, e, enc(s, e)))
            t_rel = (target["z_norm"] - sections[a]["z_norm"]) / (sections[b]["z_norm"] - sections[a]["z_norm"])
            logits = pred(target["s"][q].to(device), *contexts, t_rel)
            loss = (F.cross_entropy(logits, target["p"][q].to(device), reduction="none") * target["w"][q].to(device)).mean()
            checked_step(loss, opt, params)
            total += loss.item()
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"  Niche {ep + 1}/{cfg['niche_epochs']} | loss={total / len(order):.4f}", flush=True)
    for name, model in (("niche_encoder", enc), ("niche_predictor", pred)):
        save_checkpoint(cpu_state_dict(model), Path(cfg["output_dir"]) / "checkpoints" / f"{name}.pt")
    enc.eval(); pred.eval()
    # Each inference chunk includes neighbours from the complete section.
    from .models import chunked_knn
    with torch.no_grad():
        for section in sections:
            s, e = section["s"], section["e"]
            dists, neighbours = chunked_knn(s, s, 12, exclude_self=True)
            parts = []
            for start in range(0, len(s), cfg["inference_batch_size"]):
                end = start + cfg["inference_batch_size"]
                idx = neighbours[start:end]
                rel = s[idx] - s[start:end, None]
                if idx.shape[1] == 0:
                    parts.append(torch.zeros(len(idx), cfg["d_niche"]))
                else:
                    parts.append(enc.aggregate(e[start:end].to(device), e[idx].to(device),
                                               rel.to(device), dists[start:end].to(device)).cpu())
            section["n"] = torch.cat(parts)
    return enc, pred


def train_flow(sections, triplets, centers, rare_protos, major_protos, niche_pred, cfg, device):
    print("[Phase 3/4] Training RareFlowNet", flush=True)
    train_ids = sorted({i for triplet in triplets for i in triplet})
    reference = np.concatenate([sections[i]["s"].numpy() for i in train_ids])
    tiles = build_adaptive_quadtree_tiles(reference, cfg["max_cells_per_tile"] * len(train_ids))
    pool, weights = [], []
    for a, m, b in triplets:
        for tile in tiles:
            mask = crop_mask(sections[m], tile)
            if mask.sum() >= 12:
                pool.append((a, m, b, tile))
                weights.append(3.0 if np.isin(sections[m]["p"][mask].numpy(), list(rare_protos)).any() else 1.0)
    if not pool:
        raise ValueError("No tiles contain enough training cells; raise max_cells_per_tile or max_cells_per_section")
    probs = np.asarray(weights, dtype=float); probs /= probs.sum()
    flow = RareFlowNet(cfg["d_cell"]).to(device)
    opt = torch.optim.AdamW(flow.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    centers_t = torch.as_tensor(centers, device=device)
    majority = centers_t[sorted(major_protos)]
    steps = min(len(pool), cfg["max_fm_steps_per_epoch"])
    for ep in range(cfg["fm_epochs"]):
        flow.train()
        total, success = 0.0, 0
        for chosen in np.random.choice(len(pool), steps, replace=False, p=probs):
            a, m, b, tile = pool[chosen]
            target = sections[m]
            idx = torch.where(crop_mask(target, tile))[0]
            if m in (a, b):
                idx = idx[torch.randperm(len(idx))[:len(idx) // 2]]
            idx = idx[random_rows(len(idx), cfg["max_cells_per_tile"])]
            s1, e1, w1, p1 = [target[key][idx].to(device) for key in ("s", "e", "w", "p")]
            contexts = []
            for k in (a, b):
                contexts.extend(get_context(sections[k], crop_mask(sections[k], tile, cfg["tile_halo"]),
                                            cfg["max_context_cells"], device, idx if k == m else None))
            sa, ea, na, wa, pa, sb, eb, nb, wb, pb = contexts
            if len(sa) + len(sb) == 0:
                continue
            t_rel = (target["z_norm"] - sections[a]["z_norm"]) / (sections[b]["z_norm"] - sections[a]["z_norm"])
            s0, e0, p0 = sample_rare_anchored_prior(*contexts, t_rel, len(s1), centers_t, rare_protos, niche_pred)
            ri, ci = prototype_constrained_ot_coupling(s0, e0, p0, s1, e1, p1, rare_protos)
            s0, e0, s1, e1, w1, p1 = s0[ri], e0[ri], s1[ci], e1[ci], w1[ci], p1[ci]
            ctx_s, ctx_e, ctx_dz = make_flow_context(sa, ea, sb, eb, t_rel, cfg["max_context_cells"])
            tau = torch.sigmoid(torch.randn(1, device=device)).clamp(1e-3, 1 - 1e-3)
            st, et = (1 - tau) * s0 + tau * s1, (1 - tau) * e0 + tau * e1
            qm = torch.ones(1, len(s1), dtype=torch.bool, device=device)
            cm = torch.ones(1, len(ctx_s), dtype=torch.bool, device=device)
            drop = torch.tensor([random.random() < 0.1], device=device)
            vs, ve = flow(tau, tau.new_tensor([t_rel]), st[None], et[None], qm,
                          ctx_s[None], ctx_e[None], ctx_dz[None], cm, drop)
            vs, ve = vs[0], ve[0]
            cfm = (w1 * ((vs - (s1 - s0)).square().mean(-1) + (ve - (e1 - e0)).square().mean(-1))).mean()
            rare = torch.isin(p1, torch.tensor(sorted(rare_protos), device=device))
            anti, topo = cfm.new_zeros(()), cfm.new_zeros(())
            if rare.any() and len(majority):
                predicted = et[rare] + (1 - tau) * ve[rare]
                pos = (predicted - e1[rare]).square().sum(-1)
                neg = torch.cdist(predicted, majority).square().min(-1).values
                anti = tau.item() * F.relu(pos - neg + 0.5).mean()
            if rare.any() and len(s1) > 4:
                predicted = st + (1 - tau) * vs
                topo = F.smooth_l1_loss(torch.cdist(predicted[rare], predicted), torch.cdist(s1[rare], s1))
            loss = cfm + 0.35 * anti + 0.25 * topo
            checked_step(loss, opt, flow.parameters(), 2.0)
            total += loss.item(); success += 1
        if not success:
            raise RuntimeError("Flow training performed no updates; check section overlap")
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"  Flow {ep + 1}/{cfg['fm_epochs']} | loss={total / success:.4f}", flush=True)
    save_checkpoint(cpu_state_dict(flow), Path(cfg["output_dir"]) / "checkpoints" / "flow.pt")
    return flow.eval()
