"""Sparse expression evaluation and file-based comparison plots."""
import os
import tempfile
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hmst-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "hmst-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score, f1_score, recall_score

def evaluate_rare_recovery(
    s_pred, e_pred, x_pred, s_true, e_true, x_true,
    centers, rare_protos, maj_protos, gene_weights,
    true_bio_labels: Optional[np.ndarray] = None, save_fig_path: Optional[str] = None
):
    """低内存评估无监督原型 + 真实 ABCA 稀有亚群恢复情况，并输出可视化图"""
    if len(s_pred) == 0:
        print("[Validation] No cells survived tile boundaries; metrics unavailable", flush=True)
        return {"status": "empty_reconstruction", "true_cells": len(s_true), "predicted_cells": 0}
    nn_c = NearestNeighbors(n_neighbors=1).fit(centers)
    p_true = nn_c.kneighbors(e_true, return_distance=False)[:, 0]
    p_pred = nn_c.kneighbors(e_pred, return_distance=False)[:, 0]
    true_is_rare = np.isin(p_true, list(rare_protos))
    pred_is_rare = np.isin(p_pred, list(rare_protos))

    # 空间近邻对齐 (O(N log N) 内存安全)
    nn_sp = NearestNeighbors(n_neighbors=1).fit(s_pred)
    nn_pred_for_true = nn_sp.kneighbors(s_true, return_distance=False)[:, 0]
    rare_recall = recall_score(true_is_rare, pred_is_rare[nn_pred_for_true], zero_division=0)
    rare_f1 = f1_score(true_is_rare, pred_is_rare[nn_pred_for_true], zero_division=0)

    # 抗嵌合体纯度边际
    if pred_is_rare.sum() > 0 and len(maj_protos) > 0:
        nn_r = NearestNeighbors(n_neighbors=1).fit(centers[list(rare_protos)])
        nn_m = NearestNeighbors(n_neighbors=1).fit(centers[list(maj_protos)])
        d_r, _ = nn_r.kneighbors(e_pred[pred_is_rare])
        d_m, _ = nn_m.kneighbors(e_pred[pred_is_rare])
        margin = float(np.mean(d_m[:, 0] - d_r[:, 0]))
    else:
        margin = 0.0

    # 稀疏 Marker AUPRC
    marker_idx = np.argsort(-gene_weights)[:min(30, len(gene_weights))]
    auprcs = []
    if len(s_pred) > 0:
        x_t_sub, x_p_sub = x_true[:, marker_idx], x_pred[nn_pred_for_true][:, marker_idx]
        if sp.issparse(x_t_sub):
            x_t_sub = x_t_sub.toarray()
        if sp.issparse(x_p_sub):
            x_p_sub = x_p_sub.toarray()
        thresh = np.percentile(x_t_sub, 85, axis=0)
        for g in range(x_t_sub.shape[1]):
            yb = (x_t_sub[:, g] > thresh[g]).astype(int)
            if 0 < yb.sum() < len(yb):
                auprcs.append(average_precision_score(yb, x_p_sub[:, g]))
    marker_auprc = float(np.mean(auprcs)) if auprcs else 0.0

    # 生态位共定位误差
    def niche_nn(s_arr, mask):
        if mask.sum() == 0 or (~mask).sum() == 0:
            return 0.0
        d, _ = NearestNeighbors(n_neighbors=1).fit(s_arr[~mask]).kneighbors(s_arr[mask])
        return float(d.mean())
    niche_err = abs(niche_nn(s_true, true_is_rare) - niche_nn(s_pred, pred_is_rare))

    print("\n" + "=" * 65)
    print("         Rare3D-FM 验证切片稀有细胞恢复专项报告")
    print("=" * 65)
    print(f"• 无监督稀有占比保真度   : 真实={true_is_rare.mean()*100:.2f}% | 重建={pred_is_rare.mean()*100:.2f}%")
    print(f"• 空间+表型双重匹配指标  : Recall={rare_recall:.4f} | F1={rare_f1:.4f}")
    print(f"• 抗嵌合体纯度边际       : Margin={margin:.4f} (>0 代表未混合为多数类)")
    print(f"• 稀有 Marker 基因 AUPRC : {marker_auprc:.4f}")
    print(f"• 生态位共定位距离误差   : {niche_err:.4f}")

    # 若数据包含 ABCA 真实标签，额外计算生物学稀有亚群 (<1%) 恢复率
    if true_bio_labels is not None:
        vc = pd.Series(true_bio_labels).value_counts(normalize=True)
        bio_rare_types = set(vc[vc < 0.01].index)
        if bio_rare_types:
            bio_rare_mask = np.isin(true_bio_labels, list(bio_rare_types))
            # 检查这些真实生物学稀有细胞位置处，有多少被模型识别并重建为稀有原型（或匹配近邻表达相关性）
            bio_rec_rate = float(pred_is_rare[nn_pred_for_true][bio_rare_mask].mean())
            print(f"• ABCA真实低丰度亚型(<1%): 共{len(bio_rare_types)}类({bio_rare_mask.sum()}细胞), 稀有表型检出率={bio_rec_rate:.4f}")
    print("=" * 65 + "\n")

    if save_fig_path:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150)
        for ax, s_arr, r_mask, title in [
            (axes[0], s_true, true_is_rare, f"Ground Truth Slice (Rare: {true_is_rare.sum()})"),
            (axes[1], s_pred, pred_is_rare, f"Rare3D-FM Reconstructed (Rare: {pred_is_rare.sum()})"),
        ]:
            ax.scatter(s_arr[~r_mask, 0], s_arr[~r_mask, 1], s=1.0, c="#D3D3D3", alpha=0.5, label="Majority")
            if r_mask.any():
                ax.scatter(s_arr[r_mask, 0], s_arr[r_mask, 1], s=7.0, c="#E63946", alpha=0.9, label="Rare Cells")
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.invert_yaxis()  # 匹配鼠脑 CCF 背腹方向
            ax.legend(loc="upper right", markerscale=2)
        plt.tight_layout()
        plt.savefig(save_fig_path)
        plt.close(fig)
        print(f"[Info] Comparison plot saved to: {save_fig_path}")

    return {"status": "ok", "true_cells": len(s_true), "predicted_cells": len(s_pred),
            "true_rare_fraction": float(true_is_rare.mean()),
            "predicted_rare_fraction": float(pred_is_rare.mean()),
            "rare_recall": float(rare_recall), "rare_f1": float(rare_f1),
            "purity_margin": margin, "marker_auprc": marker_auprc,
            "niche_distance_error": float(niche_err)}
