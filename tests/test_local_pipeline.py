"""Regression checks for the local migration (stdlib unittest)."""
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from rare3d.config import load_config, smoke_config
from rare3d.data import find_h5ad_files, load_zhuang_sections
from rare3d.models import chunked_knn
from rare3d.reconstruction import heun_ode_reconstruct, export_volume
from rare3d.training import build_train_triplets
from rare3d.transport import sample_rare_anchored_prior


class LocalMigrationTests(unittest.TestCase):
    def test_knn_removes_self_with_duplicate_positions(self):
        xy = torch.tensor([[0., 0.], [0., 0.], [1., 0.], [2., 0.]])
        distances, indices = chunked_knn(xy, xy, 2, exclude_self=True)
        self.assertFalse((indices == torch.arange(4)[:, None]).any())
        expected = torch.cdist(xy, xy)
        expected.fill_diagonal_(float("inf"))
        torch.testing.assert_close(distances, expected.sort().values[:, :2])

    def test_physical_depth_and_gene_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, depth, genes in [("sec_10", 3.7, ["b", "a"]), ("sec_2", 0.2, ["a", "b"])]:
                obs = pd.DataFrame({"x": [1., 2., 3.], "y": [3., 1., 2.], "z": depth}, index=["c1", "c2", "c3"])
                a = ad.AnnData(X=np.array([[0, 1], [2, 0], [0, 0]], dtype=np.float32),
                               obs=obs, var=pd.DataFrame(index=genes))
                a.obsm["spatial"] = obs[["x", "y"]].to_numpy()
                a.write_h5ad(Path(directory) / f"{name}.h5ad")
            files = find_h5ad_files(directory)
            sections, genes, is_count, axis, _ = load_zhuang_sections(files, load_config())
            self.assertEqual(axis, 2)
            self.assertEqual([s["z_val"] for s in sections], [float(np.float32(.2)), float(np.float32(3.7))])
            self.assertEqual(genes.tolist(), ["a", "b"])
            self.assertTrue(is_count)
            np.testing.assert_array_equal(sections[1]["X_csr"].toarray()[0], [1, 0])

    def test_validation_never_enters_training_triplets(self):
        for count in range(2, 9):
            validation = count // 2 if count >= 3 else -1
            triplets = build_train_triplets(count, validation)
            self.assertTrue(triplets)
            self.assertTrue(all(validation not in t and t[0] < t[2] for t in triplets))

    def test_heun_integrates_final_step(self):
        def velocity(tau, t_rel, s, e, *args):
            return torch.ones_like(s) * tau[:, None, None], torch.ones_like(e) * tau[:, None, None]
        s, e = torch.zeros(3, 2), torch.zeros(3, 4)
        result_s, result_e = heun_ode_reconstruct(velocity, .5, s, e, torch.zeros(3, dtype=torch.long),
                                                 s, e, torch.zeros(3, 1), set(), n_steps=2)
        torch.testing.assert_close(result_s, torch.full_like(s, .5))
        torch.testing.assert_close(result_e, torch.full_like(e, .5))

    def test_one_sided_prior_keeps_local_anchors(self):
        sa, ea, na = torch.tensor([[.8, .7]]), torch.ones(1, 4), torch.ones(1, 2)
        s, e, p = sample_rare_anchored_prior(sa, ea, na, torch.ones(1), torch.zeros(1, dtype=torch.long),
                                            sa[:0], ea[:0], na[:0], torch.ones(0), torch.zeros(0, dtype=torch.long),
                                            .5, 8, ea, {0}, None, sigma_s=0, sigma_e=0)
        torch.testing.assert_close(s, sa.expand(8, -1))
        torch.testing.assert_close(e, ea.expand(8, -1))

    def test_sparse_export_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            sections = [dict(X_csr=sp.csr_matrix([[1, 0], [0, 2]], dtype=np.float32),
                             xy=np.array([[0., 1.], [2., 3.]], dtype=np.float32), z_val=z,
                             file=f"sec_{k}", obs_names=["c1", "c2"], labels=None,
                             coord_names=["x", "y", "z"]) for k, z in enumerate([.2, 3.7])]
            def reconstruct(*args):
                return np.array([[1., 2.]], dtype=np.float32), None, sp.csr_matrix([[3, 4]], dtype=np.float32)
            path = export_volume(sections, ["a", "b"], np.zeros(2), 1., 2, [0, 1], reconstruct, directory)
            result = ad.read_h5ad(path)
            self.assertEqual(result.shape, (5, 2))
            np.testing.assert_array_equal(result.X.toarray()[2], [3, 4])
            np.testing.assert_allclose(result.obsm["spatial_3d"][:, 2], [.2, .2, 1.95, 3.7, 3.7])
            self.assertEqual(result.obs["source"].value_counts()["observed"], 4)
            self.assertTrue(result.obs_names.is_unique)

    def test_normalized_two_section_pipeline(self):
        from rare3d.pipeline import run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            rng = np.random.default_rng(7)
            for k in range(2):
                xy = rng.uniform(0, 1, (32, 2))
                obs = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "z": float(k)},
                                   index=[f"c{i}" for i in range(32)])
                x = np.log1p(rng.poisson(2, (32, 6))).astype(np.float32)
                a = ad.AnnData(X=sp.csr_matrix(x), obs=obs,
                               var=pd.DataFrame(index=[f"g{i}" for i in range(6)]))
                a.write_h5ad(data / f"sec_{k}.h5ad")
            cfg = smoke_config(load_config())
            cfg.update(data_dir=str(data), output_dir=str(root / "output"), device="cpu")
            result = run(cfg)
            import json
            summary = json.loads((result / "data_summary.json").read_text())
            self.assertEqual(summary["expression_mode"], "normalized")
            self.assertEqual(summary["validation_index"], -1)
            output = ad.read_h5ad(result / "zhuang_abca2_rare3d_reconstructed.h5ad")
            self.assertEqual(output.n_vars, 6)
            self.assertEqual(output.obs["source"].value_counts()["observed"], 64)
            self.assertTrue(np.isfinite(output.X.data).all())
            from rare3d.checkpoints import load_models
            models, checkpoint = load_models(result / "models.pt", device="cpu")
            for name, model in models.items():
                self.assertFalse(model.training)
                weights = torch.load(result / "checkpoints" / f"{name}.pt",
                                     map_location="cpu", weights_only=True)
                self.assertEqual(weights.keys(), model.state_dict().keys())
                for key, value in model.state_dict().items():
                    self.assertEqual(value.device.type, "cpu")
                    self.assertTrue(torch.isfinite(value).all())
                    torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
                    torch.testing.assert_close(value, checkpoint[name][key], rtol=0, atol=0)
            self.assertFalse(list(result.rglob("*.partial.pt")))
            with torch.no_grad():
                x = torch.ones(2, 6)
                embedding, _ = models["vae"].encode(x, x.sum(-1, keepdim=True))
                self.assertEqual(embedding.shape, (2, cfg["d_cell"]))
                self.assertTrue(torch.isfinite(embedding).all())


if __name__ == "__main__":
    unittest.main()
