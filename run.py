"""CLI alternative to run.ipynb; use physical GPU 1 for CUDA training."""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Rare3D-FM local Zhuang-ABCA-2 reconstruction")
    parser.add_argument("--config", help="JSON configuration overrides")
    parser.add_argument("--data-dir", help="Data root or Zhuang-ABCA-2 section directory")
    parser.add_argument("--output-dir", help="Output directory (relative to the current directory)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--inspect", action="store_true", help="Print dataset metadata without training")
    parser.add_argument("--smoke-test", action="store_true", help="Run all stages on 3 sections / 256 cells / 1 epoch")
    parser.add_argument("--skip-export", action="store_true", help="Train and validate without exporting the volume")
    args = parser.parse_args()
    # Set visibility before config imports PyTorch or initializes CUDA.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from rare3d.config import load_config, smoke_config, validate_config
    cfg = load_config(args.config)
    if args.smoke_test:
        smoke_config(cfg)
        cfg["output_dir"] = "outputs/smoke"
    for key in ("data_dir", "output_dir", "device"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.skip_export:
        cfg["export_reconstruction"] = False
    validate_config(cfg)
    if args.inspect:
        from rare3d.data import find_h5ad_files, select_files, inspect_dataset
        rows = inspect_dataset(select_files(find_h5ad_files(cfg["data_dir"]), cfg["max_sections"]))
        print(json.dumps({"sections": len(rows), "cells": sum(r["cells"] for r in rows), "files": rows},
                         indent=2, ensure_ascii=False))
        return
    from rare3d.pipeline import run
    run(cfg)


if __name__ == "__main__":
    main()
