from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one Go2 training checkpoint as a deploy artifact.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--label", required=True, help="Unique staging label such as go2-v197-it500")
    parser.add_argument("--staging-root", type=Path, default=Path("runs/checkpoint_exports"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (args.run_dir / "metadata.json").is_file():
        raise FileNotFoundError(f"run metadata not found: {args.run_dir / 'metadata.json'}")
    if not (args.run_dir / "task_config.yaml").is_file():
        raise FileNotFoundError(f"task config not found: {args.run_dir / 'task_config.yaml'}")
    if not args.checkpoint.is_file() or args.checkpoint.suffix != ".pt":
        raise FileNotFoundError(f"checkpoint must be an existing .pt file: {args.checkpoint}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.output}")

    staging = args.staging_root / args.label
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite existing staging run: {staging}")
    checkpoint_dir = staging / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    shutil.copy2(args.run_dir / "metadata.json", staging / "metadata.json")
    shutil.copy2(args.run_dir / "task_config.yaml", staging / "task_config.yaml")
    shutil.copy2(args.checkpoint, checkpoint_dir / "latest.pt")
    manifest = {
        "version": 1,
        "artifacts": {
            "best_policy": {"path": "latest.pt", "kind": "policy", "format": "pt"},
            "latest_training_state": {"path": "latest.pt", "kind": "training_state", "format": "pt"},
        },
    }
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"staging": str(staging), "checkpoint": str(args.checkpoint)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
