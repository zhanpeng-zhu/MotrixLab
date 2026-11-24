from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditItem:
    iteration: int
    checkpoint: Path
    staging: Path
    artifact: Path
    audit_output: Path
    zero_hold_output: Path
    speed_friction_output: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and audit Go2 checkpoints after a completed training run."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("label", help="Artifact label prefix, such as go2-swing-tight")
    parser.add_argument(
        "--start",
        type=int,
        default=800,
        help="First checkpoint iteration to audit (inclusive)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=1999,
        help="Last checkpoint iteration to audit (inclusive when present in step interval)",
    )
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of checkpoints to export and audit concurrently",
    )
    parser.add_argument("--zero-hold", action="store_true", help="Also run 60s zero-command hold")
    parser.add_argument(
        "--speed-friction-matrix",
        action="store_true",
        help="Also run the 0.3/0.5/0.8/1.0 speed by 0.6/0.8/1.0/1.2/1.4 friction matrix",
    )
    args = parser.parse_args()
    if args.step <= 0 or args.start <= 0 or args.end < args.start or args.jobs <= 0:
        raise ValueError("require 0 < start <= end and positive step/jobs")
    return args


def iterations(args: argparse.Namespace) -> list[int]:
    values = list(range(args.start, args.end + 1, args.step))
    if not values or values[-1] != args.end:
        values.append(args.end)
    return values


def build_items(args: argparse.Namespace) -> list[AuditItem]:
    items: list[AuditItem] = []
    for iteration in iterations(args):
        checkpoint = args.run_dir / f"model_{iteration}.pt"
        stem = f"{args.label}-it{iteration}"
        items.append(
            AuditItem(
                iteration=iteration,
                checkpoint=checkpoint,
                staging=Path("runs/checkpoint_exports") / stem,
                artifact=Path("artifacts") / f"{stem}.deploy",
                audit_output=Path("runs/audits") / f"{stem}-final.json",
                zero_hold_output=Path("runs/audits") / f"{stem}-zero-hold-60s.json",
                speed_friction_output=Path("runs/audits") / f"{stem}-speed-friction-matrix.json",
            )
        )
    missing = [str(item.checkpoint) for item in items if not item.checkpoint.is_file()]
    if missing:
        raise FileNotFoundError("missing checkpoints: " + ", ".join(missing))
    return items


def run(command: list[str]) -> None:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(command, check=False, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")


def audit_item(item: AuditItem, run_dir: Path, zero_hold: bool, speed_friction_matrix: bool) -> None:
    if not item.artifact.exists():
        if item.staging.exists():
            raise FileExistsError(f"refusing to reuse staging path: {item.staging}")
        run(
            [
                sys.executable,
                "tools/go2/export_checkpoint.py",
                str(run_dir),
                str(item.checkpoint),
                str(item.artifact),
                "--label",
                item.artifact.stem,
            ]
        )
        run([".venv/bin/mxlab", "export-deploy", f"run_dir={item.staging}", f"output={item.artifact}"])
    if not item.audit_output.exists():
        run(
            [
                "env",
                "PYTHONPATH=runs/audits",
                sys.executable,
                "runs/audits/go2_checkpoint_audit.py",
                str(item.artifact),
                str(item.audit_output),
            ]
        )
    if zero_hold and not item.zero_hold_output.exists():
        run(
            [
                "env",
                "PYTHONPATH=runs/audits",
                sys.executable,
                "runs/audits/go2_zero_hold_60s_audit.py",
                str(item.artifact),
                str(item.zero_hold_output),
            ]
        )
    if speed_friction_matrix and not item.speed_friction_output.exists():
        run(
            [
                "env",
                "PYTHONPATH=runs/audits",
                sys.executable,
                "tools/go2/speed_friction_audit.py",
                str(item.artifact),
                str(item.speed_friction_output),
            ]
        )


def main() -> int:
    args = parse_args()
    if not (args.run_dir / "metadata.json").is_file():
        raise FileNotFoundError(f"run metadata not found: {args.run_dir / 'metadata.json'}")

    items = build_items(args)
    if args.jobs == 1:
        for item in items:
            print(f"[audit] iteration {item.iteration}", flush=True)
            audit_item(item, args.run_dir, args.zero_hold, args.speed_friction_matrix)
    else:
        print(
            f"[audit] running {len(items)} checkpoints with {args.jobs} workers",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(
                    audit_item,
                    item,
                    args.run_dir,
                    args.zero_hold,
                    args.speed_friction_matrix,
                ): item
                for item in items
            }
            failures: list[str] = []
            for future in as_completed(futures):
                item = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failure = f"iteration {item.iteration}: {exc}"
                    failures.append(failure)
                    print(f"[audit] FAILED {failure}", flush=True)
                else:
                    print(f"[audit] completed iteration {item.iteration}", flush=True)
            if failures:
                raise RuntimeError("audit failures: " + "; ".join(failures))
    print(f"[audit] completed {len(items)} checkpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
