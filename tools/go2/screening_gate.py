#!/usr/bin/env python3
"""Prevent duplicate Go2 one-factor screening launches."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


BLOCKING_STATUSES = {"tested_fail", "tested_pass", "in_progress"}
RELEVANT_CODE_PATHS = (
    "motrix_envs",
    "motrix_rl",
    "motrix_rl_rslrl",
    "motrixlab",
    "tools/go2",
)


def run_git(args: list[str], repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ledger(path: Path) -> dict:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "training_axis", "records"}
    missing = required - ledger.keys()
    if missing:
        raise ValueError(f"ledger missing keys: {sorted(missing)}")
    return ledger


def fingerprint(
    commit_tree: str,
    task_config_sha256: str,
    seed: int,
    num_envs: int,
    max_iterations: int,
    quick_evaluator_sha256: str,
    full_evaluator_sha256: str,
) -> str:
    canonical = json.dumps(
        {
            "commit_tree": commit_tree,
            "task_config_sha256": task_config_sha256,
            "seed": seed,
            "num_envs": num_envs,
            "max_iterations": max_iterations,
            "quick_evaluator_sha256": quick_evaluator_sha256,
            "full_evaluator_sha256": full_evaluator_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def record_fingerprint(record: dict) -> str:
    return fingerprint(
        record["commit_tree"],
        record["task_config_sha256"],
        record["seed"],
        record["num_envs"],
        record["max_iterations"],
        record["quick_evaluator_sha256"],
        record["full_evaluator_sha256"],
    )


def validate_training_axis(ledger: dict, repo: Path, quick_path: Path, full_path: Path) -> None:
    axis = ledger["training_axis"]
    expected = {
        "task": "go2-walk-flat/rslrl.ppo",
        "seed": 15,
        "num_envs": 4096,
        "max_iterations": 2000,
    }
    actual_task = {key: axis.get(key) for key in expected}
    if actual_task != expected:
        raise ValueError(f"non-screening training axis: {actual_task}")
    actual_evaluators = {
        "quick_evaluator_sha256": sha256_file(quick_path),
        "full_evaluator_sha256": sha256_file(full_path),
    }
    for key, value in actual_evaluators.items():
        if axis.get(key) != value:
            raise ValueError(f"{key} changed: ledger={axis.get(key)} actual={value}")
    if subprocess.run(
        ["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", *RELEVANT_CODE_PATHS],
    ).returncode != 0:
        raise ValueError("relevant tracked code differs from HEAD; commit the candidate first")


def check_command(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    ledger = load_ledger(args.ledger.resolve())
    quick_path = args.quick_evaluator.resolve()
    full_path = args.full_evaluator.resolve()
    validate_training_axis(ledger, repo, quick_path, full_path)

    head = run_git(["rev-parse", "HEAD"], repo)
    tree = run_git(["rev-parse", "HEAD^{tree}"], repo)
    axis = ledger["training_axis"]
    current = fingerprint(
        tree,
        axis["task_config_sha256"],
        axis["seed"],
        axis["num_envs"],
        axis["max_iterations"],
        axis["quick_evaluator_sha256"],
        axis["full_evaluator_sha256"],
    )
    same_name = [record for record in ledger["records"] if record["candidate"] == args.candidate]
    if same_name:
        print(
            json.dumps(
                {
                    "decision": "reject",
                    "reason": "candidate_name_already_registered",
                    "candidate": args.candidate,
                    "records": same_name,
                },
                indent=2,
            )
        )
        return 2

    matches = [record for record in ledger["records"] if record_fingerprint(record) == current]
    blocking = [record for record in matches if record.get("status") in BLOCKING_STATUSES]
    if blocking:
        print(
            json.dumps(
                {
                    "decision": "reject",
                    "reason": "fingerprint_already_tested",
                    "candidate": args.candidate,
                    "head": head,
                    "commit_tree": tree,
                    "fingerprint": current,
                    "matches": matches,
                    "blocking_matches": blocking,
                },
                indent=2,
            )
        )
        return 2
    if matches:
        print(
            json.dumps(
                {
                    "decision": "error",
                    "reason": "fingerprint_has_no_blocking_record",
                    "candidate": args.candidate,
                    "head": head,
                    "fingerprint": current,
                    "matches": matches,
                },
                indent=2,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "decision": "allow",
                "reason": "ledger_miss",
                "candidate": args.candidate,
                "head": head,
                "commit_tree": tree,
                "task_config_sha256": axis["task_config_sha256"],
                "seed": axis["seed"],
                "num_envs": axis["num_envs"],
                "max_iterations": axis["max_iterations"],
                "quick_evaluator_sha256": axis["quick_evaluator_sha256"],
                "full_evaluator_sha256": axis["full_evaluator_sha256"],
                "fingerprint": current,
            },
            indent=2,
        )
    )
    return 0


def register_command(args: argparse.Namespace) -> int:
    ledger_path = args.ledger.resolve()
    ledger = load_ledger(ledger_path)
    axis = ledger["training_axis"]
    records = ledger["records"]
    existing_names = {record["candidate"] for record in records}
    new_records = []
    for row in args.mapping.read_text(encoding="utf-8").splitlines():
        if not row.strip() or row.startswith("#"):
            continue
        candidate, commit, status, run_dir = row.split("\t", 3)
        if candidate in existing_names:
            raise ValueError(f"duplicate candidate name in ledger: {candidate}")
        record = {
            "candidate": candidate,
            "commit": commit,
            "commit_tree": run_git(["rev-parse", f"{commit}^{{tree}}"], args.repo.resolve()),
            "task_config_sha256": axis["task_config_sha256"],
            "seed": axis["seed"],
            "num_envs": axis["num_envs"],
            "max_iterations": axis["max_iterations"],
            "quick_evaluator_sha256": axis["quick_evaluator_sha256"],
            "full_evaluator_sha256": axis["full_evaluator_sha256"],
            "status": status,
            "run_dir": run_dir,
        }
        records.append(record)
        new_records.append(record)
        existing_names.add(candidate)
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"registered": new_records}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="check a committed candidate before launch")
    check.add_argument("candidate")
    check.add_argument("--ledger", type=Path, default=Path("tools/go2/tested_factor_ledger.json"))
    check.add_argument("--repo", type=Path, default=Path("."))
    check.add_argument("--quick-evaluator", type=Path, default=Path("runs/audits/go2_checkpoint_audit.py"))
    check.add_argument("--full-evaluator", type=Path, default=Path("runs/audits/go2_v197_audit.py"))
    check.set_defaults(func=check_command)

    register = subparsers.add_parser("register", help="batch backfill candidates from TSV")
    register.add_argument("mapping", type=Path, help="TSV: candidate, commit, status, run_dir")
    register.add_argument("--ledger", type=Path, default=Path("tools/go2/tested_factor_ledger.json"))
    register.add_argument("--repo", type=Path, default=Path("."))
    register.set_defaults(func=register_command)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(arguments.func(arguments))
