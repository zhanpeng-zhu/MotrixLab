from __future__ import annotations

import itertools
import json
from pathlib import Path

import go2_v197_audit as full_audit


SPEEDS = (0.3, 0.5, 0.8, 1.0)
FRICTIONS = (0.6, 0.8, 1.0, 1.2, 1.4)


def token(value: float) -> str:
    return str(value).replace(".", "p")


def axis_case(axis: int, speed: float, friction: float) -> dict:
    axis_name = ("forward_reverse", "lateral_both", "yaw_both")[axis]
    # Lateral is capped at the task's supported 0.5 m/s envelope.
    command_speed = min(speed, 0.5) if axis == 1 else speed
    positive = [0.0, 0.0, 0.0]
    negative = [0.0, 0.0, 0.0]
    positive[axis] = command_speed
    negative[axis] = -command_speed
    return {
        "name": f"{axis_name}_v{token(speed)}_mu{token(friction)}",
        "seconds": 12.0,
        "floor_friction": friction,
        "schedule": [
            (0.0, tuple(positive)),
            (5.0, (0.0, 0.0, 0.0)),
            (7.0, tuple(negative)),
        ],
        "segments": [
            {"name": "positive_3s_to_5s", "start_s": 3.0, "end_s": 5.0, "target": positive},
            {"name": "negative_10s_to_12s", "start_s": 10.0, "end_s": 12.0, "target": negative},
        ],
    }


def corner_case(signs: tuple[int, int, int], speed: float, friction: float) -> dict:
    target = [sign * speed for sign in signs]
    target[1] = signs[1] * min(speed, 0.5)
    sign_token = "".join("p" if sign > 0 else "n" for sign in signs)
    return {
        "name": f"corner_{sign_token}_v{token(speed)}_mu{token(friction)}",
        "seconds": 6.0,
        "floor_friction": friction,
        "schedule": [(0.0, tuple(target))],
        "segments": [
            {"name": "steady_3s_to_6s", "start_s": 3.0, "end_s": 6.0, "target": target}
        ],
    }


def cases() -> list[dict]:
    result = []
    for friction, speed in itertools.product(FRICTIONS, SPEEDS):
        result.extend(axis_case(axis, speed, friction) for axis in range(3))
        result.extend(
            corner_case(signs, speed, friction)
            for signs in itertools.product((-1, 1), repeat=3)
        )
    return result


def main() -> int:
    output = full_audit.OUTPUT
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    partial = Path(f"{output}.partial")
    reports = json.loads(partial.read_text(encoding="utf-8")) if partial.exists() else []
    completed = {report["name"] for report in reports}
    audit_cases = cases()
    for index, case in enumerate(audit_cases, start=1):
        if case["name"] in completed:
            continue
        report = full_audit.run_case(case)
        reports.append(report)
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        compact = {
            "case": f"{index}/{len(audit_cases)}",
            "name": report["name"],
            "success": report["success"],
            "exit": report["exit_reason"],
            "roll": report["max_abs_roll_rad"],
            "pitch": report["max_abs_pitch_rad"],
            "min_up_z": report["min_up_z"],
            "forward_shift_p95": report["base_forward_shift_abs_p95_m"],
            "swing_mean": report["foot_mean_swing_height_m"],
            "touchdown_p95": report["foot_touchdown_downward_speed_abs_p95_mps"],
        }
        print(json.dumps(compact, ensure_ascii=False), flush=True)

    partial.replace(output)
    print(json.dumps({"output": str(output), "cases": len(reports)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
