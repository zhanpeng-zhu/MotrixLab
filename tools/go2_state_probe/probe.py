# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Bounded, read-only Unitree Go2 ``rt/lowstate`` probe.

This module deliberately creates only a DDS subscriber.  It never imports or
constructs a publisher, never imports LowCmd, and never contacts a service that
changes robot mode.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO


JOINT_NAMES = (
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
)
FOOT_NAMES = ("FL", "FR", "RL", "RR")
DEFAULT_TOPIC = "rt/lowstate"
DEFAULT_DURATION_S = 5.0


@dataclass
class ProbeConfig:
    network_interface: str
    domain_id: int = 0
    topic: str = DEFAULT_TOPIC
    duration_s: float = DEFAULT_DURATION_S
    print_interval_s: float = 1.0
    queue_depth: int = 16
    validate_crc: bool = True
    output: Path | None = None
    append: bool = False


@dataclass
class ProbeCounters:
    received: int = 0
    valid: int = 0
    crc_errors: int = 0
    decode_errors: int = 0


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def decode_state(message: Any, received_at_ns: int) -> dict[str, Any]:
    """Return one serializable state sample, raising on invalid payloads."""
    if len(message.motor_state) < 12:
        raise ValueError(f"expected at least 12 motor states, got {len(message.motor_state)}")
    if len(message.foot_force) < 4 or len(message.foot_force_est) < 4:
        raise ValueError("expected four foot-force samples")

    quaternion_wxyz = [_finite_float(value, "quaternion") for value in message.imu_state.quaternion]
    gyroscope = [_finite_float(value, "gyroscope") for value in message.imu_state.gyroscope]
    accelerometer = [_finite_float(value, "accelerometer") for value in message.imu_state.accelerometer]
    rpy = [_finite_float(value, "rpy") for value in message.imu_state.rpy]
    if len(quaternion_wxyz) != 4 or len(gyroscope) != 3 or len(accelerometer) != 3 or len(rpy) != 3:
        raise ValueError("invalid IMU field shape")

    joints = []
    for name, motor in zip(JOINT_NAMES, message.motor_state[:12], strict=True):
        joints.append(
            {
                "name": name,
                "position_rad": _finite_float(motor.q, "joint position"),
                "velocity_rad_s": _finite_float(motor.dq, "joint velocity"),
                "torque_est_n_m": _finite_float(motor.tau_est, "joint torque estimate"),
                "temperature_c": int(motor.temperature),
                "lost": int(motor.lost),
            }
        )

    return {
        "received_at_ns": received_at_ns,
        "robot_tick": int(message.tick),
        "imu": {
            "quaternion_wxyz": quaternion_wxyz,
            "quaternion_xyzw": quaternion_wxyz[1:] + quaternion_wxyz[:1],
            "gyroscope_rad_s": gyroscope,
            "accelerometer_m_s2": accelerometer,
            "rpy_rad": rpy,
            "temperature_c": int(message.imu_state.temperature),
        },
        "joints": joints,
        "feet": {
            name: {"force_raw": int(force), "force_est_raw": int(force_est)}
            for name, force, force_est in zip(
                FOOT_NAMES,
                message.foot_force[:4],
                message.foot_force_est[:4],
                strict=True,
            )
        },
        "power": {
            "voltage_v": _finite_float(message.power_v, "battery voltage"),
            "current_a": _finite_float(message.power_a, "battery current"),
        },
    }


class LowStateProbe:
    """Validate LowState frames and record periodic human-readable summaries."""

    def __init__(
        self,
        crc: Any,
        *,
        clock_ns=lambda: time.monotonic_ns(),
        print_interval_s: float,
        validate_crc: bool = True,
        stream: TextIO | None = None,
        history_size: int = 128,
    ) -> None:
        self.crc = crc
        self.clock_ns = clock_ns
        self.print_interval_ns = round(print_interval_s * 1e9)
        self.validate_crc = validate_crc
        self.stream = stream if stream is not None else sys.stdout
        self.counters = ProbeCounters()
        self.samples: list[dict[str, Any]] = []
        self.receive_times_ns: deque[int] = deque(maxlen=history_size)
        self.robot_ticks: deque[int] = deque(maxlen=history_size)
        self._last_print_ns: int | None = None

    @staticmethod
    def frequency_hz(intervals_s: list[float]) -> float | None:
        if len(intervals_s) < 2:
            return None
        elapsed_s = sum(intervals_s)
        if elapsed_s <= 0.0:
            return None
        return len(intervals_s) / elapsed_s

    def statistics(self) -> dict[str, Any]:
        local = list(self.receive_times_ns)
        intervals_s = [(right - left) / 1e9 for left, right in zip(local, local[1:])]
        wall_hz = self.frequency_hz(intervals_s)
        ticks = list(self.robot_ticks)
        tick_deltas = [right - left for left, right in zip(ticks, ticks[1:])]
        if wall_hz is not None and tick_deltas:
            seconds_per_tick = sum(intervals_s) / sum(tick_deltas) if sum(tick_deltas) else 0.0
            tick_hz = wall_hz if not seconds_per_tick else 1.0 / seconds_per_tick
        else:
            tick_hz = None
        return {
            **asdict(self.counters),
            "wall_frequency_hz": wall_hz,
            "robot_tick_frequency_hz": tick_hz,
            "last_robot_tick": ticks[-1] if ticks else None,
        }

    def __call__(self, message: Any) -> None:
        received_at_ns = self.clock_ns()
        self.counters.received += 1
        self.receive_times_ns.append(received_at_ns)
        self.robot_ticks.append(int(message.tick))
        try:
            if self.validate_crc and self.crc.Crc(message) != message.crc:
                self.counters.crc_errors += 1
                return
            sample = decode_state(message, received_at_ns)
        except Exception:
            self.counters.decode_errors += 1
            return

        self.counters.valid += 1
        self.samples.append(sample)
        if self._last_print_ns is not None and received_at_ns - self._last_print_ns < self.print_interval_ns:
            return
        self._last_print_ns = received_at_ns
        self.print_summary(sample)

    def print_summary(self, sample: dict[str, Any]) -> None:
        stats = self.statistics()
        imu = sample["imu"]
        print(
            f"t={sample['received_at_ns']} tick={sample['robot_tick']} "
            f"frames={stats['received']} valid={stats['valid']} "
            f"wall_hz={stats['wall_frequency_hz']} tick_hz={stats['robot_tick_frequency_hz']}",
            file=self.stream,
        )
        print(
            f"imu rpy={imu['rpy_rad']} gyro={imu['gyroscope_rad_s']} accel={imu['accelerometer_m_s2']}",
            file=self.stream,
        )
        print(
            "joints q rad: " + ", ".join(f"{joint['name']}={joint['position_rad']:.4f}" for joint in sample["joints"]),
            file=self.stream,
        )
        print(
            "joints dq rad/s: "
            + ", ".join(f"{joint['name']}={joint['velocity_rad_s']:.4f}" for joint in sample["joints"]),
            file=self.stream,
        )
        print(
            "feet raw: "
            + ", ".join(f"{name}={data['force_raw']}/{data['force_est_raw']}" for name, data in sample["feet"].items()),
            file=self.stream,
        )
        self.stream.flush()


def load_sdk() -> tuple[Any, Any, Any, Any]:
    """Import the official Python SDK without touching any SDK source file."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    from unitree_sdk2py.utils.crc import CRC

    return ChannelFactoryInitialize, ChannelSubscriber, LowState_, CRC


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Unitree Go2 rt/lowstate probe")
    parser.add_argument("network_interface", help="interface connected to Go2, for example enp5s0")
    parser.add_argument("--domain-id", type=_non_negative_int, default=0)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--duration-s", type=_positive_float, default=DEFAULT_DURATION_S)
    parser.add_argument("--print-interval-s", type=_positive_float, default=1.0)
    parser.add_argument("--queue-depth", type=_positive_int, default=16)
    parser.add_argument("--no-validate-crc", action="store_true")
    parser.add_argument("--output", type=Path, help="write JSON records to this path")
    parser.add_argument("--append", action="store_true", help="append records instead of replacing the output")
    return parser


def config_from_args(args: argparse.Namespace) -> ProbeConfig:
    return ProbeConfig(
        network_interface=args.network_interface,
        domain_id=args.domain_id,
        topic=args.topic,
        duration_s=args.duration_s,
        print_interval_s=args.print_interval_s,
        queue_depth=args.queue_depth,
        validate_crc=not args.no_validate_crc,
        output=args.output,
        append=args.append,
    )


def run(config: ProbeConfig, *, sdk: tuple[Any, Any, Any, Any] | None = None, stream: TextIO | None = None) -> int:
    try:
        initialize, subscriber_type, low_state_type, crc_type = sdk if sdk is not None else load_sdk()
    except ImportError as error:
        raise RuntimeError(
            "unitree_sdk2py is not installed; install the official Unitree Python SDK wheel"
        ) from error

    output_stream = stream or sys.stdout
    print(
        f"read-only probe: interface={config.network_interface} domain={config.domain_id} "
        f"topic={config.topic} duration={config.duration_s:g}s crc={config.validate_crc}",
        file=output_stream,
    )
    print(
        "safety: subscriber only; no rt/lowcmd publisher, no mode switch, no motor command",
        file=output_stream,
    )

    subscriber: Any = None
    probe: LowStateProbe | None = None
    interrupted = False
    try:
        initialize(config.domain_id, config.network_interface)
        probe = LowStateProbe(
            crc_type(),
            print_interval_s=config.print_interval_s,
            validate_crc=config.validate_crc,
            stream=output_stream,
        )
        subscriber = subscriber_type(config.topic, low_state_type)
        subscriber.Init(probe, config.queue_depth)
        deadline_ns = time.monotonic_ns() + round(config.duration_s * 1e9)
        while time.monotonic_ns() < deadline_ns:
            time.sleep(0.1)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if subscriber is not None:
            close = getattr(subscriber, "Close", None) or getattr(subscriber, "close", None)
            if close is not None:
                close()

    summary = probe.statistics() if probe is not None else {}
    if interrupted:
        summary["interrupted"] = True
    result = {
        "config": {
            "network_interface": config.network_interface,
            "domain_id": config.domain_id,
            "topic": config.topic,
            "duration_s": config.duration_s,
            "validate_crc": config.validate_crc,
            "safety": "subscriber-only",
        },
        "statistics": summary,
        "records": probe.samples if probe is not None else [],
    }
    if config.output is not None:
        mode = "a" if config.append else "w"
        config.output.parent.mkdir(parents=True, exist_ok=True)
        with config.output.open(mode, encoding="utf-8") as output_file:
            json.dump(result, output_file, ensure_ascii=False, separators=(",", ":"))
            output_file.write("\n")
    print("summary=" + json.dumps(summary, ensure_ascii=False, sort_keys=True), file=output_stream)
    return 0 if summary.get("valid", 0) > 0 else 1


def main(argv: list[str] | None = None) -> int:
    config = config_from_args(build_parser().parse_args(argv))
    try:
        return run(config)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
