# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Bounded, read-only ONNX dry run for the Go2 walking artifact.

This module subscribes to ``rt/lowstate`` and infers policy actions solely for
diagnostics.  It never constructs a command publisher and never converts an
inferred action to a robot command.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, TextIO

import numpy as np

from motrix_deploy.artifact import Artifact, read_artifact
from motrix_deploy.contracts import RobotSpec, RobotState
from motrix_deploy.policy import OnnxPolicy
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy_tasks.tasks.go2_walk import Go2WalkPolicyProcessor
from motrix_env_core.input import PlanarVelocityCommand
from tools.go2_state_probe import probe as lowstate_probe


DEFAULT_ARTIFACT = Path("artifacts/go2-walk-flat-v11-it3000.deploy")
DEFAULT_TOPIC = lowstate_probe.DEFAULT_TOPIC
DEFAULT_DURATION_S = 5.0
SUPPORTED_TASK = "go2_walk/v1"

OBSERVATION_LAYOUT = {
    "base_angular_velocity_body": (0, 3),
    "projected_gravity_body": (3, 3),
    "joint_position_minus_default": (6, 12),
    "joint_velocity": (18, 12),
    "previous_policy_action": (30, 12),
    "planar_velocity_command": (42, 3),
    "feet_phase": (45, 4),
}


@dataclass(frozen=True)
class DryRunConfig:
    artifact: Path
    network_interface: str
    domain_id: int = 0
    topic: str = DEFAULT_TOPIC
    command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    duration_s: float = DEFAULT_DURATION_S
    print_interval_s: float = 1.0
    queue_depth: int = 16
    policy_decimation: int = 25
    validate_crc: bool = True
    state_label: str = "read_only_dry_run"
    output: Path | None = None


@dataclass
class DryRunCounters:
    received: int = 0
    decoded: int = 0
    valid: int = 0
    crc_errors: int = 0
    decode_errors: int = 0
    inferred: int = 0
    inference_errors: int = 0


def load_artifact(path: str | Path) -> Artifact:
    """Read and validate an artifact, including the ONNX payload SHA-256."""
    return read_artifact(path)


def artifact_summary(artifact: Artifact) -> dict[str, Any]:
    manifest = artifact.manifest
    return {
        "schema_version": manifest.schema_version,
        "source": manifest.source.to_dict(),
        "task_name": manifest.task.name,
        "task_config": manifest.task.to_dict()["config"],
        "robot": {
            "base_link_name": manifest.robot.base_link_name,
            "joint_names": list(manifest.robot.joint_names),
            "default_joint_position": manifest.robot.default_joint_position.tolist(),
        },
        "policy": {
            "component_version": manifest.policy.component_version,
            "input": {
                "name": manifest.policy.input.name,
                "shape": list(manifest.policy.input.shape),
                "dtype": manifest.policy.input.dtype,
            },
            "output": {
                "name": manifest.policy.output.name,
                "shape": list(manifest.policy.output.shape),
                "dtype": manifest.policy.output.dtype,
            },
            "sha256": manifest.policy.sha256,
        },
        "control": {
            "period_s": manifest.control.period_s,
            "angular_velocity_frame": manifest.control.angular_velocity_frame,
            "base_orientation": manifest.control.base_orientation,
            "quaternion_order": manifest.control.quaternion_order,
        },
        "observation_layout": {
            name: {"offset": offset, "size": size} for name, (offset, size) in OBSERVATION_LAYOUT.items()
        },
    }


def validate_contract(artifact: Artifact) -> tuple[RobotSpec, Any]:
    """Validate the exact Go2 V11 contract and return robot/spec objects."""
    manifest = artifact.manifest
    if manifest.task.name != SUPPORTED_TASK:
        raise ValueError(f"dry-run supports {SUPPORTED_TASK!r}, got {manifest.task.name!r}")
    if manifest.policy.input.shape != (1, 49):
        raise ValueError(f"expected policy input [1, 49], got {list(manifest.policy.input.shape)}")
    if manifest.policy.output.shape != (1, 12):
        raise ValueError(f"expected policy output [1, 12], got {list(manifest.policy.output.shape)}")
    if manifest.control.angular_velocity_frame != "body":
        raise ValueError("Go2 dry run requires body-frame angular velocity")
    if manifest.control.base_orientation != "body_to_world" or manifest.control.quaternion_order != "xyzw":
        raise ValueError("Go2 dry run requires body-to-world quaternion in xyzw order")
    robot = manifest.robot
    if robot.joint_count != 12:
        raise ValueError(f"expected 12 Go2 joints, got {robot.joint_count}")
    return robot, manifest.task


def make_robot_state(sample: dict[str, Any], robot: RobotSpec) -> RobotState:
    """Map decoded LowState data into canonical artifact joint order."""
    lowstate_by_name = {
        name: joint
        for name, joint in zip(lowstate_probe.JOINT_NAMES, sample["joints"], strict=True)
    }
    positions = []
    velocities = []
    for joint_name in robot.joint_names:
        short_name = joint_name.removesuffix("_joint")
        if short_name not in lowstate_by_name:
            raise ValueError(f"LowState lacks joint required by artifact: {joint_name}")
        joint = lowstate_by_name[short_name]
        positions.append(joint["position_rad"])
        velocities.append(joint["velocity_rad_s"])
    return RobotState(
        sample_time_ns=sample["robot_tick"],
        receive_time_ns=sample["received_at_ns"],
        joint_position=np.asarray(positions, dtype=np.float32),
        joint_velocity=np.asarray(velocities, dtype=np.float32),
        base_orientation_xyzw=np.asarray(sample["imu"]["quaternion_xyzw"], dtype=np.float32),
        base_angular_velocity=np.asarray(sample["imu"]["gyroscope_rad_s"], dtype=np.float32),
        base_linear_acceleration=np.asarray(sample["imu"]["accelerometer_m_s2"], dtype=np.float32),
    )


class ActionStats:
    """Finite running statistics without retaining action payloads."""

    def __init__(self) -> None:
        self.shape: tuple[int, ...] | None = None
        self.count = 0
        self.nan_count = 0
        self.inf_count = 0
        self.min_values: list[float] = []
        self.max_values: list[float] = []
        self.sum_values: list[float] = []
        self.sum_squares: list[float] = []
        self._last_min: list[float] = []
        self._last_max: list[float] = []

    def update(self, action: np.ndarray) -> None:
        value = np.asarray(action)
        self.shape = value.shape
        self.count += 1
        finite = np.isfinite(value)
        self.nan_count += int(np.isnan(value).sum())
        self.inf_count += int(np.isinf(value).sum())
        if value.ndim == 1:
            safe = np.where(finite, value, np.nan)
            local_min = np.nanmin(safe)
            local_max = np.nanmax(safe)
            self._last_min = [float(local_min)]
            self._last_max = [float(local_max)]
        else:
            safe = np.where(finite, value, np.nan)
            with np.errstate(invalid="ignore"):
                local_min = np.nanmin(safe, axis=0).astype(np.float64)
                local_max = np.nanmax(safe, axis=0).astype(np.float64)
            self._last_min = local_min.tolist()
            self._last_max = local_max.tolist()
        if not self.min_values:
            self.min_values = local_min.tolist()
            self.max_values = local_max.tolist()
            self.sum_values = np.zeros(value.shape, dtype=np.float64).tolist()
            self.sum_squares = np.zeros(value.shape, dtype=np.float64).tolist()
        else:
            self.min_values = np.minimum(self.min_values, local_min).tolist()
            self.max_values = np.maximum(self.max_values, local_max).tolist()
        numeric = np.where(finite, value, 0.0).astype(np.float64)
        self.sum_values = (np.asarray(self.sum_values) + numeric).tolist()
        self.sum_squares = (np.asarray(self.sum_squares) + numeric * numeric).tolist()

    def snapshot(self) -> dict[str, Any]:
        if not self.count:
            return {
                "shape": None,
                "count": 0,
                "nan_count": 0,
                "inf_count": 0,
            }
        sums = np.asarray(self.sum_values, dtype=np.float64)
        squares = np.asarray(self.sum_squares, dtype=np.float64)
        mean = sums / self.count
        variance = np.maximum(squares / self.count - mean * mean, 0.0)
        std = np.sqrt(variance)
        return {
            "shape": list(self.shape) if self.shape is not None else None,
            "count": self.count,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "min": self.min_values,
            "max": self.max_values,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "last_min_per_action": self._last_min,
            "last_max_per_action": self._last_max,
            "last_mean": fmean(np.asarray(self._last_min + self._last_max).tolist()) if self._last_min else None,
        }


class ObservationStats:
    """Per-dimension finite checks, moments, and bounded distribution samples.

    Samples retain only finite scalar values used for percentile estimates.  They
    never retain an entire observation payload and are capped at 10,000 policy
    steps (well under 2 MiB for a 49-dimensional input).
    """

    SAMPLE_CAPACITY = 10_000
    PERCENTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)

    def __init__(self) -> None:
        self.count = 0
        self.shape: tuple[int, ...] | None = None
        self.nan_counts: np.ndarray | None = None
        self.inf_counts: np.ndarray | None = None
        self.min_values: np.ndarray | None = None
        self.max_values: np.ndarray | None = None
        self.finite_samples: list[np.ndarray] = []

    def update(self, observation: np.ndarray) -> None:
        value = np.asarray(observation, dtype=np.float32).reshape(-1)
        finite = np.isfinite(value)
        nan_counts = np.isnan(value).astype(np.int64)
        inf_counts = np.isinf(value).astype(np.int64)
        if self.nan_counts is None or self.nan_counts.shape != value.shape:
            if self.count:
                raise ValueError("observation shape changed during dry run")
            self.shape = value.shape
            self.nan_counts = nan_counts
            self.inf_counts = inf_counts
        else:
            self.nan_counts += nan_counts
            self.inf_counts += inf_counts
        numeric = value.astype(np.float64)
        local_min = np.where(finite, numeric, np.inf)
        local_max = np.where(finite, numeric, -np.inf)
        self.min_values = local_min if self.min_values is None else np.minimum(self.min_values, local_min)
        self.max_values = local_max if self.max_values is None else np.maximum(self.max_values, local_max)
        if len(self.finite_samples) < self.SAMPLE_CAPACITY:
            self.finite_samples.append(np.where(finite, numeric, np.nan))
        self.count += 1

    @staticmethod
    def _finite_or_none(value: float) -> float | None:
        return float(value) if math.isfinite(value) else None

    @classmethod
    def _nearest_rank_percentile(cls, sorted_values: np.ndarray, percentile: float) -> float | None:
        if sorted_values.size == 0:
            return None
        rank = max(1, math.ceil(percentile * sorted_values.size))
        return float(sorted_values[rank - 1])

    @classmethod
    def _distribution_summary(cls, values: np.ndarray) -> dict[str, Any]:
        finite = values[np.isfinite(values)]
        percentile_keys = [f"p{int(percentile * 100):02d}" for percentile in cls.PERCENTILES]
        if finite.size == 0:
            return {"count": 0, "min": None, "max": None, "mean": None, "std": None, **dict.fromkeys(percentile_keys)}
        sorted_values = np.sort(finite)
        mean = float(np.mean(sorted_values))
        variance = max(float(np.mean(sorted_values * sorted_values)) - mean * mean, 0.0)
        percentiles = {
            key: cls._nearest_rank_percentile(sorted_values, percentile)
            for key, percentile in zip(percentile_keys, cls.PERCENTILES)
        }
        return {
            "count": int(sorted_values.size),
            "min": float(sorted_values[0]),
            "max": float(sorted_values[-1]),
            "mean": mean,
            "std": math.sqrt(variance),
            **percentiles,
        }

    def _section_summary(self, offset: int, size: int) -> dict[str, Any]:
        empty = {"offset": offset, "size": size, "all_finite": True, "nan_count": 0, "inf_count": 0}
        if self.count == 0 or self.nan_counts is None:
            return empty
        nan_count = int(self.nan_counts[offset : offset + size].sum())
        inf_count = int(self.inf_counts[offset : offset + size].sum())
        section_samples = [sample[offset : offset + size].reshape(-1) for sample in self.finite_samples]
        values = np.concatenate(section_samples) if section_samples else np.asarray([], dtype=np.float64)
        return {
            **empty,
            "all_finite": nan_count + inf_count == 0,
            "nan_count": nan_count,
            "inf_count": inf_count,
            **self._distribution_summary(values),
        }

    def snapshot(self) -> dict[str, Any]:
        if self.count == 0 or self.nan_counts is None:
            return {
                "shape": None,
                "count": 0,
                "all_finite": True,
                "per_dimension_finite": [],
                "nan_count": [],
                "inf_count": [],
                "min": [],
                "max": [],
                "mean": [],
                "std": [],
                **{f"p{int(percentile * 100):02d}": [] for percentile in self.PERCENTILES},
                "by_section": {},
            }
        finite_per_dimension = (self.nan_counts + self.inf_counts) == 0
        samples = (
            np.stack(self.finite_samples, axis=0)
            if self.finite_samples
            else np.empty((0, self.shape[0]), dtype=np.float64)
        )
        dimensions = [self._distribution_summary(samples[:, index]) for index in range(self.shape[0])]
        return {
            "shape": list(self.shape) if self.shape is not None else None,
            "count": self.count,
            "sample_count": len(self.finite_samples),
            "all_finite": bool(np.all(finite_per_dimension)),
            "per_dimension_finite": finite_per_dimension.tolist(),
            "nan_count": self.nan_counts.tolist(),
            "inf_count": self.inf_counts.tolist(),
            "min": [self._finite_or_none(value) for value in self.min_values],
            "max": [self._finite_or_none(value) for value in self.max_values],
            "mean": [item["mean"] for item in dimensions],
            "std": [item["std"] for item in dimensions],
            **{
                f"p{int(percentile * 100):02d}": [item[f"p{int(percentile * 100):02d}"] for item in dimensions]
                for percentile in self.PERCENTILES
            },
            "by_section": {
                name: self._section_summary(offset, size) for name, (offset, size) in OBSERVATION_LAYOUT.items()
            },
        }


class PolicyDryRun:
    """Decode LowState, infer actions, update statistics, then discard actions."""

    def __init__(
        self,
        artifact: Artifact,
        command: tuple[float, float, float],
        *,
        policy: Any,
        clock_ns=lambda: time.monotonic_ns(),
        print_interval_s: float,
        validate_crc: bool = True,
        stream: TextIO | None = None,
        policy_decimation: int = 25,
        inference_clock_ns=lambda: time.monotonic_ns(),
    ) -> None:
        self.robot, self.task_spec = validate_contract(artifact)
        self.processor = Go2WalkPolicyProcessor(self.task_spec, self.robot)
        self.policy = policy
        self.inference_clock_ns = inference_clock_ns
        self.command = np.asarray(command, dtype=np.float32)
        self.lowstate = lowstate_probe.LowStateProbe(
            lowstate_probe_crc_dummy(),
            clock_ns=clock_ns,
            print_interval_s=print_interval_s,
            validate_crc=False,
            stream=io.StringIO(),
        )
        self.validate_crc = validate_crc
        self.crc: Any | None = None
        self.stream = stream if stream is not None else sys.stdout
        self.counters = DryRunCounters()
        self.stats = ActionStats()
        self.observation_stats = ObservationStats()
        self.inference_latencies_ns: list[int] = []
        self._first_valid_receive_ns: int | None = None
        self._last_valid_receive_ns: int | None = None
        self._first_valid_robot_tick: int | None = None
        self._last_valid_robot_tick: int | None = None
        self._last_decoded_robot_tick: int | None = None
        self.tick_gap_events = 0
        self.skipped_robot_ticks = 0
        self.tick_out_of_order_events = 0
        self.duplicate_tick_events = 0
        self._first_inference_start_ns: int | None = None
        self._last_inference_start_ns: int | None = None
        self.step = 0
        self.policy_period_s = artifact.manifest.control.period_s
        self.policy_decimation = policy_decimation
        self.observation_snapshots: deque[np.ndarray] = deque(maxlen=1)

    def bind_crc(self, crc: Any) -> None:
        self.crc = crc

    def __call__(self, message: Any) -> None:
        received_at_ns = self.lowstate.clock_ns()
        self.counters.received += 1
        try:
            if self.validate_crc and self.crc is not None and self.crc.Crc(message) != message.crc:
                self.counters.crc_errors += 1
                return
            sample = lowstate_probe.decode_state(message, received_at_ns)
        except Exception:
            self.counters.decode_errors += 1
            return
        self.counters.decoded += 1
        self.counters.valid += 1
        robot_tick = int(sample["robot_tick"])
        if self._first_valid_receive_ns is None:
            self._first_valid_receive_ns = received_at_ns
            self._first_valid_robot_tick = robot_tick
        self._last_valid_receive_ns = received_at_ns
        self._last_valid_robot_tick = robot_tick
        if self._last_decoded_robot_tick is not None:
            tick_delta = robot_tick - self._last_decoded_robot_tick
            if tick_delta < 0:
                self.tick_out_of_order_events += 1
            elif tick_delta == 0:
                self.duplicate_tick_events += 1
            elif tick_delta > 1:
                self.tick_gap_events += 1
                self.skipped_robot_ticks += tick_delta - 1
        self._last_decoded_robot_tick = robot_tick
        if (self.counters.received - 1) % self.policy_decimation != 0:
            return
        try:
            state = make_robot_state(sample, self.robot)
            self.processor.validate_command(PlanarVelocityCommand(self.command.reshape(1, 3)))
            context = ControlContext(
                step=self.step,
                elapsed_time_s=self.step * self.policy_period_s,
                command=PlanarVelocityCommand(self.command.reshape(1, 3)),
                dt_s=self.policy_period_s,
            )
            observation = self.processor.build_observation(state, context)
            inference_start_ns = self.inference_clock_ns()
            raw_action = self.policy.infer(observation)
            inference_latency_ns = self.inference_clock_ns() - inference_start_ns
            # Update only the processor's action history; this never creates a servo command.
            self.processor.process_action(raw_action)
            self.observation_snapshots.append(observation.copy())
        except Exception as error:
            self.counters.inference_errors += 1
            print(f"policy inference error: {type(error).__name__}: {error}", file=self.stream)
            return
        self.counters.inferred += 1
        self.step += 1
        if self._first_inference_start_ns is None:
            self._first_inference_start_ns = inference_start_ns
        self._last_inference_start_ns = inference_start_ns
        self.inference_latencies_ns.append(inference_latency_ns)
        self.observation_stats.update(observation)
        self.stats.update(raw_action)

    @staticmethod
    def _endpoint_frequency_hz(first_ns: int | None, last_ns: int | None, count: int) -> float | None:
        if first_ns is None or last_ns is None or count < 2 or last_ns <= first_ns:
            return None
        return (count - 1) / ((last_ns - first_ns) / 1e9)

    @staticmethod
    def _nearest_rank_percentile(sorted_values: list[int], percentile: float) -> int | None:
        if not sorted_values:
            return None
        rank = max(1, math.ceil(percentile * len(sorted_values)))
        return sorted_values[rank - 1]

    def diagnostics(self) -> dict[str, Any]:
        wall_hz = self._endpoint_frequency_hz(
            self._first_valid_receive_ns, self._last_valid_receive_ns, self.counters.valid
        )
        tick_hz = None
        if (
            self._first_valid_robot_tick is not None
            and self._last_valid_robot_tick is not None
            and self._last_valid_robot_tick > self._first_valid_robot_tick
            and self._first_valid_receive_ns is not None
            and self._last_valid_receive_ns is not None
            and self._last_valid_receive_ns > self._first_valid_receive_ns
        ):
            tick_hz = (self._last_valid_robot_tick - self._first_valid_robot_tick) / (
                (self._last_valid_receive_ns - self._first_valid_receive_ns) / 1e9
            )
        inference_hz = self._endpoint_frequency_hz(
            self._first_inference_start_ns, self._last_inference_start_ns, self.counters.inferred
        )
        sorted_latencies = sorted(self.inference_latencies_ns)
        latency = {
            "count": len(sorted_latencies),
            "min_ns": sorted_latencies[0] if sorted_latencies else None,
            "max_ns": sorted_latencies[-1] if sorted_latencies else None,
            "mean_ns": int(fmean(sorted_latencies)) if sorted_latencies else None,
            "p50_ns": self._nearest_rank_percentile(sorted_latencies, 0.50),
            "p95_ns": self._nearest_rank_percentile(sorted_latencies, 0.95),
            "p99_ns": self._nearest_rank_percentile(sorted_latencies, 0.99),
        }
        return {
            "valid": self.counters.valid,
            "invalid": self.counters.crc_errors + self.counters.decode_errors,
            "wall_frequency_hz": wall_hz,
            "robot_tick_frequency_hz": tick_hz,
            "first_valid_receive_ns": self._first_valid_receive_ns,
            "last_valid_receive_ns": self._last_valid_receive_ns,
            "first_valid_robot_tick": self._first_valid_robot_tick,
            "last_valid_robot_tick": self._last_valid_robot_tick,
            "tick_gap_events": self.tick_gap_events,
            "skipped_robot_ticks": self.skipped_robot_ticks,
            "tick_out_of_order_events": self.tick_out_of_order_events,
            "duplicate_tick_events": self.duplicate_tick_events,
            "inference_frequency_hz": inference_hz,
            "inference_latency": latency,
        }


def lowstate_probe_crc_dummy() -> Any:
    class NeverUsedCRC:
        def Crc(self, message: Any) -> int:
            return int(message.crc)

    return NeverUsedCRC()


def io_blackhole() -> TextIO:
    class Blackhole:
        def write(self, value: str) -> None:
            del value

        def flush(self) -> None:
            return None

    return Blackhole()  # type: ignore[return-value]


def frequency_hz(times_ns: list[int]) -> float | None:
    if len(times_ns) < 2:
        return None
    elapsed_s = (times_ns[-1] - times_ns[0]) / 1e9
    return (len(times_ns) - 1) / elapsed_s if elapsed_s > 0 else None


def run(
    config: DryRunConfig,
    *,
    sdk: tuple[Any, Any, Any, Any] | None = None,
    policy_factory: Any | None = None,
    stream: TextIO | None = None,
) -> int:
    artifact = load_artifact(config.artifact)
    robot, task_spec = validate_contract(artifact)
    del robot, task_spec
    initialize, subscriber_type, low_state_type, crc_type = (
        sdk if sdk is not None else lowstate_probe.load_sdk()
    )
    policy_type = policy_factory or OnnxPolicy
    policy = policy_type(artifact.policy_path, artifact.manifest.policy.input, artifact.manifest.policy.output)
    output_stream = stream if stream is not None else sys.stdout
    dry = PolicyDryRun(
        artifact,
        config.command,
        policy=policy,
        print_interval_s=config.print_interval_s,
        validate_crc=config.validate_crc,
        stream=output_stream,
        policy_decimation=config.policy_decimation,
    )
    print(json.dumps({"mode": "read-only-policy-dry-run", "artifact": artifact_summary(artifact)}, sort_keys=True), file=output_stream)
    print(
        "safety: subscriber + ONNX inference only; no rt/lowcmd publisher, no mode switch, no motor command",
        file=output_stream,
    )
    subscriber: Any = None
    try:
        initialize(config.domain_id, config.network_interface)
        dry.bind_crc(crc_type())
        subscriber = subscriber_type(config.topic, low_state_type)
        subscriber.Init(dry, config.queue_depth)
        deadline_ns = time.monotonic_ns() + round(config.duration_s * 1e9)
        while time.monotonic_ns() < deadline_ns:
            time.sleep(0.05)
    finally:
        if subscriber is not None:
            close = getattr(subscriber, "Close", None) or getattr(subscriber, "close", None)
            if close is not None:
                close()

    result = {
        "config": {
            "artifact": str(config.artifact),
            "network_interface": config.network_interface,
            "domain_id": config.domain_id,
            "topic": config.topic,
            "command": list(config.command),
            "duration_s": config.duration_s,
            "policy_decimation": config.policy_decimation,
            "validate_crc": config.validate_crc,
            "safety": "subscriber-and-inference-only",
            "state_label": config.state_label,
        },
        "state_label": config.state_label,
        "counters": asdict(dry.counters),
        "diagnostics": dry.diagnostics(),
        "observation_statistics": dry.observation_stats.snapshot(),
        "action_statistics": dry.stats.snapshot(),
    }
    if config.output is not None:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        with config.output.open("w", encoding="utf-8") as output_file:
            json.dump(result, output_file, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            output_file.write("\n")
    print("summary=" + json.dumps(result["counters"], ensure_ascii=False, sort_keys=True), file=output_stream)
    print(
        "diagnostics=" + json.dumps(result["diagnostics"], ensure_ascii=False, sort_keys=True),
        file=output_stream,
    )
    print(
        "observations=" + json.dumps(result["observation_statistics"], ensure_ascii=False, sort_keys=True),
        file=output_stream,
    )
    print("actions=" + json.dumps(result["action_statistics"], ensure_ascii=False, sort_keys=True), file=output_stream)
    return 0 if dry.counters.inferred > 0 and dry.counters.inference_errors == 0 else 1


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


def command_tuple(value: str) -> tuple[float, float, float]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("must contain exactly three comma-separated values")
    values = tuple(float(part) for part in parts)
    if not all(math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("values must be finite")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Go2 ONNX policy dry run")
    parser.add_argument("network_interface")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--domain-id", type=_non_negative_int, default=0)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--command", type=command_tuple, default=(0.0, 0.0, 0.0))
    parser.add_argument("--duration-s", type=_positive_float, default=DEFAULT_DURATION_S)
    parser.add_argument("--print-interval-s", type=_positive_float, default=1.0)
    parser.add_argument("--queue-depth", type=_positive_int, default=16)
    parser.add_argument("--policy-decimation", type=_positive_int, default=25)
    parser.add_argument("--no-validate-crc", action="store_true")
    parser.add_argument(
        "--state-label",
        default="read_only_dry_run",
        help="operator label for this recording; no robot posture is inferred",
    )
    parser.add_argument("--output", type=Path)
    return parser


def config_from_args(args: argparse.Namespace) -> DryRunConfig:
    return DryRunConfig(
        artifact=args.artifact,
        network_interface=args.network_interface,
        domain_id=args.domain_id,
        topic=args.topic,
        command=args.command,
        duration_s=args.duration_s,
        print_interval_s=args.print_interval_s,
        queue_depth=args.queue_depth,
        policy_decimation=args.policy_decimation,
        validate_crc=not args.no_validate_crc,
        state_label=args.state_label,
        output=args.output,
    )


def main(argv: list[str] | None = None) -> int:
    return run(config_from_args(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
