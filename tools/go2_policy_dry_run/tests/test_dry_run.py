# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the read-only Go2 policy dry-run."""

from __future__ import annotations

import dataclasses
import ast
import io
import json
from typing import Any

import numpy as np
import pytest
from tools.go2_policy_dry_run import dry_run
from tools.go2_state_probe import probe as lowstate_probe


ARTIFACT = "artifacts/go2-walk-flat-v11-it3000.deploy"


class FakeMotor:
    def __init__(self, index: int) -> None:
        self.q = float(index) * 0.1
        self.dq = float(index) * -0.2
        self.tau_est = float(index) * 0.3
        self.temperature = 30 + index
        self.lost = 0


class FakeIMU:
    quaternion = [1.0, 0.0, 0.0, 0.0]
    gyroscope = [0.1, -0.2, 0.3]
    accelerometer = [0.0, 0.1, -9.8]
    rpy = [0.01, -0.02, 0.03]
    temperature = 40


class FakeLowState:
    def __init__(self, tick: int = 100) -> None:
        self.motor_state = [FakeMotor(index) for index in range(12)]
        self.imu_state = FakeIMU()
        self.foot_force = [101, 102, 103, 104]
        self.foot_force_est = [201, 202, 203, 204]
        self.tick = tick
        self.power_v = 29.4
        self.power_a = 1.25
        self.crc = 777


class FakeCRC:
    def Crc(self, message: Any) -> int:
        return int(message.crc)


class FakeSubscriber:
    instances: list["FakeSubscriber"] = []

    def __init__(self, topic: str, message_type: type, frames: int = 50) -> None:
        self.topic = topic
        self.message_type = message_type
        self.frames = frames
        self.callback = None
        self.queue_depth = None
        self.closed = False
        self.instances.append(self)

    def Init(self, callback: Any, queue_depth: int) -> None:
        self.callback = callback
        self.queue_depth = queue_depth
        for frame in range(self.frames):
            callback(FakeLowState(tick=100 + frame))

    def Close(self) -> None:
        self.closed = True


class FakeFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def __call__(self, domain_id: int, interface: str) -> None:
        self.calls.append((domain_id, interface))


def make_sdk(frames: int = 50) -> tuple[Any, Any, Any, Any]:
    FakeSubscriber.instances.clear()
    return FakeFactory(), lambda topic, message_type: FakeSubscriber(topic, message_type, frames), FakeLowState, FakeCRC


class RecordingPolicy:
    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []
        self.clock_ns: Any = None
        self.latency_ns = 2_000_000

    def infer(self, observation: np.ndarray) -> np.ndarray:
        self.observations.append(observation.copy())
        if self.clock_ns is not None:
            self.clock_ns()
        return np.linspace(-0.5, 0.5, 12, dtype=np.float32)


def run_dry(
    artifact: Any,
    frames: int = 50,
    policy: Any | None = None,
    command: tuple[float, float, float] = (0.5, 0.0, 0.0),
) -> tuple[dry_run.PolicyDryRun, Any]:
    policy = RecordingPolicy() if policy is None else policy
    instance = dry_run.PolicyDryRun(
        artifact,
        command,
        policy=policy,
        print_interval_s=1.0,
        validate_crc=True,
        stream=io.StringIO(),
        policy_decimation=25,
    )
    instance.bind_crc(FakeCRC())
    policy.clock_ns = instance.inference_clock_ns
    for frame in range(frames):
        instance(FakeLowState(tick=100 + frame))
    return instance, policy


@dataclasses.dataclass(frozen=True)
class _Artifact:
    manifest: Any

    @classmethod
    def from_manifest(cls, manifest: Any, **changes: Any) -> "_Artifact":
        return cls(dataclasses.replace(manifest, **changes))


class BoundedClock:
    def __init__(self) -> None:
        self.now_ns = 0

    def monotonic_ns(self) -> int:
        self.now_ns += 1_000_000
        return self.now_ns

    def sleep(self, seconds: float) -> None:
        del seconds


def test_validate_contract_accepts_v11_and_rejects_wrong_shapes() -> None:
    artifact = dry_run.load_artifact(ARTIFACT)
    robot, task = dry_run.validate_contract(artifact)
    assert artifact.manifest.task.name == "go2_walk/v1"
    assert robot.joint_count == 12
    assert task.gait_frequency_hz == pytest.approx(2.0)

    wrong_input = dataclasses.replace(artifact.manifest.policy.input, shape=(1, 48))
    wrong_policy = dataclasses.replace(artifact.manifest.policy, input=wrong_input)
    with pytest.raises(ValueError, match=r"\[1, 49\]"):
        dry_run.validate_contract(_Artifact.from_manifest(artifact.manifest, policy=wrong_policy))

    class WrongTask:
        @property
        def name(self) -> str:
            return "other/v1"

    class WrongManifest:
        task = WrongTask()

    class WrongArtifact:
        manifest = WrongManifest()

    with pytest.raises(ValueError, match="other/v1"):
        dry_run.validate_contract(WrongArtifact())

def test_make_robot_state_reorders_from_unitree_lowstate_order() -> None:
    artifact = dry_run.load_artifact(ARTIFACT)
    decoded = lowstate_probe.decode_state(FakeLowState(), 456)
    state = dry_run.make_robot_state(decoded, artifact.manifest.robot)

    assert state.joint_position.shape == (12,)
    assert state.joint_velocity.shape == (12,)
    np.testing.assert_allclose(state.joint_position, np.arange(12) * 0.1, rtol=0, atol=1e-7)
    np.testing.assert_allclose(state.joint_velocity, np.arange(12) * -0.2, rtol=0, atol=1e-7)
    np.testing.assert_allclose(state.base_angular_velocity, [0.1, -0.2, 0.3], rtol=0, atol=1e-7)
    np.testing.assert_allclose(state.base_orientation_xyzw, [0.0, 0.0, 0.0, 1.0], rtol=0, atol=1e-7)


def test_policy_dry_run_decimates_observation_and_updates_action_history() -> None:
    artifact = dry_run.load_artifact(ARTIFACT)
    dry, policy = run_dry(artifact, frames=50)

    assert dry.counters.received == 50
    assert dry.counters.decoded == 50
    assert dry.counters.inferred == 2
    assert dry.counters.inference_errors == 0
    assert len(policy.observations) == 2
    assert policy.observations[0].shape == (49,)
    assert policy.observations[0].dtype == np.float32
    np.testing.assert_allclose(dry.processor._previous_action, np.linspace(-0.5, 0.5, 12), rtol=0, atol=1e-7)
    assert dry.observation_snapshots[0].shape == (49,)


def test_observation_phase_advences_with_control_period() -> None:
    artifact = dry_run.load_artifact(ARTIFACT)
    _, policy = run_dry(artifact, frames=50)
    first = policy.observations[0][45:49]
    second = policy.observations[1][45:49]

    assert artifact.manifest.control.period_s == pytest.approx(0.02)
    assert artifact.manifest.task.gait_frequency_hz == pytest.approx(2.0)
    assert first.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert second.tolist() == pytest.approx([0.04, 0.54, 0.54, 0.04])


def test_action_stats_are_finite_running_statistics_without_payload_retention() -> None:
    stats = dry_run.ActionStats()
    stats.update(np.linspace(-1.0, 1.0, 12, dtype=np.float32))
    stats.update(np.linspace(1.0, -1.0, 12, dtype=np.float32))
    result = stats.snapshot()

    assert result["count"] == 2
    assert result["shape"] == [12]
    np.testing.assert_allclose(result["min"], -1.0, rtol=0, atol=1e-7)
    np.testing.assert_allclose(result["max"], 1.0, rtol=0, atol=1e-7)
    np.testing.assert_allclose(result["mean"], np.zeros(12), rtol=0, atol=1e-7)
    np.testing.assert_allclose(result["std"], np.abs(np.linspace(-1.0, 1.0, 12)), rtol=1e-6, atol=1e-7)
    assert not hasattr(stats, "actions")

    stats.update(np.asarray([np.nan, np.inf, *np.linspace(1.0, 2.0, 10)], dtype=np.float32))
    result = stats.snapshot()
    assert result["nan_count"] == 1
    assert result["inf_count"] == 1
    assert result["mean"][0] == pytest.approx(0.0)


def test_dry_run_counts_nonfinite_actions() -> None:
    stats = dry_run.ActionStats()
    stats.update(np.asarray([np.nan, np.inf, *np.linspace(-0.4, 0.4, 10)], dtype=np.float32))
    result = stats.snapshot()

    assert result["count"] == 1
    assert result["nan_count"] == 1
    assert result["inf_count"] == 1
    assert result["last_min_per_action"][0] == pytest.approx(-0.4)


def test_observation_stats_report_each_dimension_without_payload_retention() -> None:
    stats = dry_run.ObservationStats()
    first = np.linspace(-1.0, 1.0, 49, dtype=np.float32)
    second = first * -0.5
    second[7] = np.nan
    second[11] = np.inf
    second_finite = second.copy()
    second_finite[11] = -0.5 * first[11]
    stats.update(first)
    stats.update(second)
    result = stats.snapshot()

    assert result["count"] == 2
    assert result["shape"] == [49]
    assert result["all_finite"] is False
    assert result["per_dimension_finite"][7] is False
    assert result["per_dimension_finite"][11] is False
    assert result["nan_count"][7] == 1
    assert result["inf_count"][11] == 1
    assert result["min"][0] == pytest.approx(-1.0)
    assert result["max"][48] == pytest.approx(first[48])
    assert result["min"][7] == pytest.approx(first[7])
    assert result["max"][11] == pytest.approx(first[11])
    assert result["sample_count"] == 2
    assert result["mean"][0] == pytest.approx(-0.25)
    assert result["p25"][0] == pytest.approx(-1.0)
    assert result["p50"][0] == pytest.approx(-1.0)
    assert result["p75"][0] == pytest.approx(0.5)
    assert result["p99"][0] == pytest.approx(0.5)
    assert result["by_section"]["projected_gravity_body"]["offset"] == 3
    assert result["by_section"]["projected_gravity_body"]["size"] == 3
    assert result["by_section"]["projected_gravity_body"]["all_finite"] is True
    assert not hasattr(stats, "observations")


def test_dry_run_reports_tick_gaps_and_inference_latency_statistics() -> None:
    artifact = dry_run.load_artifact(ARTIFACT)
    class FixedLatencyClock:
        def __init__(self) -> None:
            self.now_ns = 0

        def __call__(self) -> int:
            self.now_ns += 2_000_000
            return self.now_ns

    inference_clock = FixedLatencyClock()
    instance = dry_run.PolicyDryRun(
        artifact,
        (0.0, 0.0, 0.0),
        policy=RecordingPolicy(),
        inference_clock_ns=inference_clock,
        print_interval_s=1.0,
        validate_crc=True,
        stream=io.StringIO(),
    )
    instance.bind_crc(FakeCRC())
    ticks = [10, 11, 13, 12, 12, 15]
    for tick in ticks:
        instance(FakeLowState(tick=tick))

    diagnostics = instance.diagnostics()
    assert instance.counters.valid == len(ticks)
    assert diagnostics["tick_gap_events"] == 2
    assert diagnostics["skipped_robot_ticks"] == 3
    assert diagnostics["tick_out_of_order_events"] == 1
    assert diagnostics["duplicate_tick_events"] == 1
    assert diagnostics["inference_latency"]["p50_ns"] == 2_000_000
    assert diagnostics["inference_latency"]["p95_ns"] == 2_000_000
    assert diagnostics["inference_latency"]["p99_ns"] == 2_000_000


def test_run_is_bounded_writes_json_and_closes_subscriber(monkeypatch, tmp_path) -> None:
    artifact = dry_run.load_artifact(ARTIFACT)
    FakeSubscriber.instances.clear()
    clock = BoundedClock()
    monkeypatch.setattr(dry_run.time, "monotonic_ns", clock.monotonic_ns)
    monkeypatch.setattr(dry_run.time, "sleep", clock.sleep)
    output = tmp_path / "dryrun" / "result.json"
    config = dry_run.DryRunConfig(
        artifact=dry_run.Path(ARTIFACT),
        network_interface="eno1",
        duration_s=0.001,
        output=output,
    )

    exit_code = dry_run.run(config, sdk=make_sdk(frames=25), policy_factory=lambda *args: RecordingPolicy())

    assert exit_code == 0
    assert FakeSubscriber.instances[0].closed is True
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["config"]["safety"] == "subscriber-and-inference-only"
    assert result["config"]["state_label"] == "read_only_dry_run"
    assert result["state_label"] == "read_only_dry_run"
    assert set(result["observation_statistics"]["by_section"]) == set(dry_run.OBSERVATION_LAYOUT)
    assert result["counters"] == {
        "received": 25,
        "decoded": 25,
        "crc_errors": 0,
        "decode_errors": 0,
            "valid": 25,
        "inferred": 1,
        "inference_errors": 0,
    }
    assert result["action_statistics"]["count"] == 1


def test_dry_run_contains_no_command_publisher_or_mode_switch() -> None:
    tree = ast.parse(dry_run.Path(dry_run.__file__).read_text(encoding="utf-8"))
    calls = {
        ast.unparse(node.func).lower()
        for node in ast.walk(tree)
        if isinstance(node, (ast.Call,)) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    imports = {
        ast.unparse(node).lower()
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    for forbidden in ("channelpublisher", "lowcmd", "motionswitcher", "releasemode", "serviceswitch"):
        assert not any(forbidden in value for value in calls | imports)

