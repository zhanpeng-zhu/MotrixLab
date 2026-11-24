# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the read-only Go2 LowState probe."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from tools.go2_state_probe import probe


class FakeClock:
    def __init__(self) -> None:
        self.now_ns = 1_000_000_000

    def monotonic_ns(self) -> int:
        self.now_ns += 10_000_000
        return self.now_ns

    def sleep(self, seconds: float) -> None:
        del seconds


class FakeCRC:
    def __init__(self, expected: int | None = None) -> None:
        self.expected = expected

    def Crc(self, message: Any) -> int:
        return self.expected if self.expected is not None else int(message.crc)


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


class FakeSubscriber:
    instances: list["FakeSubscriber"] = []

    def __init__(self, topic: str, message_type: type, frames: int = 2) -> None:
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


def make_sdk(frames: int = 2) -> tuple[FakeFactory, Any, Any, type[FakeCRC]]:
    FakeSubscriber.instances.clear()
    factory = FakeFactory()
    low_state_type = FakeLowState

    def subscriber_factory(topic: str, message_type: type) -> FakeSubscriber:
        return FakeSubscriber(topic, message_type, frames=frames)

    return factory, subscriber_factory, low_state_type, FakeCRC


def test_decode_state_maps_all_required_fields() -> None:
    sample = probe.decode_state(FakeLowState(tick=123), 456)

    assert sample["received_at_ns"] == 456
    assert sample["robot_tick"] == 123
    assert sample["imu"]["quaternion_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert sample["imu"]["quaternion_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert sample["imu"]["gyroscope_rad_s"] == [0.1, -0.2, 0.3]
    assert sample["imu"]["accelerometer_m_s2"] == [0.0, 0.1, -9.8]
    assert sample["imu"]["rpy_rad"] == [0.01, -0.02, 0.03]
    assert sample["imu"]["temperature_c"] == 40
    assert [joint["name"] for joint in sample["joints"]] == list(probe.JOINT_NAMES)
    assert sample["joints"][2]["position_rad"] == pytest.approx(0.2)
    assert sample["joints"][2]["velocity_rad_s"] == pytest.approx(-0.4)
    assert sample["joints"][2]["torque_est_n_m"] == pytest.approx(0.6)
    assert sample["joints"][2]["temperature_c"] == 32
    assert list(sample["feet"]) == ["FL", "FR", "RL", "RR"]
    assert sample["feet"]["RR"] == {"force_raw": 104, "force_est_raw": 204}
    assert sample["power"] == {"voltage_v": 29.4, "current_a": 1.25}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("motor_state", []),
        ("foot_force", []),
        ("foot_force_est", []),
    ],
)
def test_decode_state_rejects_invalid_shapes(field: str, value: list[Any]) -> None:
    message = FakeLowState()
    setattr(message, field, value)

    with pytest.raises(ValueError):
        probe.decode_state(message, 1)


def test_probe_counts_crc_errors_without_recording_sample() -> None:
    message = FakeLowState()
    message.crc = 999
    device = probe.LowStateProbe(FakeCRC(expected=123), print_interval_s=1.0, stream=io.StringIO())

    device(message)

    assert device.counters.received == 1
    assert device.counters.valid == 0
    assert device.counters.crc_errors == 1
    assert device.counters.decode_errors == 0
    assert device.samples == []


def test_probe_records_valid_frame_and_reports_frequencies() -> None:
    message = FakeLowState()
    device = probe.LowStateProbe(
        FakeCRC(),
        clock_ns=lambda: 0,
        print_interval_s=1.0,
        stream=io.StringIO(),
    )

    device(message)

    assert device.counters.valid == 1
    assert device.samples[0]["robot_tick"] == 100
    assert device.statistics()["wall_frequency_hz"] is None
    assert probe.LowStateProbe.frequency_hz([0.1, 0.1, 0.1]) == pytest.approx(10.0)


def test_parser_accepts_domain_zero_and_rejects_negative_domain() -> None:
    parser = probe.build_parser()

    config = probe.config_from_args(parser.parse_args(["enp5s0", "--domain-id", "0"]))
    assert config.domain_id == 0

    with pytest.raises(SystemExit):
        parser.parse_args(["enp5s0", "--domain-id", "-1"])


def test_run_is_bounded_writes_json_and_closes_subscriber(monkeypatch, tmp_path) -> None:
    clock = FakeClock()
    monkeypatch.setattr(probe.time, "monotonic_ns", clock.monotonic_ns)
    monkeypatch.setattr(probe.time, "sleep", clock.sleep)
    sdk = make_sdk(frames=2)
    output = tmp_path / "lowstate" / "probe.json"
    config = probe.ProbeConfig(
        network_interface="enp5s0",
        domain_id=0,
        duration_s=0.001,
        queue_depth=4,
        output=output,
    )

    exit_code = probe.run(config, sdk=sdk, stream=io.StringIO())

    assert exit_code == 0
    assert sdk[0].calls == [(0, "enp5s0")]
    assert len(FakeSubscriber.instances) == 1
    subscriber = FakeSubscriber.instances[0]
    assert subscriber.topic == probe.DEFAULT_TOPIC
    assert subscriber.message_type is FakeLowState
    assert subscriber.queue_depth == 4
    assert subscriber.closed
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["config"]["safety"] == "subscriber-only"
    assert result["statistics"]["valid"] == 2
    assert len(result["records"]) == 2


def test_run_reports_missing_sdk_as_runtime_error(monkeypatch) -> None:
    def missing_sdk() -> None:
        raise ImportError("unitree_sdk2py unavailable")

    monkeypatch.setattr(probe, "load_sdk", missing_sdk)

    with pytest.raises(RuntimeError, match="unitree_sdk2py"):
        probe.run(probe.ProbeConfig(network_interface="enp5s0", duration_s=0.001))
