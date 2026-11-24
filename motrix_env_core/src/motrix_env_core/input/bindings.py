# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Bindings that produce high-level commands from devices or configuration."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

import numpy as np

from motrix_env_core.input.command import FloatArray, PlanarVelocityCommand
from motrix_env_core.input.device import GamePadDevice, KeyboardDevice

CommandT = TypeVar("CommandT")


class CommandBinding(ABC, Generic[CommandT]):
    """Produce task commands from a device, configuration, or another input mechanism."""

    @abstractmethod
    def read_command(self, *, batch_size: int = 1) -> CommandT:
        """Produce one command batch of the requested size."""


class KeyboardPlanarVelocityBinding(CommandBinding[PlanarVelocityCommand]):
    """Map held ``W/S``, ``A/D``, and ``Q/E`` keys to planar velocity."""

    def __init__(
        self,
        device: KeyboardDevice,
        *,
        command_lower: Sequence[float] | FloatArray,
        command_upper: Sequence[float] | FloatArray,
    ) -> None:
        self._device = device
        self._command_lower = np.array(command_lower, dtype=np.float32, copy=True)
        self._command_upper = np.array(command_upper, dtype=np.float32, copy=True)
        if self._command_lower.shape != (3,) or self._command_upper.shape != (3,):
            raise ValueError("keyboard command lower and upper must have shape (3,)")
        if not np.all(np.isfinite(self._command_lower)) or not np.all(np.isfinite(self._command_upper)):
            raise ValueError("keyboard command lower and upper must contain only finite numbers")
        if np.any(self._command_lower > 0.0) or np.any(self._command_upper < 0.0):
            raise ValueError("keyboard command range must contain zero on every axis")

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        _validate_batch_size(batch_size)
        self._device.poll()
        direction = np.asarray(
            (
                int(self._device.is_pressing("w")) - int(self._device.is_pressing("s")),
                int(self._device.is_pressing("a")) - int(self._device.is_pressing("d")),
                int(self._device.is_pressing("q")) - int(self._device.is_pressing("e")),
            ),
            dtype=np.int8,
        )
        value = np.where(
            direction > 0,
            self._command_upper,
            np.where(direction < 0, self._command_lower, np.float32(0.0)),
        )
        return PlanarVelocityCommand(np.repeat(value[None, :], batch_size, axis=0))


class ScalableKeyboardPlanarVelocityBinding(CommandBinding[PlanarVelocityCommand]):
    """Map held movement keys to independently adjustable command fractions.

    ``UP``/``DOWN`` change all three scales together. Number-key pairs adjust
    one axis at a time: ``1``/``2`` forward/reverse, ``3``/``4`` lateral, and
    ``5``/``6`` yaw. Movement keys produce commands at the selected scales.
    """

    def __init__(
        self,
        device: KeyboardDevice,
        *,
        command_lower: Sequence[float] | FloatArray,
        command_upper: Sequence[float] | FloatArray,
        levels: Sequence[float] | None = None,
    ) -> None:
        self._binding = KeyboardPlanarVelocityBinding(
            device,
            command_lower=command_lower,
            command_upper=command_upper,
        )
        self._device = device
        self._command_lower = np.asarray(command_lower, dtype=np.float32)
        self._command_upper = np.asarray(command_upper, dtype=np.float32)
        if self._command_lower.shape != (3,) or self._command_upper.shape != (3,):
            raise ValueError("keyboard command lower and upper must have shape (3,)")
        if levels is None:
            levels = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        self._levels = tuple(float(value) for value in levels)
        if not self._levels:
            raise ValueError("speed levels must not be empty")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
            for value in self._levels
        ):
            raise ValueError("speed levels must contain finite numbers")
        if any(value <= 0.0 or value > 1.0 for value in self._levels):
            raise ValueError("speed levels must be inside (0, 1]")
        if len(set(self._levels)) != len(self._levels):
            raise ValueError("speed levels must be unique")
        self._levels = tuple(sorted(self._levels))
        self._level_indexes = np.full(3, len(self._levels) - 1, dtype=np.int32)

    @property
    def speed_scale(self) -> float:
        return self._levels[self._level_indexes[0]]

    @property
    def axis_speed_scales(self) -> tuple[float, float, float]:
        return tuple(self._levels[index] for index in self._level_indexes)  # type: ignore[return-value]

    @property
    def speed_levels(self) -> tuple[float, ...]:
        return self._levels

    @property
    def speed_level_index(self) -> int:
        return int(self._level_indexes[0])

    @property
    def axis_speed_level_indexes(self) -> tuple[int, int, int]:
        return tuple(int(index) for index in self._level_indexes)

    def _select_scale(self) -> None:
        changes = (
            (("up", "down"), (0, 1, 2)),
            (("1", "2"), (0,)),
            (("3", "4"), (1,)),
            (("5", "6"), (2,)),
        )
        for (increase_key, decrease_key), axes in changes:
            if self._device.is_key_down(increase_key):
                for axis in axes:
                    self._level_indexes[axis] = min(self._level_indexes[axis] + 1, len(self._levels) - 1)
            if self._device.is_key_down(decrease_key):
                for axis in axes:
                    self._level_indexes[axis] = max(self._level_indexes[axis] - 1, 0)

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        _validate_batch_size(batch_size)
        # The inner binding polls exactly once and freezes the frame. Reading
        # scale edges afterward shares that same frame instead of discarding it.
        command = self._binding.read_command(batch_size=1).values[0, :]
        self._select_scale()
        value = command * np.asarray(self.axis_speed_scales, dtype=np.float32)
        return PlanarVelocityCommand(np.repeat(value[None, :], batch_size, axis=0))


class GamePadPlanarVelocityBinding(CommandBinding[PlanarVelocityCommand]):
    """Map three gamepad axes to planar velocity."""

    def __init__(
        self,
        device: GamePadDevice,
        *,
        linear_x_axis: str,
        linear_y_axis: str,
        yaw_axis: str,
        linear_x_scale: float,
        linear_y_scale: float,
        yaw_scale: float,
        deadzone: float = 0.0,
        invert_linear_x: bool = False,
        invert_linear_y: bool = False,
        invert_yaw: bool = False,
    ) -> None:
        axes = (linear_x_axis, linear_y_axis, yaw_axis)
        if any(not isinstance(axis, str) or not axis for axis in axes):
            raise ValueError("axis names must be non-empty strings")
        self._device = device
        self._axes = axes
        self._scale = np.asarray((linear_x_scale, linear_y_scale, yaw_scale), dtype=np.float32)
        if np.any(self._scale < 0.0):
            raise ValueError("scale values must be non-negative")
        if isinstance(deadzone, bool) or not isinstance(deadzone, (int, float)):
            raise ValueError("deadzone must be a finite value inside [0, 1)")
        self._deadzone = float(deadzone)
        if not np.isfinite(self._deadzone) or not 0.0 <= self._deadzone < 1.0:
            raise ValueError("deadzone must be a finite value inside [0, 1)")
        inversions = (invert_linear_x, invert_linear_y, invert_yaw)
        if any(not isinstance(value, bool) for value in inversions):
            raise ValueError("axis inversion flags must be bool values")
        self._direction = np.asarray(tuple(-1.0 if value else 1.0 for value in inversions), dtype=np.float32)

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        _validate_batch_size(batch_size)
        self._device.poll()
        axes = np.asarray(tuple(self._device.axis_value(axis) for axis in self._axes), dtype=np.float32)
        axes[np.abs(axes) < self._deadzone] = 0.0
        value = axes * self._direction * self._scale
        return PlanarVelocityCommand(np.repeat(value[None, :], batch_size, axis=0))


class BoundedGamePadPlanarVelocityBinding(CommandBinding[PlanarVelocityCommand]):
    """Map normalized gamepad axes into artifact-bounded planar velocity commands."""

    def __init__(
        self,
        device: GamePadDevice,
        *,
        linear_x_axis: str,
        linear_y_axis: str,
        yaw_axis: str,
        command_lower: Sequence[float] | FloatArray,
        command_upper: Sequence[float] | FloatArray,
        deadzone: float = 0.0,
        range_scale: Sequence[float] | FloatArray = (1.0, 1.0, 1.0),
        invert_linear_x: bool = False,
        invert_linear_y: bool = False,
        invert_yaw: bool = False,
        deadman_button: str | None = None,
    ) -> None:
        axes = (linear_x_axis, linear_y_axis, yaw_axis)
        if any(not isinstance(axis, str) or not axis for axis in axes):
            raise ValueError("axis names must be non-empty strings")
        self._device = device
        self._axes = axes
        self._command_lower = np.asarray(command_lower, dtype=np.float32)
        self._command_upper = np.asarray(command_upper, dtype=np.float32)
        self._range_scale = np.asarray(range_scale, dtype=np.float32)
        for name, value in (
            ("command_lower", self._command_lower),
            ("command_upper", self._command_upper),
            ("range_scale", self._range_scale),
        ):
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain three finite values")
        if np.any(self._command_lower > 0.0) or np.any(self._command_upper < 0.0):
            raise ValueError("gamepad command range must contain zero on every axis")
        if np.any(self._command_lower > self._command_upper):
            raise ValueError("gamepad command lower must not exceed upper")
        if np.any(self._range_scale < 0.0) or np.any(self._range_scale > 1.0):
            raise ValueError("range_scale values must be inside [0, 1]")
        if isinstance(deadzone, bool) or not isinstance(deadzone, (int, float)):
            raise ValueError("deadzone must be a finite value inside [0, 1)")
        self._deadzone = float(deadzone)
        if not np.isfinite(self._deadzone) or not 0.0 <= self._deadzone < 1.0:
            raise ValueError("deadzone must be a finite value inside [0, 1)")
        inversions = (invert_linear_x, invert_linear_y, invert_yaw)
        if any(not isinstance(value, bool) for value in inversions):
            raise ValueError("axis inversion flags must be bool values")
        self._direction = np.asarray(tuple(-1.0 if value else 1.0 for value in inversions), dtype=np.float32)
        if deadman_button is not None and (not isinstance(deadman_button, str) or not deadman_button):
            raise ValueError("deadman_button must be null or a non-empty string")
        self._deadman_button = deadman_button

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        _validate_batch_size(batch_size)
        self._device.poll()
        if self._deadman_button is not None and not self._device.is_button_pressing(self._deadman_button):
            value = np.zeros(3, dtype=np.float32)
        else:
            axes = np.asarray(tuple(self._device.axis_value(axis) for axis in self._axes), dtype=np.float32)
            axes = np.clip(axes * self._direction, -1.0, 1.0)
            magnitude = np.maximum(np.abs(axes) - self._deadzone, 0.0) / (1.0 - self._deadzone)
            normalized = np.copysign(magnitude, axes)
            bounded = np.where(
                normalized >= 0.0,
                normalized * self._command_upper,
                (-normalized) * self._command_lower,
            )
            value = (bounded * self._range_scale).astype(np.float32)
        return PlanarVelocityCommand(np.repeat(value[None, :], batch_size, axis=0))


class ConstantPlanarVelocityBinding(CommandBinding[PlanarVelocityCommand]):
    """Replicate one configured planar velocity across the requested batch."""

    def __init__(self, value: Sequence[float] | FloatArray) -> None:
        self._value = np.asarray(value, dtype=np.float32)

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        _validate_batch_size(batch_size)
        return PlanarVelocityCommand(np.repeat(self._value[None, :], batch_size, axis=0))


class RateLimitedPlanarVelocityBinding(CommandBinding[PlanarVelocityCommand]):
    """Limit per-axis command change with separate acceleration and braking rates."""

    def __init__(
        self,
        binding: CommandBinding[PlanarVelocityCommand],
        *,
        max_acceleration: Sequence[float] | FloatArray,
        startup_acceleration: Sequence[float] | FloatArray | None = None,
        stop_acceleration: Sequence[float] | FloatArray | None = None,
        dt_s: float,
    ) -> None:
        self._binding = binding
        startup = np.asarray(
            max_acceleration if startup_acceleration is None else startup_acceleration,
            dtype=np.float32,
        )
        stop = np.asarray(max_acceleration if stop_acceleration is None else stop_acceleration, dtype=np.float32)
        for name, values in (("max/startup_acceleration", startup), ("max/stop_acceleration", stop)):
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain three finite values")
            if np.any(values <= 0.0):
                raise ValueError(f"{name} values must be positive")
        self._startup_acceleration = startup
        self._stop_acceleration = stop
        if isinstance(dt_s, bool) or not isinstance(dt_s, (int, float)) or not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be a positive finite number")
        self._dt_s = float(dt_s)
        self._previous = np.zeros(3, dtype=np.float32)

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        target = self._binding.read_command(batch_size=1).values[0, :]
        delta = target - self._previous
        moving_away_from_zero = np.sign(delta) == np.sign(self._previous)
        moving_from_rest = self._previous == 0.0
        accelerates = moving_away_from_zero | moving_from_rest
        maximum_delta = np.where(
            accelerates,
            self._startup_acceleration * self._dt_s,
            self._stop_acceleration * self._dt_s,
        ).astype(np.float32)
        limited = self._previous + np.clip(delta, -maximum_delta, maximum_delta)
        self._previous = limited.astype(np.float32)
        return PlanarVelocityCommand(np.repeat(self._previous[None, :], batch_size, axis=0))


def _validate_batch_size(batch_size: int) -> None:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")


__all__ = [
    "BoundedGamePadPlanarVelocityBinding",
    "CommandBinding",
    "ConstantPlanarVelocityBinding",
    "GamePadPlanarVelocityBinding",
    "KeyboardPlanarVelocityBinding",
    "RateLimitedPlanarVelocityBinding",
    "ScalableKeyboardPlanarVelocityBinding",
]
