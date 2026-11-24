# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-specific random planar-velocity command binding for quadruped training."""

import numpy as np
from numpy.typing import NDArray

from motrix_env_core.input import CommandBinding, PlanarVelocityCommand


class RandomPlanarVelocityBinding(CommandBinding[PlanarVelocityCommand]):
    """Sample independent planar velocities and optional standing commands."""

    def __init__(
        self,
        lower: NDArray[np.float32],
        upper: NDArray[np.float32],
        *,
        rng: np.random.Generator,
        standing_probability: float = 0.0,
        hard_case_probability: float = 0.0,
        hard_cases: tuple[tuple[float, float, float], ...] = (),
    ) -> None:
        self._lower = lower
        self._upper = upper
        self._standing_probability = standing_probability
        self._hard_case_probability = hard_case_probability
        self._hard_cases = np.asarray(hard_cases, dtype=np.float32)
        self._rng = rng

        if hard_case_probability > 0.0 and not hard_cases:
            raise ValueError("hard_case_probability requires at least one hard case")
        if len(self._hard_cases) and (
            np.any(self._hard_cases < lower) or np.any(self._hard_cases > upper)
        ):
            raise ValueError("hard-case commands must lie inside the configured velocity bounds")

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        values = self._rng.uniform(self._lower, self._upper, size=(batch_size, 3))
        if self._hard_case_probability > 0.0:
            hard = self._rng.random(batch_size) < self._hard_case_probability
            choices = self._rng.integers(0, len(self._hard_cases), size=batch_size)
            values[hard] = self._hard_cases[choices[hard]]
        if self._standing_probability > 0.0:
            standing = self._rng.random(batch_size) < self._standing_probability
            values[standing] = 0.0
        return PlanarVelocityCommand(values.astype(np.float32))

    @property
    def hard_case_probability(self) -> float:
        return self._hard_case_probability

    def set_hard_case_probability(self, probability: float) -> None:
        """Update the hard-case sampling probability for a training curriculum."""
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"hard-case probability must be inside [0, 1], got {probability}")
        if probability > 0.0 and not len(self._hard_cases):
            raise ValueError("hard-case probability requires at least one hard case")
        self._hard_case_probability = float(probability)


__all__ = ["RandomPlanarVelocityBinding"]
