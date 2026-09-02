"""Isolated random streams for method and control-generation randomness."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import MutableSequence, TypeVar

_T = TypeVar("_T")


class RandomSource(ABC):
    """Minimal random interface consumed by benchmark components."""

    @abstractmethod
    def random(self) -> float:
        """Return a value uniformly distributed in [0, 1)."""

    @abstractmethod
    def randbelow(self, upper: int) -> int:
        """Return a uniform integer in ``range(upper)``."""

    def shuffle(self, values: MutableSequence[_T]) -> None:
        """In-place Fisher-Yates shuffle using only this source."""

        for i in range(len(values) - 1, 0, -1):
            j = self.randbelow(i + 1)
            values[i], values[j] = values[j], values[i]


class _PythonRandomSource(RandomSource):
    """Local ``random.Random`` wrapper; never touches module-global RNG state."""

    stream_role = "generic"

    def __init__(self, seed: int | str | bytes | bytearray) -> None:
        self._rng = random.Random(seed)

    def random(self) -> float:
        return self._rng.random()

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("upper must be positive")
        return self._rng.randrange(upper)


class MethodRandomSource(_PythonRandomSource):
    """Random stream reserved for method-internal stochasticity."""

    stream_role = "method"


class ControlRandomSource(_PythonRandomSource):
    """Random stream reserved for sampling ordinary control text."""

    stream_role = "control"
