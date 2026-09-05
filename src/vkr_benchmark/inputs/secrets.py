"""Benchmark secret-source construction."""

from __future__ import annotations

from vkr_benchmark.randomness import SecretSource, Shake256SecretSource


def create_secret_source(secret_id: str) -> SecretSource:
    """Create the specification-v0.1 deterministic SHAKE256 secret stream."""

    return Shake256SecretSource(secret_id)
