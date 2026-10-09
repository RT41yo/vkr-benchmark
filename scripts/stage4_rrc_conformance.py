"""Reusable Stage-4 RRC conformance fixtures and pinned-source oracle.

The oracle is a small independent Python transcription of the arithmetic in
``ryehr/RRC_steganography@dae326259.../rrc_core.py``.  It deliberately lives
outside ``src/vkr_benchmark``.  The companion author runner executes the
original pinned ``rrc_core.py`` directly; this module exists so the ordinary
unit suite can exercise conformance without an external checkout.

All fixtures begin from an already-final ``P_reference``.  This is deliberate:
normalized RRC does not own temperature/top-k/top-p.  The conformance question
here is whether, given the same per-step probability partitions, message and
key, the benchmark adapter makes the same range-coding decisions and recovers
the same message as the pinned author core.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_DOWN, localcontext
from math import ceil, log10
import random
from typing import Iterable, Sequence

import numpy as np

from vkr_benchmark.distributions import ReferenceDistribution, StepContext
from vkr_benchmark.methods import MethodEnvironment, RRCMethod
from vkr_benchmark.randomness import MethodRandomSource, SecretSource

REFERENCE_REPOSITORY = "https://github.com/ryehr/RRC_steganography"
REFERENCE_COMMIT = "dae326259e4fca8bc4fcf460dafdbc0e88a0a71a"


@dataclass(frozen=True, slots=True)
class RRCFixture:
    fixture_id: str
    probabilities: tuple[tuple[float, ...], ...]
    secret_bits: tuple[int, ...]
    seed: int
    precision_margin: int = 40
    verify_termination: bool = True

    @property
    def vocab_size(self) -> int:
        return len(self.probabilities[0])

    @property
    def payload_bits(self) -> int:
        return len(self.secret_bits)

    @property
    def max_tokens(self) -> int:
        # A generous deterministic safety cap for synthetic fixtures.
        return max(64, 8 * self.payload_bits)

    def reference_at(self, step: int) -> ReferenceDistribution:
        row = self.probabilities[step % len(self.probabilities)]
        return ReferenceDistribution(np.asarray(row, dtype=np.float32))


@dataclass(frozen=True, slots=True)
class RRCOutcome:
    token_ids: tuple[int, ...]
    recovered_bits: tuple[int, ...]
    precision_digits: int
    termination_retries: int


class FixedSecretSource(SecretSource):
    def __init__(self, bits: Iterable[int]) -> None:
        self._bits = tuple(int(bit) for bit in bits)
        if not self._bits or any(bit not in (0, 1) for bit in self._bits):
            raise ValueError("RRC conformance secret must be a non-empty bit string")
        self._position = 0

    @property
    def position(self) -> int:
        return self._position

    def read_bits(self, count: int) -> tuple[int, ...]:
        stop = self._position + count
        if stop > len(self._bits):
            raise RuntimeError("RRC conformance secret exhausted")
        out = self._bits[self._position:stop]
        self._position = stop
        return out


def _required_precision(bit_length: int, margin: int) -> int:
    return int(ceil(bit_length * log10(2.0))) + margin


def _decimal_mod(value: Decimal, modulus: Decimal) -> Decimal:
    return value - modulus * (value / modulus).to_integral_value(rounding=ROUND_FLOOR)


def _round_half_down(value: Decimal) -> Decimal:
    return value.to_integral_value(rounding=ROUND_HALF_DOWN)


def _partition(reference: ReferenceDistribution) -> tuple[np.ndarray, np.ndarray]:
    order = np.asarray(reference.token_order, dtype=np.int64)
    support = reference.probabilities[order].astype(np.float64, copy=True)
    boundaries = np.empty(order.size + 1, dtype=np.float64)
    boundaries[0] = 0.0
    np.cumsum(support, dtype=np.float64, out=boundaries[1:])
    boundaries[1:] /= boundaries[-1]
    return order, boundaries


def _rotate(point: Decimal, left: Decimal, width: Decimal, offset: Decimal) -> Decimal:
    return left + _decimal_mod((point - left) + offset * width, width)


def _unrotate(midpoint: Decimal, left: Decimal, width: Decimal, offset: Decimal) -> Decimal:
    return left + _decimal_mod(midpoint - left - offset * width, width)


def _select(
    boundaries: np.ndarray,
    left: Decimal,
    right: Decimal,
    point: Decimal,
) -> tuple[int, Decimal, Decimal]:
    width = right - left
    lo, hi = 0, len(boundaries) - 2
    while lo < hi:
        mid = (lo + hi) // 2
        upper = left + Decimal(str(boundaries[mid + 1])) * width
        if upper > point:
            hi = mid
        else:
            lo = mid + 1
    lower = left + Decimal(str(boundaries[lo])) * width
    upper = left + Decimal(str(boundaries[lo + 1])) * width
    if not upper > point:
        raise RuntimeError("source oracle could not locate the rotated point")
    return lo, lower, upper


def _narrow(
    boundaries: np.ndarray,
    left: Decimal,
    right: Decimal,
    index: int,
) -> tuple[Decimal, Decimal]:
    width = right - left
    return (
        left + Decimal(str(boundaries[index])) * width,
        left + Decimal(str(boundaries[index + 1])) * width,
    )


def _terminated(left: Decimal, right: Decimal, point: Decimal) -> bool:
    diff = (left + right) / 2 - point
    return Decimal("-0.5") < diff <= Decimal("0.5")


def _replay(
    left: Decimal,
    right: Decimal,
    history: Sequence[tuple[Decimal, Decimal, Decimal]],
) -> int:
    midpoint = (left + right) / 2
    for previous_left, previous_width, offset in reversed(history):
        midpoint = _unrotate(midpoint, previous_left, previous_width, offset)
    return int(_round_half_down(midpoint))


def reference_run(fixture: RRCFixture) -> RRCOutcome:
    """Run the independent pinned-source oracle over final P_reference rows."""

    precision = _required_precision(fixture.payload_bits, fixture.precision_margin)
    rng = random.Random(fixture.seed)
    secret_text = "".join(str(bit) for bit in fixture.secret_bits)
    secret_integer = int(secret_text, 2)

    with localcontext() as ctx:
        ctx.prec = precision
        point = Decimal(secret_integer)
        left = Decimal(0)
        right = Decimal(2) ** fixture.payload_bits
        history: list[tuple[Decimal, Decimal, Decimal]] = []
        tokens: list[int] = []
        retries = 0

        for step in range(fixture.max_tokens):
            reference = fixture.reference_at(step)
            token_ids, boundaries = _partition(reference)
            width = right - left
            offset = Decimal(rng.random())
            history.append((left, width, offset))
            point = _rotate(point, left, width, offset)
            index, left, right = _select(boundaries, left, right, point)
            tokens.append(int(token_ids[index]))

            if _terminated(left, right, point):
                if not fixture.verify_termination or _replay(left, right, history) == secret_integer:
                    break
                retries += 1
        else:
            raise RuntimeError(
                f"source oracle did not terminate within {fixture.max_tokens} tokens"
            )

        # Algorithm 4: replay the observed tokens with an independent PRNG.
        decode_rng = random.Random(fixture.seed)
        decode_left = Decimal(0)
        decode_right = Decimal(2) ** fixture.payload_bits
        decode_history: list[tuple[Decimal, Decimal, Decimal]] = []
        for step, token_id in enumerate(tokens):
            reference = fixture.reference_at(step)
            token_ids, boundaries = _partition(reference)
            hits = np.flatnonzero(token_ids == token_id)
            if not hits.size:
                raise RuntimeError("source oracle token left P_reference support")
            width = decode_right - decode_left
            decode_history.append((decode_left, width, Decimal(decode_rng.random())))
            decode_left, decode_right = _narrow(
                boundaries, decode_left, decode_right, int(hits[0])
            )

        recovered_integer = _replay(decode_left, decode_right, decode_history)
        recovered_text = bin(recovered_integer)[2:].zfill(fixture.payload_bits)
        recovered = tuple(int(bit) for bit in recovered_text)

    return RRCOutcome(
        token_ids=tuple(tokens),
        recovered_bits=recovered,
        precision_digits=precision,
        termination_retries=retries,
    )


def normalized_run(fixture: RRCFixture) -> RRCOutcome:
    environment = MethodEnvironment(
        output_vocab_size=fixture.vocab_size,
        allowed_token_ids=tuple(range(fixture.vocab_size)),
    )
    config = {
        "precision_margin": fixture.precision_margin,
        "verify_termination": fixture.verify_termination,
    }
    encoder = RRCMethod().create_encoder(
        config=config,
        environment=environment,
        secret_source=FixedSecretSource(fixture.secret_bits),
        random_source=MethodRandomSource(fixture.seed),
        target_payload_bits=fixture.payload_bits,
    )

    tokens: list[int] = []
    for step in range(fixture.max_tokens):
        decision = encoder.step(StepContext(step, fixture.reference_at(step)))
        tokens.append(int(decision.token_id))
        if encoder.done:
            break
    else:
        raise RuntimeError(
            f"normalized RRC did not terminate within {fixture.max_tokens} tokens"
        )
    final = encoder.finalize()

    decoder = RRCMethod().create_decoder(
        config=config,
        environment=environment,
        random_source=MethodRandomSource(fixture.seed),
        expected_payload_bits=fixture.payload_bits,
    )
    for step, token_id in enumerate(tokens):
        decoder.observe(StepContext(step, fixture.reference_at(step)), token_id)
    recovered = decoder.finalize()
    if not recovered.complete:
        raise RuntimeError(f"normalized RRC decoder failed: {recovered.metadata}")

    return RRCOutcome(
        token_ids=tuple(tokens),
        recovered_bits=tuple(recovered.recovered_bits),
        precision_digits=int(final.metadata["precision_digits"]),
        termination_retries=int(final.metadata["termination_retries"]),
    )


class AuthorDistribution:
    """Duck-typed distribution consumed by the original ``rrc_core.py``."""

    def __init__(self, reference: ReferenceDistribution) -> None:
        self.token_ids, self.boundaries = _partition(reference)
        probs = reference.probabilities.astype(np.float64)
        positive = probs[probs > 0.0]
        self.entropy = float(-np.sum(positive * np.log2(positive), dtype=np.float64))

    def index_of(self, token_id: int) -> int:
        hits = np.flatnonzero(self.token_ids == int(token_id))
        return int(hits[0]) if hits.size else -1


class AuthorFixtureStepper:
    """Fake LM stepper so original author encode/decode can run without a model."""

    def __init__(self, fixture: RRCFixture) -> None:
        self.fixture = fixture
        self.step_index = 0

    def prefill(self, context_tokens) -> None:  # noqa: ANN001 - author API
        del context_tokens
        self.step_index = 0

    def distribution(self, top_k: int) -> AuthorDistribution:
        del top_k
        return AuthorDistribution(self.fixture.reference_at(self.step_index))

    def step(self, token_id) -> None:  # noqa: ANN001 - author API
        del token_id
        self.step_index += 1


def fixture_suite(random_cases: int = 64) -> tuple[RRCFixture, ...]:
    fixed: list[RRCFixture] = [
        RRCFixture(
            "stationary_8bit",
            ((0.4, 0.3, 0.2, 0.1),),
            (1, 0, 0, 0, 0, 0, 0, 1),
            42,
        ),
        RRCFixture(
            "varying_16bit",
            (
                (0.52, 0.23, 0.15, 0.10),
                (0.10, 0.20, 0.30, 0.40),
                (0.05, 0.15, 0.25, 0.55),
            ),
            tuple(int(x) for x in "1011010010110110"),
            7,
        ),
        RRCFixture(
            "sparse_support_24bit",
            (
                (0.50, 0.0, 0.25, 0.15, 0.10, 0.0),
                (0.05, 0.0, 0.15, 0.30, 0.50, 0.0),
            ),
            tuple(int(x) for x in "110010101111000110100101"),
            1234,
        ),
        RRCFixture(
            "balanced_32bit",
            ((0.25, 0.25, 0.25, 0.25),),
            tuple(int(x) for x in "01101001110010110100111000110101"),
            2026,
        ),
    ]

    rng = random.Random(20261009)
    for case_index in range(random_cases):
        vocab_size = rng.randint(3, 12)
        rows: list[tuple[float, ...]] = []
        for row_index in range(rng.randint(1, 4)):
            weights = np.asarray(
                [rng.randint(20, 1000) for _ in range(vocab_size)], dtype=np.float64
            )
            if vocab_size >= 5 and (case_index + row_index) % 5 == 0:
                weights[rng.randrange(vocab_size)] = 0.0
            weights /= np.sum(weights, dtype=np.float64)
            rows.append(tuple(float(x) for x in weights))
        bit_length = rng.randint(4, 40)
        bits = tuple(rng.randrange(2) for _ in range(bit_length))
        fixed.append(
            RRCFixture(
                fixture_id=f"random_{case_index:03d}",
                probabilities=tuple(rows),
                secret_bits=bits,
                seed=rng.randrange(1, 2**31),
            )
        )
    return tuple(fixed)
