"""Normalized Adaptive Dynamic Grouping (ADG) steganography.

Publication reference:
    Zhang et al., "Provably Secure Generative Linguistic Steganography",
    Findings of ACL-IJCNLP 2021, Algorithm 1.

Repository reference:
    Mhzzzzz/ADG-steganography,
    commit b4a7e802a97bc3a3b84d07e63f24627b16c51a10.

The normalized adapter preserves the publication-level ADG semantics.  At each
recursive level it chooses the largest power-of-two number of groups compatible
with the largest normalized token probability, constructs approximately equal
probability groups, lets little-endian secret bits choose one group, and
recurses inside that group.  When the selected group can no longer be divided
(max probability > 0.5), one token is sampled proportionally to P_reference
inside that terminal group.

Important reference discrepancy: the public repository's ``near`` helper uses
``anum - up`` in its final nearest-neighbour comparison.  Algorithm 1 says to
choose the token whose *probability* is nearest to the residual.  Normalized
mode follows the paper and uses ``anum - alist[up]`` semantics rather than
silently reproducing the apparent repository typo.  Stage-4 conformance tooling
characterizes this discrepancy explicitly.

ADG has no method-local tuning parameter in normalized Stage 4.  The injected
MethodRandomSource is used only for the terminal categorical sample, making the
run reproducible without exposing RNG details to the decoder.  Common
LM-generation policy remains benchmark infrastructure and is already reflected
in P_reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from vkr_benchmark.distributions import (
    DistributionInfo,
    QMode,
    QSource,
    StepContext,
    explicit_adg_q,
)
from vkr_benchmark.errors import ConfigurationError, MethodError, SessionStateError
from vkr_benchmark.methods.base import (
    DecoderSession,
    EncoderSession,
    KeyMaterial,
    MethodConfig,
    StegoMethod,
)
from vkr_benchmark.methods.types import (
    DecodeProgress,
    DecoderFinalization,
    EncodeDecision,
    EncoderFinalization,
    MethodEnvironment,
)
from vkr_benchmark.randomness import RandomSource, SecretSource

_REFERENCE_COMMIT = "b4a7e802a97bc3a3b84d07e63f24627b16c51a10"


@dataclass(frozen=True, slots=True)
class ADGConfig:
    """Normalized ADG exposes no method-local hyperparameter in Stage 4."""

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ADGConfig":
        if values:
            raise ConfigurationError(
                f"unknown ADG configuration fields: {sorted(values)}"
            )
        return cls()


@dataclass(frozen=True, slots=True)
class _Group:
    token_ids: np.ndarray
    probabilities: np.ndarray


def _validate_reference(
    context: StepContext, environment: MethodEnvironment
) -> _Group:
    reference = context.reference
    if reference.vocab_size != environment.output_vocab_size:
        raise MethodError(
            "ADG MethodEnvironment output vocabulary does not match P_reference"
        )
    token_ids = np.asarray(reference.token_order, dtype=np.int64)
    if token_ids.size == 0:
        raise MethodError("ADG requires non-empty P_reference support")
    for token_id in token_ids:
        if not environment.is_allowed(int(token_id)):
            raise MethodError(
                f"P_reference support token {int(token_id)} lies outside V_allowed"
            )
    probabilities = reference.probabilities[token_ids].astype(np.float64, copy=True)
    return _sort_group(token_ids, probabilities)


def _sort_group(token_ids: np.ndarray, probabilities: np.ndarray) -> _Group:
    total = float(np.sum(probabilities, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise MethodError(f"ADG group probability total is invalid: {total!r}")
    normalized = probabilities.astype(np.float64, copy=True) / total
    tokens = token_ids.astype(np.int64, copy=True)
    order = np.lexsort((tokens, -normalized))
    return _Group(token_ids=tokens[order], probabilities=normalized[order])


def _bit_count(max_probability: float) -> int:
    if max_probability > 0.5:
        return 0
    bit = 1
    while (1.0 / (2 ** (bit + 1))) > max_probability:
        bit += 1
    return bit


def _nearest_index_desc(values: list[float], target: float) -> int:
    if not values:
        raise MethodError("ADG nearest search received an empty list")
    if len(values) == 1:
        return 0
    if target >= values[0]:
        return 0
    if target <= values[-1]:
        return len(values) - 1

    bottom = 0
    up = len(values) - 1
    while up - bottom > 1:
        index = (up + bottom) // 2
        value = values[index]
        if value < target:
            up = index
        elif value > target:
            bottom = index
        else:
            while index > 0 and values[index - 1] == target:
                index -= 1
            return index

    if values[bottom] - target < target - values[up]:
        return bottom
    return up


def _partition(group: _Group) -> tuple[_Group, ...] | None:
    bit = _bit_count(float(group.probabilities[0]))
    if bit == 0:
        return None

    group_count = 1 << bit
    remaining_probs = [float(value) for value in group.probabilities]
    remaining_tokens = [int(value) for value in group.token_ids]
    mean = 1.0 / group_count
    result: list[_Group] = []

    for i in range(group_count - 1):
        if not remaining_probs:
            raise MethodError("ADG ran out of candidates while constructing groups")
        selected_probs = [remaining_probs.pop(0)]
        selected_tokens = [remaining_tokens.pop(0)]
        selected_mass = selected_probs[0]

        while selected_mass < mean and remaining_probs:
            delta = mean - selected_mass
            index = _nearest_index_desc(remaining_probs, delta)
            candidate_probability = remaining_probs[index]
            if candidate_probability - delta < delta:
                selected_mass += candidate_probability
                selected_probs.append(remaining_probs.pop(index))
                selected_tokens.append(remaining_tokens.pop(index))
            else:
                break

        result.append(
            _sort_group(
                np.asarray(selected_tokens, dtype=np.int64),
                np.asarray(selected_probs, dtype=np.float64),
            )
        )
        mean = float(np.sum(remaining_probs, dtype=np.float64)) / (
            group_count - i - 1
        )

    if not remaining_probs:
        raise MethodError("ADG final group unexpectedly has no candidates")
    result.append(
        _sort_group(
            np.asarray(remaining_tokens, dtype=np.int64),
            np.asarray(remaining_probs, dtype=np.float64),
        )
    )
    return tuple(result)


def _bits_to_index_little_endian(bits: tuple[int, ...]) -> int:
    return sum(int(bit) << i for i, bit in enumerate(bits))


def _index_to_bits_little_endian(index: int, bit_count: int) -> tuple[int, ...]:
    return tuple((int(index) >> i) & 1 for i in range(bit_count))


def _sample_terminal(group: _Group, random_source: RandomSource) -> int:
    draw = float(random_source.random())
    if not 0.0 <= draw < 1.0:
        raise MethodError("ADG random source returned a value outside [0, 1)")
    cumulative = np.cumsum(group.probabilities, dtype=np.float64)
    index = int(np.searchsorted(cumulative, draw, side="right"))
    if index >= len(group.token_ids):
        index = len(group.token_ids) - 1
    return int(group.token_ids[index])


def _distribution_info(context: StepContext) -> DistributionInfo:
    return DistributionInfo.explicit(
        explicit_adg_q(context.reference),
        mode=QMode.EXACT_ENUMERATION,
        source=QSource.INDEPENDENT_ENUMERATION,
        metadata={
            "method": "adg",
            "reference_commit": _REFERENCE_COMMIT,
            "construction": "independent_full_recursion",
            "grouping_semantics": "paper_algorithm1_nearest_probability",
        },
    )


class ADGEncoderSession(EncoderSession):
    def __init__(
        self,
        *,
        config: ADGConfig,
        environment: MethodEnvironment,
        secret_source: SecretSource,
        random_source: RandomSource,
    ) -> None:
        self._config = config
        self._environment = environment
        self._secret_source = secret_source
        self._random_source = random_source
        self._start_secret_position = secret_source.position
        self._steps = 0
        self._finalized = False

    @property
    def done(self) -> bool:
        return self._finalized

    def step(self, context: StepContext) -> EncodeDecision:
        if self._finalized:
            raise SessionStateError("cannot call ADG encoder.step() after finalize()")

        group = _validate_reference(context, self._environment)
        consumed: list[int] = []
        path: list[int] = []
        level_bits: list[int] = []

        while True:
            children = _partition(group)
            if children is None:
                break
            bit_count = int(np.log2(len(children)))
            bits = self._secret_source.read_bits(bit_count)
            selected = _bits_to_index_little_endian(bits)
            if selected >= len(children):
                raise MethodError("ADG secret bits selected an invalid group index")
            consumed.extend(bits)
            path.append(selected)
            level_bits.append(bit_count)
            group = children[selected]

        token_id = _sample_terminal(group, self._random_source)
        self._steps += 1
        return EncodeDecision(
            token_id=token_id,
            bits_consumed=len(consumed),
            method_trace={
                "group_path": tuple(path),
                "group_bits": tuple(level_bits),
                "recursion_depth": len(path),
                "terminal_support_size": int(len(group.token_ids)),
                "selected_secret_bits": tuple(consumed),
            },
        )

    def distribution_info(
        self, context: StepContext, decision: EncodeDecision
    ) -> DistributionInfo:
        del decision
        return _distribution_info(context)

    def finalize(self) -> EncoderFinalization:
        if self._finalized:
            raise SessionStateError("ADG encoder has already been finalized")
        self._finalized = True
        payload_bits = self._secret_source.position - self._start_secret_position
        return EncoderFinalization(
            payload_bits=payload_bits,
            termination_reason="streaming_session_finalized",
            metadata={
                "steps": self._steps,
                "reference_commit": _REFERENCE_COMMIT,
                "grouping_semantics": "paper_algorithm1_nearest_probability",
            },
        )


class ADGDecoderSession(DecoderSession):
    def __init__(
        self,
        *,
        config: ADGConfig,
        environment: MethodEnvironment,
        expected_payload_bits: int | None,
    ) -> None:
        self._config = config
        self._environment = environment
        self._expected_payload_bits = expected_payload_bits
        self._recovered: list[int] = []
        self._steps = 0
        self._finalized = False

    @property
    def done(self) -> bool:
        if self._finalized:
            return True
        if self._expected_payload_bits is None:
            return False
        return len(self._recovered) >= self._expected_payload_bits

    def observe(self, context: StepContext, observed_token_id: int) -> DecodeProgress:
        if self._finalized:
            raise SessionStateError("cannot call ADG decoder.observe() after finalize()")

        group = _validate_reference(context, self._environment)
        recovered: list[int] = []
        path: list[int] = []
        level_bits: list[int] = []

        while True:
            children = _partition(group)
            if children is None:
                if int(observed_token_id) not in set(map(int, group.token_ids)):
                    raise MethodError(
                        f"observed token {int(observed_token_id)} is outside the ADG terminal group"
                    )
                break

            selected_index = -1
            for index, child in enumerate(children):
                if np.any(child.token_ids == int(observed_token_id)):
                    selected_index = index
                    group = child
                    break
            if selected_index < 0:
                raise MethodError(
                    f"observed token {int(observed_token_id)} is outside current ADG grouping"
                )
            bit_count = int(np.log2(len(children)))
            bits = _index_to_bits_little_endian(selected_index, bit_count)
            recovered.extend(bits)
            path.append(selected_index)
            level_bits.append(bit_count)

        self._recovered.extend(recovered)
        self._steps += 1
        return DecodeProgress(
            recovered_bits=tuple(recovered),
            method_trace={
                "group_path": tuple(path),
                "group_bits": tuple(level_bits),
                "recursion_depth": len(path),
                "recovered_step_bits": tuple(recovered),
            },
        )

    def finalize(self) -> DecoderFinalization:
        if self._finalized:
            raise SessionStateError("ADG decoder has already been finalized")
        self._finalized = True
        recovered = tuple(self._recovered)
        complete = True
        if self._expected_payload_bits is not None:
            complete = len(recovered) >= self._expected_payload_bits
            recovered = recovered[: self._expected_payload_bits]
        return DecoderFinalization(
            recovered_bits=recovered,
            complete=complete,
            metadata={
                "steps": self._steps,
                "raw_recovered_bits": len(self._recovered),
                "reference_commit": _REFERENCE_COMMIT,
            },
        )


class ADGMethod(StegoMethod):
    method_id = "adg"

    def create_encoder(
        self,
        *,
        config: MethodConfig,
        environment: MethodEnvironment,
        secret_source: SecretSource,
        random_source: RandomSource | None = None,
        key: KeyMaterial = None,
        target_payload_bits: int | None = None,
    ) -> EncoderSession:
        del key
        if target_payload_bits is not None:
            raise ConfigurationError(
                "normalized ADG currently uses fixed_carrier_tokens, not fixed_payload_bits"
            )
        if random_source is None:
            raise ConfigurationError(
                "normalized ADG requires an injected MethodRandomSource for terminal sampling"
            )
        return ADGEncoderSession(
            config=ADGConfig.from_mapping(config),
            environment=environment,
            secret_source=secret_source,
            random_source=random_source,
        )

    def create_decoder(
        self,
        *,
        config: MethodConfig,
        environment: MethodEnvironment,
        random_source: RandomSource | None = None,
        key: KeyMaterial = None,
        expected_payload_bits: int | None = None,
    ) -> DecoderSession:
        del random_source, key
        return ADGDecoderSession(
            config=ADGConfig.from_mapping(config),
            environment=environment,
            expected_payload_bits=expected_payload_bits,
        )
