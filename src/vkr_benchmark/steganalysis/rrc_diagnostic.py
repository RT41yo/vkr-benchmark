"""CPU-testable, non-mutating RRC replay with interval-width checkpoints.

The replay uses the original normalized encoder and pinned P_reference. It
inspects private Decimal boundaries read-only; it does *not* patch method
code, alter the token limit, or serialize secret material/partial text.
"""
from __future__ import annotations

import hashlib
from decimal import Decimal, localcontext
from typing import Any

from vkr_benchmark.config import ExperimentConfig
from vkr_benchmark.distributions import ReferenceDistributionBuilder, StepContext
from vkr_benchmark.errors import ConfigurationError, ContractError
from vkr_benchmark.inputs import create_secret_source
from vkr_benchmark.lm import LMAdapter
from vkr_benchmark.randomness import RecordingSecretSource
from vkr_benchmark.runner.experiment import create_method_runtime
from vkr_benchmark.runner.streaming import method_environment_from_builder

CHECKPOINTS = frozenset({1, 64, 128, 256, 512, 1024, 1536, 2048})


def _interval_snapshot(encoder: Any, step: int, candidates: int,
                       rejections: int) -> dict[str, Any]:
    try:
        left = encoder._left
        right = encoder._right
        secret_point = encoder._secret_point
        width = right - left
    except (AttributeError, TypeError) as exc:
        raise ContractError('pinned RRC encoder lacks Decimal interval boundaries') from exc
    if not all(isinstance(v, Decimal) and v.is_finite()
               for v in (left, right, secret_point)) or width <= 0:
        raise ContractError('invalid RRC diagnostic interval')
    with localcontext() as ctx:
        ctx.prec = 32
        log2_width = float(width.ln() / Decimal(2).ln())
    return {
        'step': step, 'interval_log2_width': round(log2_width, 6),
        'interval_width_below_one': bool(width < 1),
        'interval_width_at_most_one': bool(width <= 1),
        'termination_candidates': candidates,
        'verification_rejections': rejections,
    }


def replay_rrc(
    *, config: ExperimentConfig, lm_adapter: LMAdapter, prompt_text: str,
    reference_builder: ReferenceDistributionBuilder,
    checkpoints: frozenset[int] = CHECKPOINTS,
) -> dict[str, Any]:
    """Execute a *fresh* matching RRC sender, retaining only safe diagnostics."""
    if (config.method.method_id != 'rrc' or
            config.termination.mode != 'fixed_payload_bits' or
            config.termination.target_payload_bits != 1024 or
            config.termination.max_carrier_tokens != 2048 or
            dict(config.method.params) !=
            {'precision_margin': 40, 'verify_termination': True}):
        raise ConfigurationError('diagnostic replay is restricted to frozen Stage-7 RRC point')
    if not prompt_text or reference_builder.token_space != lm_adapter.token_space or \
            reference_builder.policy != config.generation.to_policy():
        raise ConfigurationError('RRC replay prompt, LM or reference policy mismatch')

    runtime = create_method_runtime(config)
    environment = method_environment_from_builder(reference_builder)
    source = RecordingSecretSource(create_secret_source(config.secret_id))
    encoder = runtime.method.create_encoder(
        config=config.method.params, environment=environment,
        secret_source=source, random_source=runtime.encoder_random_source,
        key=config.method.key, target_payload_bits=1024,
    )
    prompt_ids = lm_adapter.encode_prompt(prompt_text)
    state = lm_adapter.prefill(prompt_ids)
    digest = hashlib.sha256()
    n = candidates = rejections = 0
    snapshots = []
    for step in range(2048):
        raw, state = lm_adapter.next_logits(state)
        reference = reference_builder.build(raw)
        decision = encoder.step(StepContext(step_index=step, reference=reference))
        token_id = int(decision.token_id)
        if not environment.is_allowed(token_id):
            raise ContractError('diagnostic RRC emitted forbidden token')
        digest.update(token_id.to_bytes(8, 'big', signed=False))
        n += 1
        trace = decision.method_trace
        if trace.get('termination_test') is True:
            candidates += 1
            if trace.get('termination_verified') is False:
                rejections += 1
        if n in checkpoints or encoder.done or n == 2048:
            snapshots.append(_interval_snapshot(encoder, n, candidates, rejections))
        if encoder.done or n == 2048:
            break
        _, state = lm_adapter.next_logits(state, token_id)
    return {
        'terminated': bool(encoder.done),
        'generated_carrier_tokens': n,
        'termination_candidates': candidates,
        'verification_rejections': rejections,
        'token_ids_sha256': digest.hexdigest(),
        'snapshots': snapshots,
    }
