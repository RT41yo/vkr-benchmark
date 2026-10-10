#!/usr/bin/env python3
"""Stage-4 technical compatibility matrix for all benchmark methods.

This is deliberately a *compatibility* smoke test, not a comparative benchmark.
It runs the same prompt/secret through every method on each selected model and
checks:

1. normalized token-space construction;
2. encoder completion;
3. direct token-ID secret recovery;
4. ordinary-text token round-trip;
5. ordinary-text secret recovery.

Expensive distribution instrumentation (explicit Q, KL/TVD, entropy/NLL) is
intentionally disabled here. Those metrics are validated by the method-specific
reproducibility probes and belong to later benchmark experiments, not to this
matrix.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vkr_benchmark.config import LocalModelConfig
from vkr_benchmark.distributions import GenerationPolicy, ReferenceDistributionBuilder, StepContext
from vkr_benchmark.errors import ContractError
from vkr_benchmark.lm import HFCausalLMAdapter, LMAdapter
from vkr_benchmark.methods import (
    ADGMethod,
    ArithmeticMethod,
    BinsMethod,
    DAIRMethod,
    DiscopMethod,
    HuffmanMethod,
    MethodEnvironment,
    RRCMethod,
    StegoMethod,
)
from vkr_benchmark.methods.base import KeyMaterial
from vkr_benchmark.randomness import (
    MethodRandomSource,
    RandomSource,
    RecordingSecretSource,
    Shake256SecretSource,
)
from vkr_benchmark.runner import method_environment_from_builder
from vkr_benchmark.transport import TextChannel


@dataclass(frozen=True, slots=True)
class MethodCase:
    name: str
    method_id: str
    method: StegoMethod
    config: Mapping[str, Any]
    encoder_rng_factory: Callable[[], RandomSource | None]
    decoder_rng_factory: Callable[[], RandomSource | None]
    key: KeyMaterial = None
    fixed_payload: bool = False


@dataclass(frozen=True, slots=True)
class EncodeLite:
    carrier_token_ids: tuple[int, ...]
    payload_secret_bits: tuple[int, ...]
    payload_bits: int
    secret_bits_read: int


@dataclass(frozen=True, slots=True)
class DecodeLite:
    recovered_bits: tuple[int, ...]
    complete: bool


@dataclass(frozen=True, slots=True)
class CompatibilityRow:
    method: str
    method_id: str
    model: str
    revision: str
    tokenizer_size: int
    model_output_vocab: int
    output_only_ids_excluded: int
    special_ids_excluded: int
    allowed_token_ids: int
    carrier_tokens: int | None
    payload_bits: int | None
    bits_per_token: float | None
    direct_secret_roundtrip_exact: bool
    text_token_roundtrip_exact: bool
    text_secret_roundtrip_exact: bool
    status: str
    adaptation: str | None
    error: str | None


def _none_rng() -> RandomSource | None:
    return None


def _rng(seed: int) -> Callable[[], RandomSource]:
    return lambda: MethodRandomSource(seed)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _advance_state(lm: LMAdapter, state: Any, token_id: int) -> Any:
    _, updated = lm.next_logits(state, token_id)
    return updated


def _encode_without_metrics(
    *,
    lm: LMAdapter,
    builder: ReferenceDistributionBuilder,
    environment: MethodEnvironment,
    method_case: MethodCase,
    prompt: str,
    secret_id: str,
    fixed_carrier_tokens: int,
    rrc_payload_bits: int,
    rrc_max_carrier_tokens: int,
) -> EncodeLite:
    target_payload_bits = rrc_payload_bits if method_case.fixed_payload else None
    carrier_limit = rrc_max_carrier_tokens if method_case.fixed_payload else fixed_carrier_tokens

    secret = RecordingSecretSource(Shake256SecretSource(secret_id))
    encoder = method_case.method.create_encoder(
        config=method_case.config,
        environment=environment,
        secret_source=secret,
        random_source=method_case.encoder_rng_factory(),
        key=method_case.key,
        target_payload_bits=target_payload_bits,
    )

    prompt_ids = lm.encode_prompt(prompt)
    state = lm.prefill(prompt_ids)
    generated: list[int] = []

    for step_index in range(carrier_limit):
        raw_logits, state = lm.next_logits(state)
        reference = builder.build(raw_logits)
        decision = encoder.step(StepContext(step_index=step_index, reference=reference))
        token_id = int(decision.token_id)
        if not environment.is_allowed(token_id):
            raise ContractError(f"method emitted token {token_id} outside V_allowed")
        generated.append(token_id)

        if encoder.done or step_index + 1 >= carrier_limit:
            break
        state = _advance_state(lm, state, token_id)

    if method_case.fixed_payload and not encoder.done:
        raise ContractError(
            f"fixed-payload method did not finish {rrc_payload_bits} bits within "
            f"{rrc_max_carrier_tokens} carrier tokens"
        )

    finalization = encoder.finalize()
    if method_case.fixed_payload and finalization.payload_bits != rrc_payload_bits:
        raise ContractError(
            "fixed-payload finalization disagrees with requested payload: "
            f"{finalization.payload_bits} != {rrc_payload_bits}"
        )
    if finalization.payload_bits > len(secret.consumed_bits):
        raise ContractError("method reports more payload bits than secret bits read")

    return EncodeLite(
        carrier_token_ids=tuple(generated),
        payload_secret_bits=secret.consumed_bits[: finalization.payload_bits],
        payload_bits=int(finalization.payload_bits),
        secret_bits_read=len(secret.consumed_bits),
    )


def _decode_without_metrics(
    *,
    lm: LMAdapter,
    builder: ReferenceDistributionBuilder,
    environment: MethodEnvironment,
    method_case: MethodCase,
    prompt: str,
    observed_token_ids: tuple[int, ...],
    expected_payload_bits: int,
) -> DecodeLite:
    decoder = method_case.method.create_decoder(
        config=method_case.config,
        environment=environment,
        random_source=method_case.decoder_rng_factory(),
        key=method_case.key,
        expected_payload_bits=expected_payload_bits,
    )

    prompt_ids = lm.encode_prompt(prompt)
    state = lm.prefill(prompt_ids)
    for step_index, token_id in enumerate(observed_token_ids):
        raw_logits, state = lm.next_logits(state)
        reference = builder.build(raw_logits)
        decoder.observe(
            StepContext(step_index=step_index, reference=reference),
            int(token_id),
        )
        if step_index + 1 < len(observed_token_ids):
            state = _advance_state(lm, state, int(token_id))

    finalization = decoder.finalize()
    return DecodeLite(
        recovered_bits=tuple(finalization.recovered_bits),
        complete=bool(finalization.complete),
    )


def classify_status(
    *,
    direct_secret_exact: bool,
    text_token_exact: bool,
    text_secret_exact: bool,
    output_only_count: int,
) -> str:
    """Classify one method/model cell under the frozen Stage-4 rule."""

    if not (direct_secret_exact and text_token_exact and text_secret_exact):
        return "FAIL"
    if output_only_count > 0:
        return "PASS_WITH_ADAPTATION"
    return "PASS"


def _method_cases(raw: Mapping[str, Any]) -> list[MethodCase]:
    bins = raw["bins"]
    huffman = raw["huffman"]
    arithmetic = raw["arithmetic"]
    discop = raw["discop"]
    rrc = raw["rrc"]
    adg = raw["adg"]
    dair = raw["dairstega"]

    bins_seed = int(bins["method_seed"])
    discop_key = int(discop["key"])
    rrc_key = int(rrc["key"])
    adg_seed = int(adg["method_seed"])

    return [
        MethodCase(
            name="Bins",
            method_id="bins",
            method=BinsMethod(),
            config={"block_size": int(bins["block_size"])},
            encoder_rng_factory=_rng(bins_seed),
            decoder_rng_factory=_rng(bins_seed),
        ),
        MethodCase(
            name="Huffman",
            method_id="huffman",
            method=HuffmanMethod(),
            config={"bits_per_word": int(huffman["bits_per_word"])},
            encoder_rng_factory=_none_rng,
            decoder_rng_factory=_none_rng,
        ),
        MethodCase(
            name="Arithmetic Coding",
            method_id="arithmetic",
            method=ArithmeticMethod(),
            config={
                "precision": int(arithmetic["precision"]),
                "top_k": int(arithmetic["top_k"]),
            },
            encoder_rng_factory=_none_rng,
            decoder_rng_factory=_none_rng,
        ),
        MethodCase(
            name="Discop",
            method_id="discop",
            method=DiscopMethod(),
            config={},
            encoder_rng_factory=_rng(discop_key),
            decoder_rng_factory=_rng(discop_key),
            key=discop_key,
        ),
        MethodCase(
            name="RRC",
            method_id="rrc",
            method=RRCMethod(),
            config={
                "precision_margin": int(rrc["precision_margin"]),
                "verify_termination": bool(rrc["verify_termination"]),
            },
            encoder_rng_factory=_rng(rrc_key),
            decoder_rng_factory=_rng(rrc_key),
            key=rrc_key,
            fixed_payload=True,
        ),
        MethodCase(
            name="ADG",
            method_id="adg",
            method=ADGMethod(),
            config={},
            encoder_rng_factory=_rng(adg_seed),
            decoder_rng_factory=_none_rng,
        ),
        MethodCase(
            name="DAIRstega",
            method_id="dairstega",
            method=DAIRMethod(),
            config={
                "alpha": int(dair["alpha"]),
                "beta": float(dair["beta"]),
                "top_k": int(dair["top_k"]),
            },
            encoder_rng_factory=_none_rng,
            decoder_rng_factory=_none_rng,
        ),
    ]


def _adaptation_label(output_only_count: int) -> str | None:
    if output_only_count <= 0:
        return None
    return f"exclude_and_renormalize_{output_only_count}_output_only_ids"


def _failed_row(
    *,
    case: MethodCase,
    config: LocalModelConfig,
    lm: LMAdapter,
    builder: ReferenceDistributionBuilder,
    error: Exception,
) -> CompatibilityRow:
    token_space = lm.token_space
    special_in_output = sum(
        1
        for token_id in token_space.special_token_ids
        if 0 <= token_id < token_space.output_vocab_size
        and token_id in token_space.valid_tokenizer_ids
    )
    return CompatibilityRow(
        method=case.name,
        method_id=case.method_id,
        model=config.model_id,
        revision=config.revision,
        tokenizer_size=token_space.tokenizer_vocab_size,
        model_output_vocab=token_space.output_vocab_size,
        output_only_ids_excluded=token_space.output_only_count,
        special_ids_excluded=special_in_output,
        allowed_token_ids=int(builder.allowed_mask.sum()),
        carrier_tokens=None,
        payload_bits=None,
        bits_per_token=None,
        direct_secret_roundtrip_exact=False,
        text_token_roundtrip_exact=False,
        text_secret_roundtrip_exact=False,
        status="FAIL",
        adaptation=_adaptation_label(token_space.output_only_count),
        error=f"{type(error).__name__}: {error}",
    )


def run_case(
    *,
    lm: LMAdapter,
    config: LocalModelConfig,
    builder: ReferenceDistributionBuilder,
    environment: MethodEnvironment,
    case: MethodCase,
    prompt: str,
    secret_id: str,
    fixed_carrier_tokens: int,
    rrc_payload_bits: int,
    rrc_max_carrier_tokens: int,
) -> CompatibilityRow:
    token_space = lm.token_space
    special_in_output = sum(
        1
        for token_id in token_space.special_token_ids
        if 0 <= token_id < token_space.output_vocab_size
        and token_id in token_space.valid_tokenizer_ids
    )

    try:
        encoded = _encode_without_metrics(
            lm=lm,
            builder=builder,
            environment=environment,
            method_case=case,
            prompt=prompt,
            secret_id=secret_id,
            fixed_carrier_tokens=fixed_carrier_tokens,
            rrc_payload_bits=rrc_payload_bits,
            rrc_max_carrier_tokens=rrc_max_carrier_tokens,
        )

        direct = _decode_without_metrics(
            lm=lm,
            builder=builder,
            environment=environment,
            method_case=case,
            prompt=prompt,
            observed_token_ids=encoded.carrier_token_ids,
            expected_payload_bits=encoded.payload_bits,
        )
        direct_exact = direct.complete and direct.recovered_bits == encoded.payload_secret_bits

        transport = TextChannel(lm).transmit(encoded.carrier_token_ids)
        text_exact = False
        text_error: str | None = None
        try:
            text_decoded = _decode_without_metrics(
                lm=lm,
                builder=builder,
                environment=environment,
                method_case=case,
                prompt=prompt,
                observed_token_ids=transport.receiver_token_ids,
                expected_payload_bits=encoded.payload_bits,
            )
            text_exact = (
                text_decoded.complete
                and text_decoded.recovered_bits == encoded.payload_secret_bits
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic matrix records failure
            text_error = f"{type(exc).__name__}: {exc}"

        status = classify_status(
            direct_secret_exact=direct_exact,
            text_token_exact=transport.token_sequence_roundtrip_exact,
            text_secret_exact=text_exact,
            output_only_count=token_space.output_only_count,
        )
        bpt = (
            encoded.payload_bits / len(encoded.carrier_token_ids)
            if encoded.carrier_token_ids
            else None
        )
        return CompatibilityRow(
            method=case.name,
            method_id=case.method_id,
            model=config.model_id,
            revision=config.revision,
            tokenizer_size=token_space.tokenizer_vocab_size,
            model_output_vocab=token_space.output_vocab_size,
            output_only_ids_excluded=token_space.output_only_count,
            special_ids_excluded=special_in_output,
            allowed_token_ids=int(builder.allowed_mask.sum()),
            carrier_tokens=len(encoded.carrier_token_ids),
            payload_bits=encoded.payload_bits,
            bits_per_token=bpt,
            direct_secret_roundtrip_exact=direct_exact,
            text_token_roundtrip_exact=transport.token_sequence_roundtrip_exact,
            text_secret_roundtrip_exact=text_exact,
            status=status,
            adaptation=_adaptation_label(token_space.output_only_count),
            error=text_error,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic matrix records failure
        return _failed_row(
            case=case,
            config=config,
            lm=lm,
            builder=builder,
            error=exc,
        )


def render_markdown(rows: list[CompatibilityRow]) -> str:
    model_order: list[str] = []
    method_order: list[str] = []
    for row in rows:
        if row.model not in model_order:
            model_order.append(row.model)
        if row.method not in method_order:
            method_order.append(row.method)

    by_key = {(row.method, row.model): row for row in rows}
    lines = [
        "# Stage 4 — матрица совместимости методов и моделей",
        "",
        "Статусы отражают короткую техническую проверку совместимости, а не сравнительное качество методов.",
        "`PASS_WITH_ADAPTATION` означает, что общая инфраструктура исключила output ID модели без отображения в токенизаторе и перенормировала P_reference.",
        "",
        "| Метод | " + " | ".join(model_order) + " |",
        "|---|" + "|".join("---" for _ in model_order) + "|",
    ]
    for method in method_order:
        cells = [by_key[(method, model)].status for model in model_order]
        lines.append("| " + method + " | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Диагностика",
            "",
            "| Метод | Модель | direct secret RT | text token RT | text secret RT | carrier | payload | adaptation | error |",
            "|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.method,
                    row.model,
                    str(row.direct_secret_roundtrip_exact),
                    str(row.text_token_roundtrip_exact),
                    str(row.text_secret_roundtrip_exact),
                    "-" if row.carrier_tokens is None else str(row.carrier_tokens),
                    "-" if row.payload_bits is None else str(row.payload_bits),
                    row.adaptation or "-",
                    (row.error or "-").replace("|", "\\|"),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/reproducibility/stage4_compatibility_matrix.json",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPO_ROOT / "results/stage4/compatibility_matrix.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPO_ROOT / "results/stage4/compatibility_matrix.md",
    )
    args = parser.parse_args()

    raw = _load_json(args.config.resolve())
    prompt = str(raw["prompt"])
    secret_id = str(raw["secret_id"])
    fixed_carrier_tokens = int(raw["fixed_carrier_tokens"])
    rrc_payload_bits = int(raw["rrc_payload_bits"])
    rrc_max_carrier_tokens = int(raw["rrc_max_carrier_tokens"])
    common = raw["common_generation"]
    method_cases = _method_cases(raw["methods"])

    print("Stage 4 method/model/tokenizer compatibility matrix")
    print("prompt:", prompt)
    print("secret_id:", secret_id)
    print("fixed-carrier tokens:", fixed_carrier_tokens)
    print("RRC payload bits:", rrc_payload_bits)
    print("RRC max carrier tokens:", rrc_max_carrier_tokens)
    print("common top-k:", common["top_k"])
    print("common top-p:", common["top_p"])
    print("metric instrumentation: DISABLED (compatibility-only run)")

    rows: list[CompatibilityRow] = []
    for model_path_raw in raw["models"]:
        model_path = (REPO_ROOT / str(model_path_raw)).resolve()
        config = LocalModelConfig.from_json(model_path)
        print()
        print("=" * 100)
        print("MODEL:", config.model_id)
        print("REVISION:", config.revision)
        print("=" * 100)
        print("Loading local model...")

        lm = HFCausalLMAdapter.from_local_config(config)
        builder = ReferenceDistributionBuilder(
            token_space=lm.token_space,
            policy=GenerationPolicy(
                top_k=common["top_k"],
                top_p=float(common["top_p"]),
            ),
        )
        environment = method_environment_from_builder(builder)
        print("model output vocab:", lm.token_space.output_vocab_size)
        print("tokenizer size:", lm.token_space.tokenizer_vocab_size)
        print("output-only IDs excluded:", lm.token_space.output_only_count)
        print("V_allowed size:", len(environment.allowed_token_ids))

        for case in method_cases:
            row = run_case(
                lm=lm,
                config=config,
                builder=builder,
                environment=environment,
                case=case,
                prompt=prompt,
                secret_id=secret_id,
                fixed_carrier_tokens=fixed_carrier_tokens,
                rrc_payload_bits=rrc_payload_bits,
                rrc_max_carrier_tokens=rrc_max_carrier_tokens,
            )
            rows.append(row)
            carrier = "-" if row.carrier_tokens is None else str(row.carrier_tokens)
            payload = "-" if row.payload_bits is None else str(row.payload_bits)
            print(
                f"{row.method:18s} {row.status:20s} "
                f"carrier={carrier:>3s} payload={payload:>3s} "
                f"directRT={row.direct_secret_roundtrip_exact} "
                f"textTokenRT={row.text_token_roundtrip_exact} "
                f"textSecretRT={row.text_secret_roundtrip_exact}"
            )
            if row.error:
                print("  diagnostic:", row.error)

        del builder
        del environment
        del lm
        _release_cuda_memory()

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "stage4-compatibility-matrix-v1",
        "prompt": prompt,
        "secret_id": secret_id,
        "fixed_carrier_tokens": fixed_carrier_tokens,
        "rrc_payload_bits": rrc_payload_bits,
        "rrc_max_carrier_tokens": rrc_max_carrier_tokens,
        "common_generation": common,
        "metric_instrumentation": False,
        "rows": [asdict(row) for row in rows],
    }
    args.json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(render_markdown(rows), encoding="utf-8")

    print()
    print("=" * 100)
    print("MATRIX")
    print("=" * 100)
    print(render_markdown(rows))
    print("json:", args.json_output)
    print("markdown:", args.markdown_output)
    print("status:", "PASS" if all(row.status != "FAIL" for row in rows) else "FAIL")


if __name__ == "__main__":
    main()
