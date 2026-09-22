"""Author-compatible one-sentence encoding mirrors for Stage 3.

These mirrors intentionally live outside ``src/vkr_benchmark``.  They reproduce
published Harvard executable semantics closely enough to instrument a stopping
rule that the public encode helpers do not expose: stop immediately after the
first token for which ``utils.is_sent_finish`` is true while a long uniform bit
stream is still available.

The mirrors are accepted only after the GPU runner proves fixed-message parity
against the pinned author functions with ``finish_sent=False``.
"""

from __future__ import annotations

import math
from typing import Any


def _finish(result: dict[str, Any], *, total_log_probs: float, total_kl: float, token_count: int,
            payload_bits: int, secret_bits_read: int, total_entropy: float | None = None) -> dict[str, Any]:
    result["carrier_tokens"] = token_count
    result["payload_bits_confirmed"] = payload_bits
    result["secret_bits_read"] = secret_bits_read
    if token_count > 0:
        result["avg_nll_nats_author"] = -total_log_probs / token_count
        result["kl_q_stego_to_p_lm_bits_author"] = total_kl / token_count
    else:
        result["avg_nll_nats_author"] = None
        result["kl_q_stego_to_p_lm_bits_author"] = None
    result["bits_per_word_author"] = payload_bits / token_count if token_count else None
    result["words_per_bit_author"] = token_count / payload_bits if payload_bits > 0 else None
    if total_entropy is not None:
        result["avg_entropy_p_tau_bits_author_helper"] = total_entropy / token_count if token_count else None
    return result


def encode_bins_mirror(
    *, model: Any, enc: Any, utils: Any, message: list[int], context_tokens: list[int],
    block_size: int, bin2words: Any, device: str, stop_at_first_sentence: bool,
    max_generated_tokens: int | None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    context = torch.tensor(context_tokens[-1022:], device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    i = 0
    total_log_probs = 0.0
    total_kl = 0.0
    token_count = 0
    boundary_positions: list[int] = []
    terminal_reason = "message_exhausted"

    with torch.no_grad():
        while True:
            if stop_at_first_sentence:
                if max_generated_tokens is not None and token_count >= max_generated_tokens:
                    terminal_reason = "max_generated_tokens"
                    break
                if i + block_size > len(message):
                    terminal_reason = "bitstream_exhausted"
                    break
            elif i >= len(message):
                terminal_reason = "message_exhausted"
                break

            logits, past = model(prev.unsqueeze(0), past=past)
            past = utils.limit_past(past)
            logits[0, -1, -1] = -1e10
            logits[0, -1, 628] = -1e10
            logits = logits[0, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)

            logq = logits.clone()
            logq[:] = -1e10
            for bin_val in range(2 ** block_size):
                filtered_logits = logits.clone()
                filtered_logits[:] = -1e10
                available_tokens = bin2words[bin_val]
                filtered_logits[available_tokens] = logits[available_tokens]
                _, indices = filtered_logits.sort(descending=True)
                logq[indices[0]] = -block_size

            logq = logq * 0.69315
            q = torch.exp(logq)

            m_part = message[i:i + block_size]
            filtered_logits = logits.clone()
            filtered_logits[:] = -1e10
            available_tokens = bin2words[utils.bits2int(m_part)]
            filtered_logits[available_tokens] = logits[available_tokens]
            _, indices = filtered_logits.sort(descending=True)
            selected = int(indices[0].item())

            total_kl += float(utils.kl(q, logq, log_probs))
            total_log_probs += float(log_probs[selected].item())
            i += block_size
            token_count += 1
            prev = indices[0].view(1)
            output = torch.cat((output, prev))

            if bool(utils.is_sent_finish(selected, enc)):
                boundary_positions.append(token_count - 1)
                if stop_at_first_sentence:
                    terminal_reason = "sentence_boundary"
                    break

    generated = [int(x) for x in output[len(context):].tolist()]
    return _finish(
        {
            "generated_token_ids": generated,
            "terminal_reason": terminal_reason,
            "sentence_finish_token_positions_zero_based": boundary_positions,
            "used_implicit_zero_lookahead": False,
        },
        total_log_probs=total_log_probs,
        total_kl=total_kl,
        token_count=token_count,
        payload_bits=i,
        secret_bits_read=i,
    )


def encode_huffman_mirror(
    *, model: Any, enc: Any, utils: Any, huffman_module: Any, message: list[int],
    context_tokens: list[int], bits_per_word: int, device: str,
    stop_at_first_sentence: bool, max_generated_tokens: int | None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    context = torch.tensor(context_tokens[-1022:], device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    i = 0
    total_log_probs = 0.0
    total_kl = 0.0
    token_count = 0
    boundary_positions: list[int] = []
    terminal_reason = "message_exhausted"
    used_implicit_zero = False

    with torch.no_grad():
        while True:
            if stop_at_first_sentence:
                if max_generated_tokens is not None and token_count >= max_generated_tokens:
                    terminal_reason = "max_generated_tokens"
                    break
                if i >= len(message):
                    terminal_reason = "bitstream_exhausted"
                    break
            elif i >= len(message):
                terminal_reason = "message_exhausted"
                break

            logits, past = model(prev.unsqueeze(0), past=past)
            past = utils.limit_past(past)
            logits[0, -1, -1] = -1e10
            logits[0, -1, 628] = -1e10
            logits, indices = logits[0, -1, :].sort(descending=True)
            indices = indices[: 2 ** bits_per_word]
            log_probs = F.log_softmax(logits, dim=-1)[: 2 ** bits_per_word]
            probs = torch.exp(log_probs)

            probs_array = probs.cpu().numpy()
            coding = huffman_module.HuffmanCoding()
            coding.make_heap_from_array(probs_array)
            coding.merge_nodes()
            root = coding.make_codes()

            exhausted_mid_code = False
            while root.token is None:
                if i >= len(message):
                    if stop_at_first_sentence:
                        exhausted_mid_code = True
                        break
                    bit = 0
                    used_implicit_zero = True
                else:
                    bit = int(message[i])
                root = root.left if bit == 0 else root.right
                i += 1
            if exhausted_mid_code:
                terminal_reason = "bitstream_exhausted"
                break

            selection = int(root.token)
            logq = torch.tensor(
                [-len(coding.codes[idx]) for idx in range(len(probs_array))],
                dtype=torch.float,
                device=device,
            )
            logq = logq * 0.69315
            q = torch.exp(logq)
            total_kl += float(utils.kl(q, logq, log_probs))
            total_log_probs += float(log_probs[selection].item())
            token_count += 1

            selected = int(indices[selection].item())
            prev = indices[selection].view(1)
            output = torch.cat((output, prev))
            if bool(utils.is_sent_finish(selected, enc)):
                boundary_positions.append(token_count - 1)
                if stop_at_first_sentence:
                    terminal_reason = "sentence_boundary"
                    break

    generated = [int(x) for x in output[len(context):].tolist()]
    return _finish(
        {
            "generated_token_ids": generated,
            "terminal_reason": terminal_reason,
            "sentence_finish_token_positions_zero_based": boundary_positions,
            "used_implicit_zero_lookahead": used_implicit_zero,
        },
        total_log_probs=total_log_probs,
        total_kl=total_kl,
        token_count=token_count,
        payload_bits=i,
        secret_bits_read=min(i, len(message)) if stop_at_first_sentence else i,
    )


def encode_arithmetic_mirror(
    *, model: Any, enc: Any, utils: Any, message: list[int], context_tokens: list[int],
    precision: int, topk: int, temp: float, device: str, stop_at_first_sentence: bool,
    max_generated_tokens: int | None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from transformers import DynamicCache

    context = torch.tensor(context_tokens[-1022:], device=device, dtype=torch.long)
    cur_interval = [0, 2 ** precision]
    prev = context
    output = context
    past = None
    i = 0
    total_log_probs = 0.0
    total_kl = 0.0
    total_entropy = 0.0
    token_count = 0
    boundary_positions: list[int] = []
    terminal_reason = "message_exhausted"
    used_implicit_zero = False
    secret_bits_read = 0

    with torch.no_grad():
        while True:
            if stop_at_first_sentence:
                if max_generated_tokens is not None and token_count >= max_generated_tokens:
                    terminal_reason = "max_generated_tokens"
                    break
                if i + precision > len(message):
                    terminal_reason = "bitstream_exhausted"
                    break
            elif i >= len(message):
                terminal_reason = "message_exhausted"
                break

            out = model(
                prev.unsqueeze(0),
                past_key_values=DynamicCache.from_legacy_cache(past),
                use_cache=True,
            )
            logits = out.logits
            past = utils.limit_past(out.past_key_values)
            logits[0, -1, -1] = -1e20
            logits[0, -1, 628] = -1e20
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            log_probs = F.log_softmax(logits, dim=0)

            cur_int_range = cur_interval[1] - cur_interval[0]
            cur_threshold = 1 / cur_int_range
            k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
            probs_temp_int = probs_temp[:k]
            probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range
            probs_temp_int = probs_temp_int.round().long()
            cum_probs = probs_temp_int.cumsum(0)
            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[: overfill_index[0]]
            cum_probs += cur_int_range - cum_probs[-1]

            probs_final = cum_probs.clone()
            probs_final[1:] = cum_probs[1:] - cum_probs[:-1]
            cum_probs += cur_interval[0]

            message_bits = message[i:i + precision]
            if len(message_bits) < precision:
                if stop_at_first_sentence:
                    terminal_reason = "bitstream_exhausted"
                    break
                missing = precision - len(message_bits)
                message_bits = message_bits + [0] * missing
                used_implicit_zero = True
            secret_bits_read = max(secret_bits_read, i + precision)
            message_idx = utils.bits2int(reversed(message_bits))
            selection = int((cum_probs > message_idx).nonzero()[0].item())

            new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
            new_int_top = cum_probs[selection]
            lower_bits = list(reversed(utils.int2bits(new_int_bottom, precision)))
            upper_bits = list(reversed(utils.int2bits(new_int_top - 1, precision)))
            num_bits_encoded = int(utils.num_same_from_beg(lower_bits, upper_bits))
            i += num_bits_encoded
            lower_rest = lower_bits[num_bits_encoded:] + [0] * num_bits_encoded
            upper_rest = upper_bits[num_bits_encoded:] + [1] * num_bits_encoded
            cur_interval[0] = utils.bits2int(reversed(lower_rest))
            cur_interval[1] = utils.bits2int(reversed(upper_rest)) + 1

            total_log_probs += float(log_probs[selection].item())
            q = probs_final.double() / probs_final.sum()
            logq = q.log()
            total_kl += float(utils.kl(q, logq, log_probs[: len(q)]))
            total_entropy += float(utils.entropy(probs_temp, log_probs_temp))
            token_count += 1

            selected = int(indices[selection].item())
            prev = indices[selection].view(1)
            output = torch.cat((output, prev))
            if bool(utils.is_sent_finish(selected, enc)):
                boundary_positions.append(token_count - 1)
                if stop_at_first_sentence:
                    terminal_reason = "sentence_boundary"
                    break

            partial = enc.decode(output[len(context):].tolist())
            if "<eos>" in partial:
                terminal_reason = "eos_marker"
                break

    generated = [int(x) for x in output[len(context):].tolist()]
    return _finish(
        {
            "generated_token_ids": generated,
            "terminal_reason": terminal_reason,
            "sentence_finish_token_positions_zero_based": boundary_positions,
            "used_implicit_zero_lookahead": used_implicit_zero,
            "final_interval_width": int(cur_interval[1] - cur_interval[0]),
            "final_effective_precision_bits": math.log2(int(cur_interval[1] - cur_interval[0])),
        },
        total_log_probs=total_log_probs,
        total_kl=total_kl,
        token_count=token_count,
        payload_bits=i,
        secret_bits_read=secret_bits_read,
        total_entropy=total_entropy,
    )
