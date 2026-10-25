"""Exact-token native and vLLM callers for the adopted four-role protocol.

Sampling never masks logits, repairs text or invents EOS/logprobs. Public role
schema validation belongs to full_agent_v4; this boundary checks serialization.
Optional model/backend dependencies are imported only when used.
"""
from __future__ import annotations

import copy
import inspect
import json
import math
import time
from uuid import uuid4

from castfactory.agentic import distillation_protocol as protocol


SAMPLING_PROTOCOL = "merged-t1-answer-native-eos-or-budget-v1"
JSON_ROWS_SAMPLING_PROTOCOL = "agentic-json-rows-native-eos-or-budget-v1"
COMPACT_SAMPLING_PROTOCOL = "agentic-compact-markdown-native-eos-or-budget-v1"
STEP_TARGET_SAMPLING_PROTOCOL = "agentic-compact-markdown-native-eos-or-budget-v2"
COMPACT_PROFILES = ("agentic-compact-markdown-v1", "agentic-compact-markdown-v2")
COST_PROTOCOL = "full-response-prefill-plus-decode-v1"
ROLES = ("T1", "T2", "T2_refine", "T3", "T4")


class SamplingTransportErrorV4(RuntimeError):
    """Backend evidence cannot be used as an on-policy sample."""


def _field(output, key, default=None):
    return output.get(key, default) if isinstance(output, dict) else getattr(output, key, default)


def _eos(tokenizer):
    eos = tokenizer.eos_token_id
    if type(eos) is not int or eos < 0:
        raise ValueError("One explicit tokenizer EOS ID is required")
    return eos


def _seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Seed must be a nonnegative signed 64-bit integer")
    return seed


def sampling_params_v4(profile, seed, eos, supplied=None, stochastic=True):
    """Explicit evaluation argmax or RL unmodified categorical distribution."""
    params = {"temperature": 1.0 if stochastic else 0.0, "top_p": 1.0,
              "top_k": -1, "min_p": 0.0, "presence_penalty": 0.0,
              "frequency_penalty": 0.0, "repetition_penalty": 1.0,
              "max_tokens": profile.max_response_tokens, "min_tokens": 0,
              "logprobs": True, "seed": _seed(seed), "n": 1,
              "ignore_eos": False, "stop_token_ids": [eos], "stop": []}
    for key, value in (supplied or {}).items():
        if key not in params or value != params[key]:
            raise ValueError("Sampling parameter changes protocol: " + key)
    return params


def protocol_binding_v4(profile, *, stochastic=True, backend="verl_llm_server_vllm"):
    return {"profile_version": profile.profile_version,
            "sampling_protocol": _sampling_protocol(profile.profile_version),
            "cost_protocol": COST_PROTOCOL, "backend": backend,
            "profile": profile.record(), "temperature": 1.0 if stochastic else 0.0,
            "termination": "tokenizer_eos_or_full_budget",
            "cost_unit": "logical_prompt_plus_all_generated_tokens",
            "requires_rebuilt_direct_reference": True}


def _sampling_protocol(profile_version):
    if profile_version == protocol.VERSION:
        return SAMPLING_PROTOCOL
    if profile_version == "agentic-json-rows-v1":
        return JSON_ROWS_SAMPLING_PROTOCOL
    if profile_version == "agentic-compact-markdown-v1":
        return COMPACT_SAMPLING_PROTOCOL
    if profile_version == "agentic-compact-markdown-v2":
        return STEP_TARGET_SAMPLING_PROTOCOL
    raise ValueError("Unsupported sampling profile: " + str(profile_version))


def _prompt(tokenizer, messages, profile):
    prompt, prefix = protocol.student_prompt_tokens(tokenizer, messages)
    if not prompt or any(type(token) is not int or token < 0 for token in prompt):
        raise ValueError("Invalid exact prompt IDs")
    limit = profile.max_context_tokens
    if limit is not None and len(prompt) + profile.max_response_tokens > limit:
        from castfactory.agentic.full_agent_v3 import ModelContextOverflow
        raise ModelContextOverflow("Prompt exceeds context allocation; truncation forbidden")
    return list(prompt), prefix


def _record(tokenizer, prompt, prefix, response, old, *, actor_version, seed,
            started, raw_stop_reason=None, profile_version=protocol.VERSION):
    eos = _eos(tokenizer)
    sampled_eos = bool(response) and response[-1] == eos
    continuation = tokenizer.decode(response[:-1] if sampled_eos else response,
                                    skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
    # Prompt ownership changes only parsing text, never response IDs or loss.
    content = prefix + continuation
    finish, termination = "length", "token_budget"
    if sampled_eos:
        finish, termination = "stop", "validated_answer_sampled_eos"
        try:
            if profile_version in COMPACT_PROFILES:
                from castfactory.agentic.compact_protocol import parse_envelope
                parse_envelope(content)
            else:
                protocol.parse_reply(content, "student")
        except ValueError as exc:
            finish, termination = "contract_error", str(exc)
    return {"profile_version": profile_version, "content": content,
            "continuation_content": continuation, "prompt_owned_response_prefix": prefix,
            "prompt_ids": list(prompt), "response_ids": list(response),
            "old_log_probs": list(old) if old is not None else None,
            "response_loss_mask": [1] * len(response),
            "prefill_tokens": len(prompt), "decode_tokens": len(response),
            "actor_version": actor_version, "sampling_seed": seed,
            "finish_reason": finish, "termination_reason": termination,
            "generation_limit_reached": not sampled_eos, "raw_stop_reason": raw_stop_reason,
            "sampling_protocol": _sampling_protocol(profile_version), "cost_protocol": COST_PROTOCOL,
            "seconds": time.monotonic() - started}


def excess_t2_values_v4(content, expected):
    """Diagnose an overlong top-level array without changing sampled output.

    A number counts only after a delimiter arrives; partial numeric prefixes
    cannot establish its value or completion. This is not a stopping processor.
    """
    marker = content.find("<answer>")
    if marker < 0:
        return False
    raw = content[marker + len("<answer>"):].lstrip()
    if not raw.startswith("{"):
        return False
    decoder = json.JSONDecoder()
    pos = 1
    try:
        while True:
            pos += len(raw[pos:]) - len(raw[pos:].lstrip())
            key, pos = decoder.raw_decode(raw, pos)
            pos += len(raw[pos:]) - len(raw[pos:].lstrip())
            if raw[pos:pos + 1] != ":":
                return False
            pos += 1
            pos += len(raw[pos:]) - len(raw[pos:].lstrip())
            if key == "values":
                if raw[pos:pos + 1] != "[":
                    return False
                pos += 1
                count = 0
                while True:
                    pos += len(raw[pos:]) - len(raw[pos:].lstrip())
                    _, end = decoder.raw_decode(raw, pos)
                    end += len(raw[end:]) - len(raw[end:].lstrip())
                    delimiter = raw[end:end + 1]
                    if delimiter not in (",", "]"):
                        return False
                    count += 1
                    if delimiter == "," and count >= expected:
                        return True
                    if delimiter == "]":
                        return count > expected
                    pos = end + 1
            _, pos = decoder.raw_decode(raw, pos)
            pos += len(raw[pos:]) - len(raw[pos:].lstrip())
            if raw[pos:pos + 1] != ",":
                return False
            pos += 1
    except (ValueError, IndexError):
        return False


def protocol_terminal_v4(role, variables, profile, content):
    """Optional diagnostic validation; a valid answer still requires EOS."""
    if profile.profile_version in ("agentic-json-rows-v1", *COMPACT_PROFILES):
        # JSON-row output remains untouched. This is a diagnostic after an
        # answer closes, never a stop criterion or a numerical repair.
        if content.rstrip().endswith("</answer>"):
            if profile.profile_version in COMPACT_PROFILES:
                from castfactory.agentic.compact_protocol import parse_role_answer
                from castfactory.agentic.compact_prompts import request_for
            else:
                from castfactory.agentic.json_rows_protocol import parse_role_answer
                from castfactory.agentic.json_rows_prompts import request_for
            try:
                if "case" in variables:
                    request = variables
                else:
                    public_case = variables.get("public_json_case")
                    if not isinstance(public_case, dict):
                        raise ValueError("JSON diagnostic requires the actual public JSON case")
                    state = {key: value for key, value in variables.items()
                             if key != "public_json_case"}
                    request = request_for(role, state, public_case)
                parse_role_answer(content, "T2" if role == "T2_refine" else role,
                                  request, profile=profile,
                                  **({"protocol_version": profile.profile_version}
                                     if profile.profile_version in COMPACT_PROFILES else {}))
            except ValueError as exc:
                return "contract_error", str(exc)
        return None
    if role in ("T2", "T2_refine"):
        segment = variables["segment_assignment"]
        expected = segment["stop"] - segment["start"]
        if excess_t2_values_v4(content, expected):
            return "contract_error", "T2 values exceeds exactly %s assigned points" % expected
    if content.rstrip().endswith("</answer>"):
        from castfactory.agentic.full_agent_v4 import validate_public_role
        try:
            parsed = protocol.parse_reply(content, "student")
            validate_public_role(role, parsed, variables, profile=profile)
        except ValueError as exc:
            return "contract_error", str(exc)
    return None


class NativeFullRoleSamplerV4:
    """One fresh cache per call, actual selected-token logprobs at T=1."""
    def __init__(self, model, tokenizer, profile, actor_version, *, stochastic,
                 seed=17, journal=None, device=None):
        import torch
        if actor_version is None or actor_version == "":
            raise ValueError("Explicit frozen actor version is required")
        self.model, self.tokenizer, self.profile = model, tokenizer, profile
        self.actor_version = copy.deepcopy(actor_version)
        self.stochastic, self.seed, self.journal = stochastic, _seed(seed), journal
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.calls = 0
        self.transport_receipts = []
        forward = getattr(model, "forward", None)
        self.forward_options = ({"logits_to_keep": 1}
                                if callable(forward) and "logits_to_keep" in inspect.signature(forward).parameters
                                else {})
        _eos(tokenizer)

    def __call__(self, role, messages):
        import torch
        if role not in ROLES:
            raise ValueError("Unknown public role")
        prompt, prefix = _prompt(self.tokenizer, messages, self.profile)
        call_seed = (self.seed + self.calls) % (2**63)
        self.calls += 1
        self.generator.manual_seed(call_seed)
        current = torch.tensor([prompt], device=self.device, dtype=torch.long)
        attention = torch.ones_like(current)
        response, old, cache = [], [], None
        started = time.monotonic()
        self.model.eval()
        try:
            with torch.no_grad():
                for _ in range(self.profile.max_response_tokens):
                    output = self.model(input_ids=current, attention_mask=attention,
                                        past_key_values=cache, use_cache=True, **self.forward_options)
                    logits = output.logits[0, -1].float()
                    log_probs = torch.log_softmax(logits, -1)
                    token = (torch.multinomial(log_probs.exp(), 1, generator=self.generator)
                             if self.stochastic else logits.argmax().view(1))
                    response.append(int(token.item()))
                    old.append(float(log_probs[token].item()))
                    cache = output.past_key_values
                    if response[-1] == _eos(self.tokenizer):
                        break
                    current = token.view(1, 1)
                    attention = torch.ones((1, len(prompt) + len(response)),
                                           dtype=torch.long, device=self.device)
        except Exception as exc:
            record = _record(self.tokenizer, prompt, prefix, response, old,
                             actor_version=self.actor_version, seed=call_seed,
                             profile_version=self.profile.profile_version, started=started)
            record.update(role=role, call_index=self.calls - 1, error=str(exc),
                          finish_reason="transport_error", termination_reason="native_model_error")
            self.transport_receipts.append(record)
            if self.journal is not None:
                self.journal(role, record)
            raise SamplingTransportErrorV4("Native model generation failed: " + str(exc)) from exc
        record = _record(self.tokenizer, prompt, prefix, response, old,
                         actor_version=self.actor_version, seed=call_seed,
                             profile_version=self.profile.profile_version, started=started)
        record.update(role=role, call_index=self.calls - 1)
        if self.journal is not None:
            self.journal(role, record)
        return record


class ExactIDRoleCallerV4:
    """Official LLMServerClient adapter; no text re-encoding or fabricated EOS."""
    def __init__(self, server_client, tokenizer, profile, actor_version, *, seed,
                 stochastic=True, expected_global_step=None, sampling_params=None,
                 request_prefix=None):
        if actor_version is None or actor_version == "":
            raise ValueError("Explicit frozen actor version is required")
        self.server_client, self.tokenizer, self.profile = server_client, tokenizer, profile
        self.actor_version = copy.deepcopy(actor_version)
        self.seed, self.stochastic = _seed(seed), stochastic
        self.expected_global_step = expected_global_step
        self.params = sampling_params_v4(profile, seed, _eos(tokenizer), sampling_params, stochastic)
        self.request_prefix = request_prefix or uuid4().hex
        self.calls, self.transport_receipts = 0, []

    async def call(self, role, messages):
        if role not in ROLES:
            raise ValueError("Unknown public role")
        prompt, prefix = _prompt(self.tokenizer, messages, self.profile)
        index = self.calls
        self.calls += 1
        call_seed = (self.seed + index) % (2**63)
        started = time.monotonic()
        try:
            output = await self.server_client.generate(
                request_id="%s:%s" % (self.request_prefix, index), prompt_ids=list(prompt),
                sampling_params={**copy.deepcopy(self.params), "seed": call_seed})
        except Exception as exc:
            # No response was returned: retain the submitted prompt and mark
            # unavailable generation evidence explicitly, never synthesize it.
            self.transport_receipts.append({"role": role, "call_index": index,
                "profile_version": self.profile.profile_version,
                "prompt_ids": prompt, "response_ids": [], "old_log_probs": None,
                "prefill_tokens": len(prompt), "decode_tokens": 0,
                "actor_version": self.actor_version, "sampling_seed": call_seed,
                "sampling_protocol": _sampling_protocol(self.profile.profile_version), "cost_protocol": COST_PROTOCOL,
                "error": str(exc), "finish_reason": "transport_error"})
            raise SamplingTransportErrorV4("Backend generation failed: " + str(exc)) from exc
        response = list(_field(output, "token_ids", []) or [])
        old = _field(output, "log_probs")
        old = list(old) if old is not None else None
        extra = _field(output, "extra_fields", {}) or {}
        raw_stop = _field(output, "stop_reason")
        receipt = {"role": role, "call_index": index, "prompt_ids": prompt,
                   "profile_version": self.profile.profile_version,
                   "response_ids": response, "old_log_probs": old,
                   "prefill_tokens": len(prompt), "decode_tokens": len(response),
                   "actor_version": self.actor_version, "sampling_seed": call_seed,
                   "raw_stop_reason": raw_stop, "backend_global_step": extra.get("global_steps"),
                   "sampling_protocol": _sampling_protocol(self.profile.profile_version), "cost_protocol": COST_PROTOCOL}

        def fail(message):
            # Keep real IDs/logprobs even when transport evidence is inadmissible.
            if all(type(token) is int and token >= 0 for token in response):
                receipt.update(_record(self.tokenizer, prompt, prefix, response, old,
                                       actor_version=self.actor_version, seed=call_seed,
                             profile_version=self.profile.profile_version,
                                       started=started, raw_stop_reason=raw_stop))
            self.transport_receipts.append({**receipt, "error": message})
            raise SamplingTransportErrorV4(message)

        if any(type(token) is not int or token < 0 for token in response):
            fail("Invalid actual sampled token IDs")
        if not response or len(response) > self.profile.max_response_tokens:
            fail("Empty or over-budget backend response")
        if old is None or len(old) != len(response) or any(
                isinstance(p, bool) or not isinstance(p, (int, float))
                or not math.isfinite(p) or p > 1e-5 for p in old):
            fail("Missing or invalid actual selected-token log-probabilities")
        if raw_stop not in ("completed", "stop", "length"):
            fail("Backend did not complete generation: " + str(raw_stop))
        if "actor_version" in extra and extra["actor_version"] != self.actor_version:
            fail("Backend actor version mismatch")
        expected = self.expected_global_step
        if expected is None and type(self.actor_version) is int:
            expected = self.actor_version
        if expected is not None and extra.get("global_steps") != expected:
            fail("Backend global step mismatch or missing acknowledgement")
        if expected is None and extra.get("actor_version") != self.actor_version:
            fail("Backend actor acknowledgement missing; bind expected_global_step")
        eos = _eos(self.tokenizer)
        if eos in response[:-1]:
            fail("Backend generated tokens after tokenizer EOS")
        if response[-1] != eos and len(response) != self.profile.max_response_tokens:
            fail("Backend stopped before EOS or full budget")
        record = _record(self.tokenizer, prompt, prefix, response, old,
                         actor_version=self.actor_version, seed=call_seed,
                             profile_version=self.profile.profile_version,
                         started=started, raw_stop_reason=raw_stop)
        return {**receipt, **record}
