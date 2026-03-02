"""Gemini 3.1 Pro prompt behavior integration tests with structured artifacts."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pytest
from loguru import logger
from pydantic import BaseModel

from naver_connect_chatbot.prompts import get_prompt
from naver_connect_chatbot.rag.retriever.multi_query_retriever import MultiQueryOutput
from naver_connect_chatbot.service.agents.intent_classifier import IntentClassification
from naver_connect_chatbot.service.agents.query_analyzer import (
    QueryAnalysis,
    QueryExpansionResult,
    QueryQualityResult,
)


ARTIFACT_DIR = Path(__file__).parent.parent / "logs" / "tests"
KOREAN_PATTERN = re.compile(r"[가-힣]")

ALLOWED_DOC_TYPES = {
    "slack_qa",
    "pdf",
    "notebook",
    "lecture_transcript",
    "weekly_mission",
}

UNCERTAINTY_TOKENS = (
    "정보가 부족",
    "불충분",
    "확인할 수",
    "알 수",
    "제공된 문맥",
    "추가",
    "명확하지",
)


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= minimum else default


def _env_backoffs(name: str, default: tuple[int, ...], max_attempts: int) -> tuple[int, ...]:
    raw = os.getenv(name, "").strip()
    if raw:
        parsed_values: list[int] = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                value = int(token)
            except ValueError:
                continue
            if value >= 0:
                parsed_values.append(value)
        if parsed_values:
            default = tuple(parsed_values)

    if len(default) < max_attempts:
        last = default[-1] if default else 0
        extended = list(default) + [last] * (max_attempts - len(default))
        return tuple(extended[:max_attempts])

    return tuple(default[:max_attempts])


MAX_RETRY_ATTEMPTS = _env_int("PROMPT_BEHAVIOR_MAX_RETRIES", default=3, minimum=1)
ATTEMPT_TIMEOUT_SECONDS = _env_int("PROMPT_BEHAVIOR_ATTEMPT_TIMEOUT", default=120, minimum=5)
RETRY_BACKOFF_SECONDS = _env_backoffs(
    "PROMPT_BEHAVIOR_RETRY_BACKOFFS",
    default=(2, 5, 10),
    max_attempts=MAX_RETRY_ATTEMPTS,
)
SCENARIO_RATE_LIMIT_PER_MIN = _env_int("PROMPT_BEHAVIOR_RATE_LIMIT_PER_MIN", default=0, minimum=0)


@dataclass(frozen=True)
class PromptScenario:
    """Single prompt behavior scenario."""

    scenario_id: str
    prompt_name: str
    variant: Literal["normal", "boundary"]
    call_type: Literal["structured", "text"]
    llm_kind: Literal["llm", "reasoning_llm"]
    inputs: dict[str, Any]
    schema: type[BaseModel] | None


class RetryExhaustedError(RuntimeError):
    """Raised when retryable API errors exhaust all attempts."""

    def __init__(self, message: str, attempts: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.attempts = attempts


def _is_retryable_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()

    if "resource_exhausted" in text and (
        "limit: 0" in text
        or "per_day" in text
        or "generaterequestsperday" in text
        or "quota exceeded for metric" in text
    ):
        return False

    retry_markers = (
        "503",
        "429",
        "unavailable",
        "resource_exhausted",
        "rate limit",
        "timeout",
        "temporar",
        "network",
        "connection",
    )
    if any(marker in text for marker in retry_markers):
        return True

    return isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError))


def _parse_csv_env(name: str) -> set[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return set()
    return {token.strip() for token in raw.split(",") if token.strip()}


def _select_scenarios(scenarios: list[PromptScenario]) -> list[PromptScenario]:
    selected = list(scenarios)

    scenario_ids = _parse_csv_env("PROMPT_BEHAVIOR_SCENARIOS")
    if scenario_ids:
        selected = [scenario for scenario in selected if scenario.scenario_id in scenario_ids]

    prompt_names = _parse_csv_env("PROMPT_BEHAVIOR_PROMPTS")
    if prompt_names:
        selected = [scenario for scenario in selected if scenario.prompt_name in prompt_names]

    chunk_size = _env_int("PROMPT_BEHAVIOR_CHUNK_SIZE", default=0, minimum=0)
    chunk_index = _env_int("PROMPT_BEHAVIOR_CHUNK_INDEX", default=0, minimum=0)
    if chunk_size > 0:
        start = chunk_index * chunk_size
        end = start + chunk_size
        selected = selected[start:end]

    return selected


def _extract_response_text(result: Any) -> str:
    """Extract text content from model response objects."""
    if isinstance(result, str):
        return result

    if hasattr(result, "content"):
        content = result.content
    else:
        content = result

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    text_parts.append(str(block["text"]))
                elif "text" in block:
                    text_parts.append(str(block["text"]))
            elif isinstance(block, str):
                text_parts.append(block)

        if text_parts:
            return "\n".join(text_parts)

    return str(content)


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _is_score_in_range(value: float) -> bool:
    return isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0


def _is_unique_casefold(items: list[str]) -> bool:
    normalized = [item.casefold().strip() for item in items]
    return len(normalized) == len(set(normalized))


def _has_uncertainty_expression_korean(text: str) -> bool:
    return any(token in text for token in UNCERTAINTY_TOKENS)


def _has_conservative_filter_shape(filters: Any) -> bool:
    nullable_fields = [
        getattr(filters, "doc_type", None),
        getattr(filters, "course", None),
        getattr(filters, "course_topic", None),
        getattr(filters, "generation", None),
    ]
    return any(value is None for value in nullable_fields)


def _summarize_value(value: Any, max_len: int = 180) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_len else value[:max_len] + "..."

    if isinstance(value, list):
        if len(value) <= 5:
            return [_summarize_value(v, max_len=max_len) for v in value]
        head = [_summarize_value(v, max_len=max_len) for v in value[:5]]
        head.append(f"...({len(value) - 5} more)")
        return head

    if isinstance(value, dict):
        return {k: _summarize_value(v, max_len=max_len) for k, v in value.items()}

    if isinstance(value, BaseModel):
        return _summarize_value(value.model_dump(), max_len=max_len)

    return value


def _summarize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {k: _summarize_value(v) for k, v in inputs.items()}


def _summarize_output(output: Any) -> dict[str, Any]:
    if isinstance(output, BaseModel):
        dumped = output.model_dump()
        return {
            "type": type(output).__name__,
            "data": _summarize_value(dumped),
        }

    if isinstance(output, str):
        return {
            "type": "text",
            "length": len(output),
            "preview": _summarize_value(output, max_len=260),
        }

    return {
        "type": type(output).__name__,
        "data": _summarize_value(output),
    }


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_markdown_report(
    run_id: str,
    started_at: str,
    completed_at: str,
    jsonl_path: Path,
    records: list[dict[str, Any]],
    selected_scenario_ids: list[str],
) -> str:
    total = len(records)
    passed = sum(1 for record in records if record["status"] == "PASS")
    failed = total - passed
    pass_rate = (passed / total * 100.0) if total else 0.0

    per_prompt: dict[str, dict[str, int]] = defaultdict(lambda: {"PASS": 0, "FAIL": 0})
    for record in records:
        if record["status"] == "PASS":
            per_prompt[record["prompt_name"]]["PASS"] += 1
        else:
            per_prompt[record["prompt_name"]]["FAIL"] += 1

    lines: list[str] = []
    lines.append("# Gemini Prompt Behavior Report")
    lines.append("")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- started_at: `{started_at}`")
    lines.append(f"- completed_at: `{completed_at}`")
    lines.append(f"- jsonl_artifact: `{jsonl_path}`")
    lines.append(f"- selected_scenarios: `{','.join(selected_scenario_ids)}`")
    lines.append(f"- max_retry_attempts: `{MAX_RETRY_ATTEMPTS}`")
    lines.append(f"- retry_backoff_seconds: `{RETRY_BACKOFF_SECONDS}`")
    lines.append(f"- attempt_timeout_seconds: `{ATTEMPT_TIMEOUT_SECONDS}`")
    lines.append(f"- scenario_rate_limit_per_min: `{SCENARIO_RATE_LIMIT_PER_MIN}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- total_scenarios: **{total}**")
    lines.append(f"- passed: **{passed}**")
    lines.append(f"- failed: **{failed}**")
    lines.append(f"- pass_rate: **{pass_rate:.1f}%**")
    lines.append("")
    lines.append("## Prompt Breakdown")
    lines.append("")
    lines.append("| Prompt | PASS | FAIL |")
    lines.append("|---|---:|---:|")
    for prompt_name in sorted(per_prompt):
        row = per_prompt[prompt_name]
        lines.append(f"| {prompt_name} | {row['PASS']} | {row['FAIL']} |")

    failed_records = [record for record in records if record["status"] != "PASS"]
    lines.append("")
    lines.append("## Failed Scenarios")
    lines.append("")
    if not failed_records:
        lines.append("- None")
    else:
        for record in failed_records:
            lines.append(
                f"- `{record['scenario_id']}` ({record['prompt_name']}/{record['variant']}) "
                f"status=`{record['status']}` attempts={record['attempts']}"
            )
            errors = record.get("errors") or []
            for error in errors[:3]:
                lines.append(f"- error: {error}")

    lines.append("")
    lines.append("## Retry Stats")
    lines.append("")
    retries_used = sum(max(0, int(record.get("attempts", 1)) - 1) for record in records)
    lines.append(f"- total_retries_used: **{retries_used}**")

    return "\n".join(lines) + "\n"


async def _invoke_scenario_with_retry(
    scenario: PromptScenario,
    llm: Any,
    reasoning_llm: Any,
    eval_logger: Any,
) -> tuple[Any, str, int, list[dict[str, Any]], float]:
    attempt_logs: list[dict[str, Any]] = []

    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        prompt_text = ""
        step_start = time.perf_counter()
        try:
            prompt_template = get_prompt(scenario.prompt_name)
            prompt_text = prompt_template.format_prompt(**scenario.inputs).to_string()

            model = llm if scenario.llm_kind == "llm" else reasoning_llm

            if scenario.call_type == "structured":
                if scenario.schema is None:
                    raise ValueError(f"Structured scenario requires schema: {scenario.scenario_id}")
                structured_llm = model.with_structured_output(scenario.schema)
                output = await asyncio.wait_for(
                    structured_llm.ainvoke(prompt_text),
                    timeout=ATTEMPT_TIMEOUT_SECONDS,
                )
            else:
                raw_output = await asyncio.wait_for(
                    model.ainvoke(prompt_text),
                    timeout=ATTEMPT_TIMEOUT_SECONDS,
                )
                output = _extract_response_text(raw_output)

            elapsed_ms = (time.perf_counter() - step_start) * 1000.0
            return output, prompt_text, attempt, attempt_logs, elapsed_ms

        except Exception as exc:
            retryable = _is_retryable_error(exc)
            elapsed_ms = (time.perf_counter() - step_start) * 1000.0
            entry = {
                "attempt": attempt,
                "retryable": retryable,
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": round(elapsed_ms, 2),
            }
            attempt_logs.append(entry)

            eval_logger.warning(
                f"Scenario {scenario.scenario_id} attempt {attempt} failed "
                f"(retryable={retryable}): {entry['error']}"
            )

            if not retryable:
                raise

            if attempt >= MAX_RETRY_ATTEMPTS:
                raise RetryExhaustedError(
                    f"Retry exhausted for scenario={scenario.scenario_id}",
                    attempts=attempt_logs,
                )

            backoff_seconds = RETRY_BACKOFF_SECONDS[min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)]
            await asyncio.sleep(backoff_seconds)

    raise RuntimeError("Unreachable retry loop end")


def _validate_result(scenario: PromptScenario, output: Any) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    if scenario.scenario_id == "intent_normal":
        assert isinstance(output, IntentClassification)
        checks.extend(
            [
                _check(
                    "intent_not_ood",
                    output.intent != "OUT_OF_DOMAIN",
                    f"intent={output.intent}",
                ),
                _check(
                    "domain_relevance_ge_0_5",
                    output.domain_relevance >= 0.5,
                    f"domain_relevance={output.domain_relevance:.3f}",
                ),
                _check(
                    "confidence_in_range",
                    _is_score_in_range(output.confidence),
                    f"confidence={output.confidence:.3f}",
                ),
                _check(
                    "hard_rule_consistency",
                    not (output.domain_relevance < 0.2 and output.intent != "OUT_OF_DOMAIN"),
                    f"intent={output.intent}, domain_relevance={output.domain_relevance:.3f}",
                ),
            ]
        )

    elif scenario.scenario_id == "intent_boundary":
        assert isinstance(output, IntentClassification)
        checks.extend(
            [
                _check("intent_is_ood", output.intent == "OUT_OF_DOMAIN", f"intent={output.intent}"),
                _check(
                    "domain_relevance_lt_0_5",
                    output.domain_relevance < 0.5,
                    f"domain_relevance={output.domain_relevance:.3f}",
                ),
                _check(
                    "hard_rule_consistency",
                    not (output.domain_relevance < 0.2 and output.intent != "OUT_OF_DOMAIN"),
                    f"intent={output.intent}, domain_relevance={output.domain_relevance:.3f}",
                ),
            ]
        )

    elif scenario.scenario_id in {"query_analysis_normal", "query_analysis_boundary"}:
        assert isinstance(output, QueryAnalysis)
        expected_count = 3 if scenario.scenario_id == "query_analysis_normal" else 3
        checks.extend(
            [
                _check("clarity_range", _is_score_in_range(output.clarity_score), f"clarity={output.clarity_score:.3f}"),
                _check(
                    "specificity_range",
                    _is_score_in_range(output.specificity_score),
                    f"specificity={output.specificity_score:.3f}",
                ),
                _check(
                    "searchability_range",
                    _is_score_in_range(output.searchability_score),
                    f"searchability={output.searchability_score:.3f}",
                ),
                _check(
                    "query_count_matches_policy",
                    len(output.improved_queries) == expected_count,
                    f"query_count={len(output.improved_queries)} expected={expected_count}",
                ),
                _check(
                    "queries_unique",
                    _is_unique_casefold(output.improved_queries),
                    f"queries={output.improved_queries}",
                ),
                _check(
                    "filter_confidence_range",
                    _is_score_in_range(output.retrieval_filters.filter_confidence),
                    f"filter_confidence={output.retrieval_filters.filter_confidence:.3f}",
                ),
            ]
        )

        doc_type = output.retrieval_filters.doc_type
        if doc_type:
            checks.append(
                _check(
                    "doc_type_values_allowed",
                    set(doc_type).issubset(ALLOWED_DOC_TYPES),
                    f"doc_type={doc_type}",
                )
            )

        if scenario.scenario_id == "query_analysis_boundary":
            checks.append(
                _check(
                    "issues_not_empty",
                    len(output.issues) >= 1,
                    f"issues_count={len(output.issues)}",
                )
            )
            checks.append(
                _check(
                    "conservative_filter",
                    _has_conservative_filter_shape(output.retrieval_filters),
                    (
                        "at least one null filter field="
                        f"doc_type={output.retrieval_filters.doc_type}, "
                        f"course={output.retrieval_filters.course}, "
                        f"course_topic={output.retrieval_filters.course_topic}, "
                        f"generation={output.retrieval_filters.generation}"
                    ),
                )
            )

    elif scenario.scenario_id in {"quality_normal", "quality_boundary"}:
        assert isinstance(output, QueryQualityResult)
        checks.extend(
            [
                _check("clarity_range", _is_score_in_range(output.clarity_score), f"clarity={output.clarity_score:.3f}"),
                _check(
                    "specificity_range",
                    _is_score_in_range(output.specificity_score),
                    f"specificity={output.specificity_score:.3f}",
                ),
                _check(
                    "searchability_range",
                    _is_score_in_range(output.searchability_score),
                    f"searchability={output.searchability_score:.3f}",
                ),
                _check("issues_nonempty", len(output.issues) >= 1, f"issues_count={len(output.issues)}"),
                _check(
                    "recommendations_nonempty",
                    len(output.recommendations) >= 1,
                    f"recommendations_count={len(output.recommendations)}",
                ),
            ]
        )
        if scenario.scenario_id == "quality_normal":
            checks.append(
                _check(
                    "clarity_reasonable_for_clear_query",
                    output.clarity_score >= 0.5,
                    f"clarity={output.clarity_score:.3f}",
                )
            )
        else:
            checks.append(
                _check(
                    "clarity_not_too_high_for_ambiguous_query",
                    output.clarity_score <= 0.7,
                    f"clarity={output.clarity_score:.3f}",
                )
            )

    elif scenario.scenario_id in {"expansion_normal", "expansion_boundary"}:
        assert isinstance(output, QueryExpansionResult)
        expected_count = 3 if scenario.scenario_id == "expansion_normal" else 5
        checks.extend(
            [
                _check(
                    "query_count_matches_quality_policy",
                    len(output.improved_queries) == expected_count,
                    f"query_count={len(output.improved_queries)} expected={expected_count}",
                ),
                _check(
                    "queries_unique",
                    _is_unique_casefold(output.improved_queries),
                    f"queries={output.improved_queries}",
                ),
                _check(
                    "filter_confidence_range",
                    _is_score_in_range(output.retrieval_filters.filter_confidence),
                    f"filter_confidence={output.retrieval_filters.filter_confidence:.3f}",
                ),
            ]
        )
        if scenario.scenario_id == "expansion_boundary":
            checks.append(
                _check(
                    "conservative_filter",
                    _has_conservative_filter_shape(output.retrieval_filters),
                    (
                        "at least one null filter field="
                        f"doc_type={output.retrieval_filters.doc_type}, "
                        f"course={output.retrieval_filters.course}, "
                        f"course_topic={output.retrieval_filters.course_topic}, "
                        f"generation={output.retrieval_filters.generation}"
                    ),
                )
            )

    elif scenario.scenario_id in {"multi_query_normal", "multi_query_boundary"}:
        assert isinstance(output, MultiQueryOutput)
        expected_count = int(scenario.inputs["num"])
        checks.extend(
            [
                _check(
                    "exact_query_count",
                    len(output.queries) == expected_count,
                    f"query_count={len(output.queries)} expected={expected_count}",
                ),
                _check(
                    "queries_unique",
                    _is_unique_casefold(output.queries),
                    f"queries={output.queries}",
                ),
                _check(
                    "query_line_quality",
                    all(len(query.strip()) >= 8 for query in output.queries),
                    f"queries={output.queries}",
                ),
            ]
        )
        if scenario.scenario_id == "multi_query_boundary":
            checks.append(
                _check(
                    "coreference_resolved",
                    any(
                        token in " ".join(output.queries).casefold()
                        for token in ("transformer", "학습", "최적화")
                    ),
                    f"queries={output.queries}",
                )
            )

    elif scenario.scenario_id.startswith("answer_"):
        assert isinstance(output, str)
        text = output.strip()
        checks.extend(
            [
                _check("contains_korean", bool(KOREAN_PATTERN.search(text)), f"preview={text[:120]}"),
                _check("no_code_fence", "```" not in text, f"preview={text[:120]}"),
                _check(
                    "not_json_output",
                    not text.startswith("{") and not text.startswith("["),
                    f"preview={text[:120]}",
                ),
            ]
        )

        if scenario.scenario_id in {
            "answer_simple_boundary",
            "answer_complex_boundary",
        }:
            checks.append(
                _check(
                    "uncertainty_stated",
                    _has_uncertainty_expression_korean(text),
                    f"preview={text[:180]}",
                )
            )

        if scenario.scenario_id == "answer_complex_normal":
            checks.append(
                _check(
                    "has_step_or_section_structure",
                    bool(re.search(r"(^|\n)\s*(\d+\.|[가-힣A-Za-z ]+:)", text)),
                    f"preview={text[:180]}",
                )
            )
            checks.append(
                _check(
                    "conflict_handled",
                    any(token in text for token in ("충돌", "다르", "맥락", "문서")),
                    f"preview={text[:180]}",
                )
            )

        if scenario.scenario_id == "answer_exploratory_normal":
            heading_hits = sum(1 for token in ("개요", "핵심", "다음", "실습", "학습") if token in text)
            checks.append(
                _check(
                    "exploratory_structure_present",
                    heading_hits >= 2,
                    f"heading_hits={heading_hits}, preview={text[:180]}",
                )
            )

        if scenario.scenario_id == "answer_exploratory_boundary":
            checks.append(
                _check(
                    "focus_on_followup_need",
                    ("이번 주" in text or "이번주" in text) and ("실습" in text),
                    f"preview={text[:180]}",
                )
            )
            checks.append(
                _check(
                    "avoid_literal_history_repeat",
                    "로드맵 개요를 이미 제공했다" not in text,
                    f"preview={text[:180]}",
                )
            )

    else:
        checks.append(_check("known_scenario", False, f"Unknown scenario_id={scenario.scenario_id}"))

    return checks


async def _run_scenario(
    scenario: PromptScenario,
    run_id: str,
    llm: Any,
    reasoning_llm: Any,
    eval_logger: Any,
) -> dict[str, Any]:
    scenario_start = time.perf_counter()
    timestamp = datetime.now().isoformat()

    try:
        output, prompt_text, attempts, attempt_logs, last_latency_ms = await _invoke_scenario_with_retry(
            scenario=scenario,
            llm=llm,
            reasoning_llm=reasoning_llm,
            eval_logger=eval_logger,
        )

        checks = _validate_result(scenario, output)
        failed_checks = [check for check in checks if not check["passed"]]

        status = "PASS" if not failed_checks else "VALIDATION_FAIL"
        errors = [check["detail"] for check in failed_checks]

        eval_logger.info(
            f"Scenario {scenario.scenario_id} completed: status={status}, "
            f"attempts={attempts}, checks={len(checks)}"
        )

        return {
            "run_id": run_id,
            "timestamp": timestamp,
            "scenario_id": scenario.scenario_id,
            "prompt_name": scenario.prompt_name,
            "variant": scenario.variant,
            "status": status,
            "attempts": attempts,
            "latency_ms": round(last_latency_ms, 2),
            "total_elapsed_ms": round((time.perf_counter() - scenario_start) * 1000.0, 2),
            "prompt_length": len(prompt_text),
            "input_summary": _summarize_inputs(scenario.inputs),
            "output_summary": _summarize_output(output),
            "validation_checks": checks,
            "errors": errors,
            "retry_trace": attempt_logs,
        }

    except RetryExhaustedError as exc:
        eval_logger.error(f"Scenario {scenario.scenario_id} retry exhausted")
        return {
            "run_id": run_id,
            "timestamp": timestamp,
            "scenario_id": scenario.scenario_id,
            "prompt_name": scenario.prompt_name,
            "variant": scenario.variant,
            "status": "API_ERROR_RETRY_EXHAUSTED",
            "attempts": MAX_RETRY_ATTEMPTS,
            "latency_ms": 0.0,
            "total_elapsed_ms": round((time.perf_counter() - scenario_start) * 1000.0, 2),
            "prompt_length": None,
            "input_summary": _summarize_inputs(scenario.inputs),
            "output_summary": {},
            "validation_checks": [],
            "errors": [str(exc)],
            "retry_trace": exc.attempts,
        }

    except Exception as exc:
        eval_logger.error(f"Scenario {scenario.scenario_id} failed with non-retryable error: {exc}")
        return {
            "run_id": run_id,
            "timestamp": timestamp,
            "scenario_id": scenario.scenario_id,
            "prompt_name": scenario.prompt_name,
            "variant": scenario.variant,
            "status": "API_ERROR",
            "attempts": 1,
            "latency_ms": 0.0,
            "total_elapsed_ms": round((time.perf_counter() - scenario_start) * 1000.0, 2),
            "prompt_length": None,
            "input_summary": _summarize_inputs(scenario.inputs),
            "output_summary": {},
            "validation_checks": [],
            "errors": [f"{type(exc).__name__}: {exc}"],
            "retry_trace": [],
        }


def _build_scenarios() -> list[PromptScenario]:
    """Build 16 standard-depth scenarios (8 prompts x 2 variants)."""
    return [
        PromptScenario(
            scenario_id="intent_normal",
            prompt_name="intent_classification",
            variant="normal",
            call_type="structured",
            llm_kind="llm",
            schema=IntentClassification,
            inputs={
                "conversation_history": "사용자는 이전 턴에서 PyTorch 학습 파이프라인 튜닝 중이라고 말했다.",
                "question": "PyTorch DataLoader의 num_workers 설정이 학습 속도에 미치는 영향이 궁금해.",
            },
        ),
        PromptScenario(
            scenario_id="intent_boundary",
            prompt_name="intent_classification",
            variant="boundary",
            call_type="structured",
            llm_kind="llm",
            schema=IntentClassification,
            inputs={
                "conversation_history": "No prior conversation.",
                "question": "내일 서울 날씨랑 우산 필요 여부 알려줘.",
            },
        ),
        PromptScenario(
            scenario_id="query_analysis_normal",
            prompt_name="query_analysis",
            variant="normal",
            call_type="structured",
            llm_kind="llm",
            schema=QueryAnalysis,
            inputs={
                "conversation_history": "사용자는 NLP 기초 강의를 수강 중이다.",
                "data_source_context": (
                    "Available doc_type: slack_qa, pdf, notebook, lecture_transcript, weekly_mission\n"
                    "Available courses: NLP 기초, CV 이론, AI 기초\n"
                    "Available generations: 1기, 2기, 3기"
                ),
                "intent": "SIMPLE_QA",
                "question": "Slack Q&A와 강의자료에서 Transformer의 Self-Attention 수식 설명을 찾고 싶어.",
            },
        ),
        PromptScenario(
            scenario_id="query_analysis_boundary",
            prompt_name="query_analysis",
            variant="boundary",
            call_type="structured",
            llm_kind="llm",
            schema=QueryAnalysis,
            inputs={
                "conversation_history": "사용자는 이전 턴에서 CNN과 Transformer를 번갈아 언급했지만 목표를 확정하지 않았다.",
                "data_source_context": (
                    "Available doc_type: slack_qa, pdf, notebook, lecture_transcript, weekly_mission\n"
                    "Available courses: NLP 기초, CV 이론, AI 기초\n"
                    "Available generations: 1기, 2기, 3기"
                ),
                "intent": "CLARIFICATION_NEEDED",
                "question": "그거 정리해줘.",
            },
        ),
        PromptScenario(
            scenario_id="quality_normal",
            prompt_name="query_quality_analysis",
            variant="normal",
            call_type="structured",
            llm_kind="llm",
            schema=QueryQualityResult,
            inputs={
                "conversation_history": "No prior conversation.",
                "intent": "SIMPLE_QA",
                "question": "PyTorch에서 AdamW와 SGD의 차이를 코드 예제로 설명해줘.",
            },
        ),
        PromptScenario(
            scenario_id="quality_boundary",
            prompt_name="query_quality_analysis",
            variant="boundary",
            call_type="structured",
            llm_kind="llm",
            schema=QueryQualityResult,
            inputs={
                "conversation_history": "No prior conversation.",
                "intent": "CLARIFICATION_NEEDED",
                "question": "이거 왜 안돼?",
            },
        ),
        PromptScenario(
            scenario_id="expansion_normal",
            prompt_name="query_expansion",
            variant="normal",
            call_type="structured",
            llm_kind="llm",
            schema=QueryExpansionResult,
            inputs={
                "conversation_history": "사용자는 NLP 기초 과정을 중심으로 학습하고 있다.",
                "data_source_context": (
                    "Available doc_type: slack_qa, pdf, notebook, lecture_transcript, weekly_mission\n"
                    "Available courses: NLP 기초, CV 이론, AI 기초\n"
                    "Available generations: 1기, 2기, 3기"
                ),
                "intent": "COMPLEX_REASONING",
                "clarity": 0.85,
                "specificity": 0.9,
                "question": "NLP 기초 강의 기준으로 positional encoding이 길이 일반화에 미치는 영향 비교 자료 찾아줘.",
            },
        ),
        PromptScenario(
            scenario_id="expansion_boundary",
            prompt_name="query_expansion",
            variant="boundary",
            call_type="structured",
            llm_kind="llm",
            schema=QueryExpansionResult,
            inputs={
                "conversation_history": "사용자는 이전 답변에서 무엇을 더 알고 싶은지 특정하지 않았다.",
                "data_source_context": (
                    "Available doc_type: slack_qa, pdf, notebook, lecture_transcript, weekly_mission\n"
                    "Available courses: NLP 기초, CV 이론, AI 기초\n"
                    "Available generations: 1기, 2기, 3기"
                ),
                "intent": "CLARIFICATION_NEEDED",
                "clarity": 0.2,
                "specificity": 0.3,
                "question": "그거 더 알려줘.",
            },
        ),
        PromptScenario(
            scenario_id="multi_query_normal",
            prompt_name="multi_query_generation",
            variant="normal",
            call_type="structured",
            llm_kind="llm",
            schema=MultiQueryOutput,
            inputs={
                "conversation_history": "No prior conversation.",
                "query": "PyTorch mixed precision training 설정과 오류 해결 방법",
                "num": 4,
            },
        ),
        PromptScenario(
            scenario_id="multi_query_boundary",
            prompt_name="multi_query_generation",
            variant="boundary",
            call_type="structured",
            llm_kind="llm",
            schema=MultiQueryOutput,
            inputs={
                "conversation_history": "사용자는 직전 턴에서 Transformer 학습 속도 최적화 문제를 말했다.",
                "query": "그거 최적화하는 법",
                "num": 2,
            },
        ),
        PromptScenario(
            scenario_id="answer_simple_normal",
            prompt_name="answer_generation_simple",
            variant="normal",
            call_type="text",
            llm_kind="reasoning_llm",
            schema=None,
            inputs={
                "conversation_history": "사용자는 GPU utilization이 낮다고 말했다.",
                "context": (
                    "[doc-1] PyTorch DataLoader의 num_workers를 늘리면 CPU 병렬 로딩이 증가해 GPU 대기시간이 줄 수 있다.\n"
                    "[doc-2] 너무 큰 num_workers는 프로세스 오버헤드를 증가시켜 역효과가 날 수 있다."
                ),
                "question": "num_workers를 어떻게 조절해야 해?",
            },
        ),
        PromptScenario(
            scenario_id="answer_simple_boundary",
            prompt_name="answer_generation_simple",
            variant="boundary",
            call_type="text",
            llm_kind="reasoning_llm",
            schema=None,
            inputs={
                "conversation_history": "No prior conversation.",
                "context": "[doc-1] 이 문서는 모델 학습과 무관한 일반 공지사항이다.",
                "question": "Transformer의 scaled dot-product attention 수식을 알려줘.",
            },
        ),
        PromptScenario(
            scenario_id="answer_complex_normal",
            prompt_name="answer_generation_complex",
            variant="normal",
            call_type="text",
            llm_kind="reasoning_llm",
            schema=None,
            inputs={
                "conversation_history": "사용자는 모델 변형별 설정 차이를 이해하고 싶다고 했다.",
                "context": (
                    "[doc-A] Transformer base hidden size는 512이다.\n"
                    "[doc-B] 특정 실습 노트에서는 hidden size를 768로 설정했다.\n"
                    "[doc-C] 실습 노트의 설정은 모델 변형과 실험 목적에 따라 달라진다."
                ),
                "question": "Transformer hidden size가 512인지 768인지 정리해줘.",
            },
        ),
        PromptScenario(
            scenario_id="answer_complex_boundary",
            prompt_name="answer_generation_complex",
            variant="boundary",
            call_type="text",
            llm_kind="reasoning_llm",
            schema=None,
            inputs={
                "conversation_history": "No prior conversation.",
                "context": "[doc-X] 학습률은 모델 성능에 영향을 준다.",
                "question": "BoostCamp 3기 NLP 과정의 정확한 프로젝트 일정과 제출 마감 시간을 알려줘.",
            },
        ),
        PromptScenario(
            scenario_id="answer_exploratory_normal",
            prompt_name="answer_generation_exploratory",
            variant="normal",
            call_type="text",
            llm_kind="reasoning_llm",
            schema=None,
            inputs={
                "conversation_history": "No prior conversation.",
                "context": (
                    "[doc-1] RAG 학습 순서는 정보검색 기초, 임베딩/인덱싱, 검색 품질평가, 생성 품질평가 순서가 효과적이다.\n"
                    "[doc-2] 초반에는 작은 데이터셋으로 retrieval precision/recall을 먼저 측정하고, 이후 answer grounding을 개선한다."
                ),
                "question": "RAG를 처음 공부하려고 해. 어떤 순서로 학습하면 좋을까?",
            },
        ),
        PromptScenario(
            scenario_id="answer_exploratory_boundary",
            prompt_name="answer_generation_exploratory",
            variant="boundary",
            call_type="text",
            llm_kind="reasoning_llm",
            schema=None,
            inputs={
                "conversation_history": "어시스턴트는 이미 RAG 학습 로드맵 개요를 제공했다.",
                "context": (
                    "[doc-1] 이번 주 실습은 문서 30개로 임베딩 인덱스 구축, 질의 10개로 검색 정확도 측정, 실패 케이스 리포트 작성으로 구성한다.\n"
                    "[doc-2] 각 실습은 하루 단위로 분할해 실행하면 완주율이 높다."
                ),
                "question": "앞에서 말한 계획에서 이번 주에 바로 할 실습만 압축해줘.",
            },
        ),
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prompt_behaviors_gemini_matrix(gemini_llm, gemini_reasoning_llm):
    """Run 16 prompt behavior scenarios against Gemini and persist artifacts."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid4().hex[:8]}"
    started_at = datetime.now().isoformat()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = ARTIFACT_DIR / f"prompt_behavior_gemini_{run_id}.jsonl"
    markdown_path = ARTIFACT_DIR / f"prompt_behavior_gemini_{run_id}.md"

    eval_logger = logger.bind(context="evaluation", suite="prompt_behavior_gemini", run_id=run_id)
    eval_logger.info(f"Prompt behavior suite started: run_id={run_id}")

    all_scenarios = _build_scenarios()
    assert len(all_scenarios) == 16
    scenarios = _select_scenarios(all_scenarios)
    if not scenarios:
        pytest.fail(
            "No scenarios selected. "
            "Check PROMPT_BEHAVIOR_SCENARIOS / PROMPT_BEHAVIOR_PROMPTS / "
            "PROMPT_BEHAVIOR_CHUNK_SIZE / PROMPT_BEHAVIOR_CHUNK_INDEX"
        )
    selected_scenario_ids = [scenario.scenario_id for scenario in scenarios]
    eval_logger.info(
        f"Selected {len(scenarios)} scenario(s): {selected_scenario_ids} "
        f"(max_retries={MAX_RETRY_ATTEMPTS}, timeout={ATTEMPT_TIMEOUT_SECONDS}s, "
        f"rate_limit_per_min={SCENARIO_RATE_LIMIT_PER_MIN})"
    )

    records: list[dict[str, Any]] = []
    min_interval_seconds = (60.0 / SCENARIO_RATE_LIMIT_PER_MIN) if SCENARIO_RATE_LIMIT_PER_MIN > 0 else 0.0
    last_started_at: float | None = None

    for scenario in scenarios:
        if min_interval_seconds > 0 and last_started_at is not None:
            elapsed = time.perf_counter() - last_started_at
            wait_seconds = min_interval_seconds - elapsed
            if wait_seconds > 0:
                eval_logger.info(
                    f"Rate limit active: sleeping {wait_seconds:.2f}s before scenario={scenario.scenario_id}"
                )
                await asyncio.sleep(wait_seconds)

        last_started_at = time.perf_counter()
        record = await _run_scenario(
            scenario=scenario,
            run_id=run_id,
            llm=gemini_llm,
            reasoning_llm=gemini_reasoning_llm,
            eval_logger=eval_logger,
        )
        records.append(record)
        _append_jsonl(jsonl_path, record)

    completed_at = datetime.now().isoformat()
    report_text = _build_markdown_report(
        run_id=run_id,
        started_at=started_at,
        completed_at=completed_at,
        jsonl_path=jsonl_path,
        records=records,
        selected_scenario_ids=selected_scenario_ids,
    )
    markdown_path.write_text(report_text, encoding="utf-8")

    failed_records = [record for record in records if record["status"] != "PASS"]
    passed = len(records) - len(failed_records)

    eval_logger.info(
        f"Prompt behavior suite completed: total={len(records)}, passed={passed}, failed={len(failed_records)}, "
        f"jsonl={jsonl_path}, report={markdown_path}"
    )

    if failed_records:
        failed_ids = ", ".join(record["scenario_id"] for record in failed_records)
        pytest.fail(
            "Prompt behavior suite failed. "
            f"failed_scenarios=[{failed_ids}] "
            f"jsonl={jsonl_path} report={markdown_path}"
        )
