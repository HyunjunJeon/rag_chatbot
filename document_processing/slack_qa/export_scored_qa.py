"""Slack Q&A 품질 평가 결과를 하나의 JSON으로 통합하는 유틸리티.

- 입력: document_chunks/slack_qa_scored 아래의 *_merged.json 파일들
- 출력: document_chunks/slack_qa_scored/all_scored_qa.json
  - high/medium/low 등급 Q&A만 포함 (remove/error 는 포함되지 않음)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _clean_text(text: str) -> str:
    """Slack 멘션(<@UXXXXXX>) 등을 제거한 텍스트를 반환."""
    if not isinstance(text, str):
        return ""
    # 기본 Slack 멘션 패턴 제거
    cleaned = re.sub(r"<@[^>]+>", "", text)
    # Slack 이모지 코드 제거 (:emoji_name:)
    cleaned = re.sub(r":[a-zA-Z0-9_+-]+:", "", cleaned)

    # 이전 Bot 시스템의 고정 안내 문구(예: "부덕이_답변bot입니다 ...") 제거
    # - 보통 답변 텍스트 가장 앞쪽에 위치
    # - 해당 문구와 그 직후의 안내 문단까지 잘라낸 뒤, 나머지 실제 답변만 남긴다
    if "부덕이" in cleaned and "답변" in cleaned and "bot" in cleaned:
        # "부덕이 ... bot" 이 포함된 첫 문단(두 개의 연속 개행 전까지)을 제거
        cleaned = re.sub(
            r"^.*부덕이.*답변.*bot.*?(?:\n\n|$)",
            "",
            cleaned,
            flags=re.DOTALL,
        )

    return cleaned.strip()


def _sanitize_qa_pair(qa: dict[str, Any]) -> dict[str, Any]:
    """개별 Q&A에서 개인정보를 제거하고 텍스트를 정리한다."""
    sanitized = qa.copy()

    # 질문 부분 정리
    question = dict(qa.get("question", {}) or {})
    # 민감 정보 제거
    question.pop("user", None)
    question.pop("user_name", None)
    if "text" in question:
        question["text"] = _clean_text(question.get("text", ""))
    sanitized["question"] = question

    # 답변 리스트 정리
    answers: list[dict[str, Any]] = []
    for answer in qa.get("answers", []) or []:
        answer_clean = dict(answer or {})
        answer_clean.pop("user", None)
        answer_clean.pop("user_name", None)
        answer_clean.pop("metadata", None)
        if "text" in answer_clean:
            answer_clean["text"] = _clean_text(answer_clean.get("text", ""))
        answers.append(answer_clean)
    sanitized["answers"] = answers

    # reply_count 검증: question 메타데이터 기준으로 answers 개수와 비교
    metadata = question.get("metadata") or {}
    reply_count = metadata.get("reply_count")
    if isinstance(reply_count, int) and reply_count != len(answers):
        print(
            "[WARN] reply_count mismatch: "
            f"ts={question.get('timestamp')} "
            f"expected={reply_count}, actual={len(answers)}"
        )

    return sanitized


def collect_all_scored_qa(scored_dir: Path) -> dict[str, Any]:
    """slack_qa_scored 디렉토리에서 모든 high/medium/low Q&A를 모은다."""
    qa_pairs: list[dict[str, Any]] = []
    stats: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    courses: set[str] = set()
    source_files: list[dict[str, Any]] = []

    for json_file in sorted(scored_dir.glob("*.json")):
        # 내부 테스트/요약 파일 등은 스킵
        if json_file.name.startswith("_"):
            continue
        if json_file.name.startswith("sample_"):
            continue
        if json_file.name == "all_scored_qa.json":
            continue

        with json_file.open(encoding="utf-8") as f:
            data = json.load(f)

        course = data.get("course")
        if isinstance(course, str):
            courses.add(course)

        metadata = data.get("metadata", {})
        quality_stats = metadata.get("quality_stats", {})
        for key in stats:
            value = quality_stats.get(key, 0) or 0
            if isinstance(value, int):
                stats[key] += value

        file_qa_pairs = data.get("qa_pairs", [])
        for qa in file_qa_pairs:
            quality_score = qa.get("quality_score") or {}
            overall = quality_score.get("overall_quality")
            if overall in ("high", "medium", "low"):
                qa_pairs.append(_sanitize_qa_pair(qa))

        source_files.append(
            {
                "filename": json_file.name,
                "course": course,
                "qa_count": len(file_qa_pairs),
            }
        )

    metadata_out: dict[str, Any] = {
        "total_qa_pairs": len(qa_pairs),
        "by_quality": stats,
        "courses": sorted(courses),
        "source_files": source_files,
    }

    return {"metadata": metadata_out, "qa_pairs": qa_pairs}


def main() -> None:
    # repo 루트 기준: document_chunks/slack_qa_scored
    project_root = Path(__file__).resolve().parents[2]
    scored_dir = project_root / "document_chunks" / "slack_qa_scored"
    output_path = scored_dir / "all_scored_qa.json"

    print("\n🚀 Exporting all scored Slack Q&A (high/medium/low only)\n")
    print(f"입력 디렉토리: {scored_dir}")
    print(f"출력 파일:     {output_path}\n")

    if not scored_dir.exists():
        raise SystemExit(f"Input directory not found: {scored_dir}")

    aggregated = collect_all_scored_qa(scored_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)

    meta = aggregated.get("metadata", {})
    print("완료!")
    print(f"  - 총 Q&A 개수: {meta.get('total_qa_pairs', 0)}")
    by_quality = meta.get("by_quality", {})
    print(
        "  - 품질별 통계: "
        f"high={by_quality.get('high', 0)}, "
        f"medium={by_quality.get('medium', 0)}, "
        f"low={by_quality.get('low', 0)}"
    )
    print(f"  - 과정 수: {len(meta.get('courses', []))}")
    print(f"  - 원본 파일 수: {len(meta.get('source_files', []))}")


if __name__ == "__main__":
    main()
