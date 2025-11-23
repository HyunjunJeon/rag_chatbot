#!/usr/bin/env python3
"""
Automated filtering script for Slack Q&A data.
Removes low-quality answers based on predefined rules.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter


# === FILTER RULES ===

# 제거할 필러 패턴 (순수 필러만 있는 경우)
FILLER_PATTERNS = [
    r"^[ㅋㅎㅠㅜㅡ\s]+$",  # ㅋㅋ, ㅎㅎ, ㅠㅠ 등만
    r"^(ㅋ|ㅎ|ㅠ|ㅜ|ㅡ){2,}$",  # 반복되는 자음/모음
]

# 제거할 단순 응답
SIMPLE_RESPONSES = [
    "넵", "네", "ㄴㄴ", "ㅇㅇ", "ㅇㅋ", "오키",
    "감사합니다", "고맙습니다", "감사해요", "고마워요",
    "확인했습니다", "확인해보겠습니다", "알겠습니다",
    "네 감사합니다", "네 고맙습니다",
    "답변 감사합니다", "답변 드렸습니다",
]

# 제거할 무의미한 반응
MEANINGLESS_REACTIONS = [
    r"^화이팅[!]*$",
    r"^아자아자.*화이팅",
    r"^부담은\.\.",
    r"^헤헤+$",
    r"^흐흐+$",
    r"도움을 드리지 못해서 아쉽",
]

# 제거할 메타 질문
META_QUESTIONS = [
    "질문 가능한가요",
    "질문이 있습니다",
    "질문 하나만 할게요",
    "질문 드려도 될까요",
]

# 보존 신호 (이것들이 있으면 길이가 짧아도 보존)
PRESERVE_SIGNALS = [
    "http://", "https://",  # URL
    "```",  # 코드 블록
    "github.com",
    "stackoverflow",
]


def should_remove_answer(answer_text: str) -> tuple[bool, str]:
    """
    답변을 제거해야 하는지 판단합니다.

    Returns:
        (should_remove: bool, reason: str)
    """
    text = answer_text.strip()

    # 보존 신호가 있으면 무조건 보존
    for signal in PRESERVE_SIGNALS:
        if signal in text.lower():
            return False, "contains_preserve_signal"

    # 1. 너무 짧은 답변 (15자 미만)
    if len(text) < 15:
        return True, "too_short"

    # 2. 필러만 포함
    for pattern in FILLER_PATTERNS:
        if re.match(pattern, text):
            return True, "only_filler"

    # 3. 단순 응답
    text_lower = text.lower().strip()
    if text_lower in [r.lower() for r in SIMPLE_RESPONSES]:
        return True, "simple_response"

    # 4. 무의미한 반응
    for pattern in MEANINGLESS_REACTIONS:
        if re.search(pattern, text):
            return True, "meaningless_reaction"

    # 5. 메타 질문
    for meta in META_QUESTIONS:
        if meta in text:
            return True, "meta_question"

    # 6. 길이는 적당하지만 내용이 없음 (필러 비율 체크)
    filler_chars = sum(text.count(c) for c in "ㅋㅎㅠㅜㅡ헤흐")
    if len(text) < 30 and filler_chars / len(text) > 0.3:
        return True, "high_filler_ratio"

    return False, "keep"


def filter_qa_pairs(qa_pairs: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Q&A 쌍의 리스트를 필터링합니다.

    Returns:
        (filtered_qa_pairs, statistics)
    """
    filtered_pairs = []
    stats = Counter()

    for qa in qa_pairs:
        original_answer_count = len(qa.get("answers", []))
        filtered_answers = []

        for answer in qa.get("answers", []):
            answer_text = answer.get("text", "")
            should_remove, reason = should_remove_answer(answer_text)

            if should_remove:
                stats[f"removed_{reason}"] += 1
                stats["total_removed"] += 1
            else:
                filtered_answers.append(answer)
                stats["total_kept"] += 1

        # Q&A 쌍을 보존하는 조건:
        # 1. 최소 1개 이상의 답변이 남아있어야 함
        if filtered_answers:
            qa_copy = qa.copy()
            qa_copy["answers"] = filtered_answers
            filtered_pairs.append(qa_copy)
            stats["qa_pairs_kept"] += 1
        else:
            stats["qa_pairs_removed_no_answers"] += 1

    stats["qa_pairs_total"] = len(qa_pairs)

    return filtered_pairs, dict(stats)


def filter_course_file(input_path: Path, output_path: Path) -> Dict[str, Any]:
    """
    단일 코스 파일을 필터링합니다.

    Returns:
        Statistics dictionary
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # _summary.json은 메타데이터만 있으므로 스킵
    if input_path.name == "_summary.json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {"file": input_path.name, "skipped": True}

    # Q&A 쌍 필터링
    original_qa_pairs = data.get("qa_pairs", [])
    filtered_qa_pairs, stats = filter_qa_pairs(original_qa_pairs)

    # 결과 저장
    output_data = data.copy()
    output_data["qa_pairs"] = filtered_qa_pairs

    # 메타데이터 업데이트
    if "metadata" in output_data:
        output_data["metadata"]["total_qa_pairs"] = len(filtered_qa_pairs)
        output_data["metadata"]["filtering_applied"] = True
        output_data["metadata"]["original_qa_pairs"] = len(original_qa_pairs)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    stats["file"] = input_path.name
    return stats


def main():
    """메인 실행 함수"""
    # 경로 설정
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "document_chunks" / "slack_qa_merged"
    output_dir = base_dir / "document_chunks" / "slack_qa_auto_filtered"

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Slack Q&A Automated Filtering")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()

    # 모든 JSON 파일 처리
    all_stats = []
    json_files = sorted(input_dir.glob("*.json"))

    for input_file in json_files:
        output_file = output_dir / input_file.name

        print(f"Processing: {input_file.name}...", end=" ")
        stats = filter_course_file(input_file, output_file)
        all_stats.append(stats)

        if stats.get("skipped"):
            print("SKIPPED (metadata file)")
        else:
            removed = stats.get("total_removed", 0)
            kept = stats.get("total_kept", 0)
            total = removed + kept
            removed_pct = (removed / total * 100) if total > 0 else 0
            print(f"✓ Removed {removed}/{total} answers ({removed_pct:.1f}%)")

    # 전체 통계 출력
    print()
    print("=" * 60)
    print("Overall Statistics")
    print("=" * 60)

    total_removed = sum(s.get("total_removed", 0) for s in all_stats)
    total_kept = sum(s.get("total_kept", 0) for s in all_stats)
    total_answers = total_removed + total_kept

    total_qa_removed = sum(s.get("qa_pairs_removed_no_answers", 0) for s in all_stats)
    total_qa_kept = sum(s.get("qa_pairs_kept", 0) for s in all_stats)
    total_qa_pairs = sum(s.get("qa_pairs_total", 0) for s in all_stats)

    print(f"Total answers processed: {total_answers:,}")
    print(f"  - Kept:    {total_kept:,} ({total_kept/total_answers*100:.1f}%)")
    print(f"  - Removed: {total_removed:,} ({total_removed/total_answers*100:.1f}%)")
    print()
    print(f"Total Q&A pairs processed: {total_qa_pairs:,}")
    print(f"  - Kept:    {total_qa_kept:,} ({total_qa_kept/total_qa_pairs*100:.1f}%)")
    print(f"  - Removed: {total_qa_removed:,} ({total_qa_removed/total_qa_pairs*100:.1f}%)")
    print()

    # 제거 이유별 통계
    print("Removal reasons:")
    removal_reasons = Counter()
    for stat in all_stats:
        for key, value in stat.items():
            if key.startswith("removed_"):
                removal_reasons[key] += value

    for reason, count in removal_reasons.most_common():
        reason_name = reason.replace("removed_", "")
        print(f"  - {reason_name}: {count:,}")

    # 통계를 JSON으로 저장
    stats_file = output_dir / "_filtering_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_answers": total_answers,
                "answers_kept": total_kept,
                "answers_removed": total_removed,
                "total_qa_pairs": total_qa_pairs,
                "qa_pairs_kept": total_qa_kept,
                "qa_pairs_removed": total_qa_removed,
            },
            "removal_reasons": dict(removal_reasons),
            "per_file_stats": all_stats,
        }, f, ensure_ascii=False, indent=2)

    print()
    print(f"✓ Filtering complete! Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()
