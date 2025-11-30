#!/usr/bin/env python3
"""
Slack Q&A 품질 평가 CLI 스크립트.

사용법:
    python -m document_processing.slack_qa.evaluate_quality \
        --input document_chunks/slack_qa_merged \
        --output document_chunks/slack_qa_scored \
        --checkpoint quality_checkpoint.json
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from .quality_evaluator import QualityEvaluator
from .quality_schemas import extract_for_evaluation
from .batch_processor import BatchConfig, BatchProcessor


def load_merged_qa_file(file_path: Path) -> list[dict[str, Any]]:
    """병합된 Q&A JSON 파일 로드."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("qa_pairs", [])


def save_scored_qa_file(
    file_path: Path,
    original_data: dict,
    results: dict[str, dict],
) -> dict[str, int]:
    """평가된 Q&A 파일 저장."""
    qa_pairs = original_data.get("qa_pairs", [])
    scored_pairs = []
    stats = {"high": 0, "medium": 0, "low": 0, "remove": 0, "error": 0}

    for qa in qa_pairs:
        original_id = qa.get("question", {}).get("timestamp", "")
        result = results.get(original_id)

        if result is None:
            stats["error"] += 1
            continue

        overall = result.get("overall_quality", "remove")
        if overall == "remove":
            stats["remove"] += 1
            continue  # 제거 대상은 저장하지 않음

        stats[overall] += 1

        # 점수를 Q&A에 추가
        qa_with_score = qa.copy()
        qa_with_score["quality_score"] = result
        scored_pairs.append(qa_with_score)

    # 메타데이터 업데이트
    output_data = original_data.copy()
    output_data["qa_pairs"] = scored_pairs
    output_data["metadata"] = output_data.get("metadata", {})
    output_data["metadata"]["quality_filtered"] = True
    output_data["metadata"]["quality_stats"] = stats
    output_data["metadata"]["original_count"] = len(qa_pairs)
    output_data["metadata"]["filtered_count"] = len(scored_pairs)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return stats


async def process_file(
    input_path: Path,
    output_path: Path,
    processor: BatchProcessor,
) -> dict[str, int]:
    """단일 파일 처리."""
    print(f"Processing: {input_path.name}")

    with open(input_path, encoding="utf-8") as f:
        original_data = json.load(f)

    qa_pairs = original_data.get("qa_pairs", [])
    if not qa_pairs:
        print(f"  No Q&A pairs in {input_path.name}")
        return {"skipped": 1}

    # 평가 입력으로 변환
    inputs = [
        extract_for_evaluation(qa, source_file=input_path.name)
        for qa in qa_pairs
    ]

    # 배치 처리
    def progress(current: int, total: int) -> None:
        print(f"  Progress: {current}/{total}")

    await processor.process_all(inputs, progress_callback=progress)

    # 결과 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = save_scored_qa_file(output_path, original_data, processor.state.results)

    print(f"  Done: {stats}")
    return stats


async def main(
    input_dir: Path,
    output_dir: Path,
    checkpoint_path: Path | None,
    batch_size: int,
):
    """메인 실행 함수."""
    print("=" * 60)
    print("Slack Q&A LLM Quality Evaluation")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model:  Clova X HCX-007")
    print()

    # 평가기 및 프로세서 초기화
    evaluator = QualityEvaluator()
    config = BatchConfig(batch_size=batch_size)
    processor = BatchProcessor(
        evaluator=evaluator,
        config=config,
        checkpoint_path=checkpoint_path,
    )

    # 입력 파일 목록
    input_files = sorted(input_dir.glob("*.json"))
    input_files = [f for f in input_files if not f.name.startswith("_")]

    total_stats: dict[str, int] = {"high": 0, "medium": 0, "low": 0, "remove": 0, "error": 0}

    for input_file in input_files:
        output_file = output_dir / input_file.name
        stats = await process_file(input_file, output_file, processor)

        for key in total_stats:
            total_stats[key] += stats.get(key, 0)

    # 최종 통계
    print()
    print("=" * 60)
    print("Final Statistics")
    print("=" * 60)
    for key, value in total_stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Slack Q&A quality")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory with merged Q&A files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for scored Q&A files",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint file path for resume",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing",
    )

    args = parser.parse_args()

    asyncio.run(
        main(
            input_dir=args.input,
            output_dir=args.output,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
        )
    )
