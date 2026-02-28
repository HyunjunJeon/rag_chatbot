#!/usr/bin/env python3
"""
Slack Q&A 품질 평가 CLI 스크립트 (Rate Limiting 지원).

사용법:
    # 기본 실행 (분당 30 요청)
    python -m document_processing.slack_qa.evaluate_quality \
        --input document_chunks/slack_qa_merged \
        --output document_chunks/slack_qa_scored \
        --checkpoint quality_checkpoint.json

    # Rate limit 조정 (분당 20 요청, 더 안전)
    python -m document_processing.slack_qa.evaluate_quality \
        --input document_chunks/slack_qa_merged \
        --output document_chunks/slack_qa_scored \
        --checkpoint quality_checkpoint.json \
        --rate-limit 20

    # 중단 후 재시작 (체크포인트에서 자동 재개)
    python -m document_processing.slack_qa.evaluate_quality \
        --input document_chunks/slack_qa_merged \
        --output document_chunks/slack_qa_scored \
        --checkpoint quality_checkpoint.json
"""

import argparse
import asyncio
import json
import time
from datetime import timedelta
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


def format_time(seconds: float) -> str:
    """초를 읽기 쉬운 형식으로 변환."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return str(timedelta(seconds=int(seconds)))[2:]  # MM:SS
    else:
        return str(timedelta(seconds=int(seconds)))  # H:MM:SS


async def process_files(
    input_dir: Path,
    output_dir: Path,
    processor: BatchProcessor,
    config: BatchConfig,
) -> dict[str, int]:
    """모든 파일 처리."""
    # 입력 파일 목록
    input_files = sorted(input_dir.glob("*.json"))
    input_files = [f for f in input_files if not f.name.startswith("_")]

    if not input_files:
        print("No input files found!")
        return {}

    # 전체 Q&A 수집
    all_inputs = []
    file_qa_map: dict[str, tuple[Path, dict]] = {}

    for input_file in input_files:
        with open(input_file, encoding="utf-8") as f:
            original_data = json.load(f)

        qa_pairs = original_data.get("qa_pairs", [])
        for qa in qa_pairs:
            eval_input = extract_for_evaluation(qa, source_file=input_file.name)
            all_inputs.append(eval_input)
            file_qa_map[eval_input.original_id] = (input_file, original_data)

    total_count = len(all_inputs)
    already_processed = len(processor.state.processed_ids)
    remaining = total_count - already_processed

    print(f"\nTotal Q&A pairs: {total_count}")
    print(f"Already processed: {already_processed}")
    print(f"Remaining: {remaining}")

    if remaining == 0:
        print("\nAll items already processed!")
    else:
        # 예상 소요 시간 계산
        time_per_item = config.min_request_interval
        estimated_time = remaining * time_per_item
        print(f"\nEstimated time: {format_time(estimated_time)}")
        print(f"Rate limit: {config.requests_per_minute} requests/minute")
        print(f"Checkpoint interval: every {config.checkpoint_interval} items")
        print()

    # 진행 상황 표시
    start_time = time.time()
    last_progress_time = start_time

    def progress_callback(current: int, total: int) -> None:
        nonlocal last_progress_time

        now = time.time()
        elapsed = now - start_time
        items_done = current - already_processed

        if items_done > 0:
            time_per_item_actual = elapsed / items_done
            remaining_items = total - current
            eta = remaining_items * time_per_item_actual
        else:
            eta = 0

        # 10초마다 또는 10개마다 진행상황 출력
        if now - last_progress_time >= 10 or current % 10 == 0:
            percent = current / total * 100
            print(
                f"  Progress: {current}/{total} ({percent:.1f}%) "
                f"| Elapsed: {format_time(elapsed)} "
                f"| ETA: {format_time(eta)}"
            )
            last_progress_time = now

    # 배치 처리
    await processor.process_all(all_inputs, progress_callback=progress_callback)

    # 파일별 결과 저장
    print("\nSaving results...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 파일별로 결과 그룹화
    file_results: dict[Path, dict] = {}
    for original_id, result in processor.state.results.items():
        if original_id in file_qa_map:
            input_file, original_data = file_qa_map[original_id]
            if input_file not in file_results:
                file_results[input_file] = {"original_data": original_data, "results": {}}
            file_results[input_file]["results"][original_id] = result

    total_stats: dict[str, int] = {"high": 0, "medium": 0, "low": 0, "remove": 0, "error": 0}

    for input_file, data in file_results.items():
        output_file = output_dir / input_file.name
        stats = save_scored_qa_file(output_file, data["original_data"], data["results"])
        print(f"  Saved: {output_file.name} - {stats}")

        for key in total_stats:
            total_stats[key] += stats.get(key, 0)

    return total_stats


async def main(
    input_dir: Path,
    output_dir: Path,
    checkpoint_path: Path | None,
    requests_per_minute: int,
    checkpoint_interval: int,
):
    """메인 실행 함수."""
    print("=" * 60)
    print("Slack Q&A LLM Quality Evaluation")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print("Model:  Clova X HCX-007")
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")

    # 설정
    config = BatchConfig(
        batch_size=1,  # Rate limiting을 위해 순차 처리
        requests_per_minute=requests_per_minute,
        checkpoint_interval=checkpoint_interval,
        max_retries=3,
        retry_delay=2.0,
    )

    # 평가기 및 프로세서 초기화
    evaluator = QualityEvaluator()
    processor = BatchProcessor(
        evaluator=evaluator,
        config=config,
        checkpoint_path=checkpoint_path,
    )

    try:
        total_stats = await process_files(input_dir, output_dir, processor, config)

        # 최종 통계
        print()
        print("=" * 60)
        print("Final Statistics")
        print("=" * 60)
        for key, value in total_stats.items():
            print(f"  {key}: {value}")

        # 처리 통계
        stats = processor.get_stats()
        print()
        print(f"  Success rate: {stats['success_rate']:.1f}%")
        if stats["errors"] > 0:
            print(f"  Errors: {stats['errors']} (check checkpoint for details)")

    except KeyboardInterrupt:
        print("\n\n[Interrupted] Saving checkpoint...")
        if checkpoint_path:
            processor.state.save(checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")
            print("Run the same command to resume from this point.")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Slack Q&A quality with LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m document_processing.slack_qa.evaluate_quality \\
      --input document_chunks/slack_qa_merged \\
      --output document_chunks/slack_qa_scored \\
      --checkpoint quality_checkpoint.json

  # With slower rate limit (safer for API limits)
  python -m document_processing.slack_qa.evaluate_quality \\
      --input document_chunks/slack_qa_merged \\
      --output document_chunks/slack_qa_scored \\
      --checkpoint quality_checkpoint.json \\
      --rate-limit 20 \\
      --checkpoint-interval 5
        """,
    )
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
        help="Checkpoint file path for resume (recommended)",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=30,
        help="Requests per minute (default: 30, lower is safer)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N items (default: 10)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(
            main(
                input_dir=args.input,
                output_dir=args.output,
                checkpoint_path=args.checkpoint,
                requests_per_minute=args.rate_limit,
                checkpoint_interval=args.checkpoint_interval,
            )
        )
    except KeyboardInterrupt:
        print("\nAborted by user.")
