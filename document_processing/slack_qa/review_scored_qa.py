"""CLI 기반 Slack Q&A 수동 리뷰 도구.

- 입력: document_chunks/slack_qa_scored/all_scored_qa.json
- 상태: document_chunks/slack_qa_scored/all_scored_qa_review_state.json
- 출력(export 시): document_chunks/slack_qa_scored/all_scored_qa_reviewed.json

한 번에 전체를 자동 필터링하지 않고,
Q&A 한 건씩 화면에 보여준 뒤 사용자가 keep/drop/skip 을 선택하는 구조입니다.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal

Decision = Literal["keep", "drop", "skip"]


@dataclass
class ReviewState:
    """리뷰 진행 상태.

    Attributes:
        file: 리뷰 대상 파일 이름
        total: 전체 Q&A 개수
        decisions: 인덱스별 결정 (keep/drop/skip)
    """

    file: str
    total: int
    decisions: Dict[int, Decision]

    @classmethod
    def load(cls, path: Path, total: int, file_name: str) -> "ReviewState":
        if not path.exists():
            return cls(file=file_name, total=total, decisions={})

        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        decisions_raw = data.get("decisions", {})
        decisions: Dict[int, Decision] = {}
        for k, v in decisions_raw.items():
            try:
                idx = int(k)
            except ValueError:
                continue
            if v in ("keep", "drop", "skip"):
                decisions[idx] = v  # type: ignore[assignment]

        saved_total = data.get("total", total)
        saved_file = data.get("file", file_name)

        if saved_total != total or saved_file != file_name:
            print(
                "[WARN] 기존 리뷰 상태의 메타데이터가 현재 파일과 다릅니다. "
                "(파일 또는 Q&A 수가 변경된 것 같아요.)"
            )

        return cls(file=file_name, total=total, decisions=decisions)

    def save(self, path: Path) -> None:
        data = {
            "file": self.file,
            "total": self.total,
            "decisions": {str(k): v for k, v in self.decisions.items()},
        }
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)


def _next_index(state: ReviewState) -> int | None:
    """아직 결정되지 않은 다음 인덱스를 반환."""
    for idx in range(state.total):
        if idx not in state.decisions:
            return idx
    return None


def _print_qa(index: int, total: int, qa: Dict[str, Any]) -> None:
    """단일 Q&A를 사람이 보기 좋게 출력."""
    meta = {
        "index": index,
        "total": total,
        "course": qa.get("course"),
        "generation": qa.get("generation"),
        "date": qa.get("date"),
    }
    quality = (qa.get("quality_score") or {}).get("overall_quality")

    print("\n" + "=" * 80)
    print(
        f"[Q&A {index + 1}/{total}] course={meta['course']} | gen={meta['generation']} | date={meta['date']} | quality={quality}"
    )
    print("=" * 80)

    question = qa.get("question", {}) or {}
    print("\n[Question]")
    print(question.get("text", "").strip())

    answers = qa.get("answers", []) or []
    if not answers:
        print("\n[Answers] (없음)")
    else:
        print("\n[Answers]")
        for i, answer in enumerate(answers, start=1):
            print("-" * 40)
            print(f"(Answer {i})")
            print(answer.get("text", "").strip())


def review_interactive(data_path: Path, state_path: Path) -> None:
    """터미널에서 한 건씩 Q&A를 검토하는 인터랙티브 모드."""
    if not data_path.exists():
        raise SystemExit(f"Input file not found: {data_path}")

    with data_path.open(encoding="utf-8") as f:
        data = json.load(f)

    qa_pairs = data.get("qa_pairs", [])
    if not isinstance(qa_pairs, list):
        raise SystemExit("Invalid all_scored_qa.json: 'qa_pairs' must be a list")

    total = len(qa_pairs)
    if total == 0:
        print("Q&A가 없습니다.")
        return

    state = ReviewState.load(state_path, total=total, file_name=data_path.name)

    print("\n📖 수동 리뷰 모드 시작")
    print(f"  - 파일: {data_path}")
    print(f"  - 총 Q&A: {total}")
    print(f"  - 이미 결정된 항목: {len(state.decisions)}")
    print("\n입력 키:")
    print("  k = keep (유지)")
    print("  d = drop (삭제)")
    print("  s = skip (보류/건너뛰기)")
    print("  q = quit (종료)")

    while True:
        idx = _next_index(state)
        if idx is None:
            print("\n✅ 더 이상 검토할 Q&A가 없습니다.")
            break

        qa = qa_pairs[idx]
        _print_qa(idx, total, qa)

        while True:
            cmd = input("\n[k]eep / [d]rop / [s]kip / [q]uit > ").strip().lower()

            if cmd == "q":
                print("\n리뷰를 종료합니다. 진행 상태는 저장되었습니다.")
                state.save(state_path)
                return

            if cmd in {"k", "d", "s"}:
                decision: Decision
                if cmd == "k":
                    decision = "keep"
                elif cmd == "d":
                    decision = "drop"
                else:
                    decision = "skip"

                state.decisions[idx] = decision
                state.save(state_path)
                break

            print("지원하지 않는 입력입니다. k/d/s/q 중 하나를 입력해주세요.")


def export_reviewed(data_path: Path, state_path: Path, output_path: Path) -> None:
    """리뷰 결과를 적용하여 새로운 JSON 파일을 생성."""
    if not data_path.exists():
        raise SystemExit(f"Input file not found: {data_path}")

    with data_path.open(encoding="utf-8") as f:
        data = json.load(f)

    qa_pairs = data.get("qa_pairs", [])
    total = len(qa_pairs)

    state = ReviewState.load(state_path, total=total, file_name=data_path.name)
    if not state.decisions:
        print("리뷰 상태가 없습니다. 먼저 인터랙티브 리뷰를 실행해 주세요.")
        return

    kept: list[Dict[str, Any]] = []
    dropped_count = 0

    for idx, qa in enumerate(qa_pairs):
        decision = state.decisions.get(idx)
        if decision == "drop":
            dropped_count += 1
            continue
        # keep, skip, 또는 미결정(None)은 모두 유지 쪽으로 처리 (보수적으로 유지)
        kept.append(qa)

    output = data.copy()
    output["qa_pairs"] = kept
    output.setdefault("metadata", {})["manual_review"] = {
        "total": total,
        "kept": len(kept),
        "dropped": dropped_count,
        "undecided": total - len(state.decisions),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n✅ 리뷰 결과 적용 완료")
    print(f"  - 원본 Q&A: {total}")
    print(f"  - 삭제됨(drop): {dropped_count}")
    print(f"  - 유지됨(keep/skip/미결정): {len(kept)}")
    print(f"  - 출력 파일: {output_path}")


def main() -> None:
    # repo 루트 기준: document_chunks/slack_qa_scored
    project_root = Path(__file__).resolve().parents[2]
    default_data = project_root / "document_chunks" / "slack_qa_scored" / "all_scored_qa.json"
    default_state = (
        project_root / "document_chunks" / "slack_qa_scored" / "all_scored_qa_review_state.json"
    )
    default_output = (
        project_root / "document_chunks" / "slack_qa_scored" / "all_scored_qa_reviewed.json"
    )

    parser = argparse.ArgumentParser(
        description="Manually review Slack Q&A pairs from all_scored_qa.json",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=default_data,
        help="Input all_scored_qa.json path",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=default_state,
        help="Review state file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output reviewed JSON path (for --export)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Apply review decisions and write a new JSON file",
    )

    args = parser.parse_args()

    if args.export:
        export_reviewed(args.file, args.state, args.output)
    else:
        review_interactive(args.file, args.state)


if __name__ == "__main__":
    main()
