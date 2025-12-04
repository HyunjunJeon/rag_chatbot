"""
강의 녹취록 전체 처리 스크립트.

processed_combined 디렉토리의 모든 JSON 파일을 청킹하여
document_chunks/lecture_transcript_chunks/ 에 저장합니다.

사용법:
    python document_processing/lecture_transcript/process_all_transcripts.py
"""

import json
import sys
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from document_processing.lecture_transcript.lecture_transcript_chunker import (
    LectureTranscriptChunker,
)


def main() -> None:
    """메인 처리 함수."""
    print("=" * 70)
    print("🎙️  강의 녹취록 전체 처리")
    print("=" * 70)

    # 경로 설정
    input_dir = PROJECT_ROOT / "processed_combined"
    output_dir = PROJECT_ROOT / "document_chunks" / "lecture_transcript_chunks"

    if not input_dir.exists():
        print(f"❌ 입력 디렉토리가 없습니다: {input_dir}")
        sys.exit(1)

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # 청커 초기화
    chunker = LectureTranscriptChunker(
        chunk_size=1000,  # 약 1000 토큰
        chunk_overlap=150,  # 150 토큰 오버랩
        min_chunk_size=100,  # 최소 100 토큰
    )

    # 전체 처리
    print(f"\n📂 입력: {input_dir}")
    print(f"📁 출력: {output_dir}")

    all_chunks = chunker.process_directory(input_dir, verbose=True)

    if not all_chunks:
        print("\n⚠️  청크가 생성되지 않았습니다.")
        return

    # 과목별 통계
    course_stats: dict[str, int] = {}
    for chunk in all_chunks:
        course = chunk.metadata.get("course", "기타")
        course_stats[course] = course_stats.get(course, 0) + 1

    print("\n📊 과목별 청크 수:")
    for course, count in sorted(course_stats.items(), key=lambda x: -x[1]):
        print(f"   {course}: {count}개")

    # 파일별로 청크 저장
    file_chunks: dict[str, list] = {}
    for chunk in all_chunks:
        source = chunk.metadata.get("source_file", "unknown.json")
        if source not in file_chunks:
            file_chunks[source] = []
        file_chunks[source].append(chunk.to_dict())

    print(f"\n💾 {len(file_chunks)}개 파일로 저장 중...")

    for source_file, chunks in file_chunks.items():
        # 파일명에서 확장자 제거하고 _chunks.json 추가
        base_name = Path(source_file).stem
        output_file = output_dir / f"{base_name}_chunks.json"

        # 메타데이터 포함
        output_data = {
            "source_file": source_file,
            "lecture_name": chunks[0]["metadata"].get("lecture_name", ""),
            "course": chunks[0]["metadata"].get("course", ""),
            "total_chunks": len(chunks),
            "chunks": chunks,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    # 전체 요약 파일 저장
    summary = {
        "total_files": len(file_chunks),
        "total_chunks": len(all_chunks),
        "course_stats": course_stats,
        "files": [
            {
                "source_file": source,
                "lecture_name": chunks[0]["metadata"].get("lecture_name", ""),
                "course": chunks[0]["metadata"].get("course", ""),
                "chunk_count": len(chunks),
            }
            for source, chunks in file_chunks.items()
        ],
    }

    summary_file = output_dir / "_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 저장 완료: {output_dir}")
    print(f"   - 파일: {len(file_chunks)}개")
    print(f"   - 청크: {len(all_chunks)}개")
    print(f"   - 요약: {summary_file.name}")

    # 샘플 출력
    print("\n📝 샘플 청크 (처음 2개):")
    for chunk in all_chunks[:2]:
        print(f"\n   [{chunk.id}]")
        print(f"   과목: {chunk.metadata.get('course')}")
        print(f"   강의: {chunk.metadata.get('lecture_name')}")
        print(f"   토큰: ~{chunk.token_estimate}")
        preview = chunk.content[:200].replace("\n", " ")
        print(f"   내용: {preview}...")


if __name__ == "__main__":
    main()
