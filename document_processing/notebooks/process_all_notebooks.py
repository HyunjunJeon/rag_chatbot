"""
모든 Jupyter Notebook 파일을 일괄 처리하는 메인 스크립트.

practice/ 및 home_work/ 디렉토리의 모든 .ipynb 파일을 처리하여
RAG용 청크를 추출하고 저장합니다.

사용법:
    # 증분 업데이트 (기본)
    python process_all_notebooks.py

    # 전체 재처리
    python process_all_notebooks.py --recreate

    # 변경사항만 확인 (dry-run)
    python process_all_notebooks.py --dry-run
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# common 모듈 import을 위한 경로 추가
DOC_PROCESSING_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(DOC_PROCESSING_DIR))

from notebook_chunker import NotebookChunk, NotebookChunker
from notebook_loader import FileType, NotebookLoader, ParsedNotebook
from common.versioning import VersionInfo, save_version_file, load_version_file


def process_notebooks(
    input_dirs: list[Path],
    output_dir: Path,
    solution_only: bool = True,
    max_tokens: int = 500,
    incremental: bool = True,
    dry_run: bool = False,
) -> dict[str, any]:
    """
    여러 디렉토리의 노트북을 일괄 처리합니다.

    Args:
        input_dirs: 입력 디렉토리 리스트
        output_dir: 출력 디렉토리
        solution_only: 정답 파일만 코드 포함
        max_tokens: 청크 최대 토큰 수
        incremental: True면 변경된 파일만 처리
        dry_run: True면 변경사항만 확인 (실제 처리 안함)

    Returns:
        처리 통계 딕셔너리
    """
    print("=" * 80)
    print("📓 Jupyter Notebook 일괄 처리 시작")
    print("=" * 80)
    print(f"\n입력 디렉토리: {[str(d) for d in input_dirs]}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"정답 파일만 코드 포함: {solution_only}")
    print(f"최대 토큰: {max_tokens}")
    mode_str = "🔄 증분 업데이트" if incremental else "🔁 전체 재처리"
    if dry_run:
        mode_str += " (DRY-RUN)"
    print(f"📌 모드: {mode_str}\n")

    # 로더와 청커 초기화
    loader = NotebookLoader()
    chunker = NotebookChunker(
        max_tokens=max_tokens,
        min_tokens=50,
        include_outputs=True,
        max_output_lines=30,
        solution_only=solution_only,
    )

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # 기존 버전 정보 로드 (증분 업데이트용)
    version_file = output_dir / "_version.json"
    existing_version = load_version_file(version_file) if incremental else None

    # 통계
    stats = {
        "total_notebooks": 0,
        "solution_notebooks": 0,
        "problem_notebooks": 0,
        "total_chunks": 0,
        "by_course": {},
        "by_difficulty": {},
        "failed_files": [],
    }

    all_chunks: list[NotebookChunk] = []
    all_notebooks: list[ParsedNotebook] = []
    all_loaded_notebooks: list[ParsedNotebook] = []

    # 모든 노트북 먼저 로드
    for input_dir in input_dirs:
        if not input_dir.exists():
            print(f"⚠️ 디렉토리가 존재하지 않습니다: {input_dir}")
            continue

        try:
            notebooks = loader.load_from_directory(input_dir, recursive=True, solution_only=False)
            all_loaded_notebooks.extend(notebooks)
        except Exception as e:
            print(f"❌ 디렉토리 로드 실패: {e}")

    print(f"\n📥 발견된 노트북: {len(all_loaded_notebooks)}개")

    # 증분 업데이트: 변경된 파일만 필터링
    if existing_version and incremental:
        current_files = {nb.file_path.name: nb.file_path for nb in all_loaded_notebooks}
        new_files, changed_files, deleted_files = existing_version.get_changed_files(current_files)

        print(f"\n📊 변경 감지 결과:")
        print(f"   🆕 새 파일: {len(new_files)}개")
        print(f"   ✏️  변경됨: {len(changed_files)}개")
        print(f"   🗑️  삭제됨: {len(deleted_files)}개")

        if dry_run:
            if new_files:
                print(f"\n   새 파일: {new_files[:5]}{'...' if len(new_files) > 5 else ''}")
            if changed_files:
                print(f"   변경 파일: {changed_files[:5]}{'...' if len(changed_files) > 5 else ''}")
            print("\n✅ DRY-RUN 완료. 실제 처리는 수행되지 않았습니다.")
            return {
                "mode": "dry_run",
                "new": len(new_files),
                "changed": len(changed_files),
                "deleted": len(deleted_files),
            }

        # 변경된 파일만 처리 대상으로 필터링
        files_to_process = set(new_files + changed_files)
        if not files_to_process:
            print("\n✅ 변경된 파일이 없습니다. 처리를 건너뜁니다.")
            return {"mode": "incremental", "changes": 0}

        notebooks_to_process = [
            nb for nb in all_loaded_notebooks if nb.file_path.name in files_to_process
        ]
        print(f"\n   → 처리 대상: {len(notebooks_to_process)}개 파일")
    else:
        notebooks_to_process = all_loaded_notebooks

    # 각 노트북 처리
    for notebook in notebooks_to_process:
        stats["total_notebooks"] += 1

        # 파일 타입 통계
        if notebook.file_type == FileType.SOLUTION:
            stats["solution_notebooks"] += 1
        elif notebook.file_type == FileType.PROBLEM:
            stats["problem_notebooks"] += 1

        # 과목별 통계
        course = notebook.course or "기타"
        if course not in stats["by_course"]:
            stats["by_course"][course] = {"notebooks": 0, "chunks": 0}
        stats["by_course"][course]["notebooks"] += 1

        # 난이도별 통계
        difficulty = notebook.difficulty.value
        if difficulty not in stats["by_difficulty"]:
            stats["by_difficulty"][difficulty] = 0
        stats["by_difficulty"][difficulty] += 1

        # 청킹
        try:
            chunks = chunker.chunk_notebook(notebook)

            if chunks:
                all_chunks.extend(chunks)
                all_notebooks.append(notebook)
                stats["total_chunks"] += len(chunks)
                stats["by_course"][course]["chunks"] += len(chunks)

                # 진행 상황 출력
                print(
                    f"  ✓ {notebook.file_path.name}: "
                    f"{len(chunks)}개 청크 ({notebook.file_type.value})"
                )
            else:
                print(f"  ○ {notebook.file_path.name}: 청크 없음")

        except Exception as e:
            print(f"  ✗ {notebook.file_path.name}: 오류 - {e}")
            stats["failed_files"].append({"file": str(notebook.file_path), "error": str(e)})

    # 청크 저장 (과목별로 분리)
    save_chunks_by_course(all_chunks, output_dir)

    # 전체 청크 저장
    all_chunks_file = output_dir / "all_notebook_chunks.json"
    save_all_chunks(all_chunks, all_chunks_file)

    # 통계 저장
    stats["processed_at"] = datetime.now().isoformat()
    stats_file = output_dir / "_summary.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 버전 파일 저장
    save_version_info(all_notebooks, all_chunks, output_dir)

    # 최종 통계 출력
    print_final_stats(stats)

    return stats


def save_version_info(
    notebooks: list[ParsedNotebook],
    chunks: list[NotebookChunk],
    output_dir: Path,
) -> None:
    """
    버전 정보를 _version.json에 저장합니다.

    Args:
        notebooks: 처리된 노트북 리스트
        chunks: 생성된 청크 리스트
        output_dir: 출력 디렉토리
    """
    version_info = VersionInfo(
        total_chunks=len(chunks),
    )

    # 각 노트북의 청크 수 계산
    chunks_by_file: dict[str, int] = {}
    for chunk in chunks:
        source_file = chunk.metadata.get("source_file", "")
        if source_file:
            file_name = Path(source_file).name
            chunks_by_file[file_name] = chunks_by_file.get(file_name, 0) + 1

    # 원본 파일 정보 추가
    for notebook in notebooks:
        file_name = notebook.file_path.name
        chunk_count = chunks_by_file.get(file_name, 0)
        if chunk_count > 0:
            version_info.add_source_file(
                file_name=file_name,
                file_path=notebook.file_path,
                chunk_count=chunk_count,
            )

    # 저장
    version_file = output_dir / "_version.json"
    save_version_file(version_file, version_info)
    print(f"\n📝 버전 정보 저장: {version_file.name}")


def save_chunks_by_course(chunks: list[NotebookChunk], output_dir: Path) -> None:
    """
    청크를 과목별로 분리하여 저장합니다.

    Args:
        chunks: 전체 청크 리스트
        output_dir: 출력 디렉토리
    """
    # 과목별 그룹화
    by_course: dict[str, list[NotebookChunk]] = {}

    for chunk in chunks:
        course = chunk.metadata.get("course", "기타") or "기타"
        if course not in by_course:
            by_course[course] = []
        by_course[course].append(chunk)

    # 각 과목별 파일 저장
    for course, course_chunks in by_course.items():
        # 파일명 정리
        safe_course = course.replace("/", "_").replace(" ", "_")
        output_file = output_dir / f"{safe_course}_chunks.json"

        data = {
            "course": course,
            "total_chunks": len(course_chunks),
            "chunks": [chunk.to_dict() for chunk in course_chunks],
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"  📄 저장: {output_file.name} ({len(course_chunks)}개 청크)")


def save_all_chunks(chunks: list[NotebookChunk], output_file: Path) -> None:
    """
    모든 청크를 하나의 파일로 저장합니다.

    Args:
        chunks: 전체 청크 리스트
        output_file: 출력 파일 경로
    """
    data = {
        "total_chunks": len(chunks),
        "generated_at": datetime.now().isoformat(),
        "chunks": [chunk.to_dict() for chunk in chunks],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n📦 전체 청크 저장: {output_file.name}")


def print_final_stats(stats: dict) -> None:
    """최종 통계를 출력합니다."""
    print(f"\n{'=' * 80}")
    print("📊 최종 통계")
    print(f"{'=' * 80}")

    print("\n📓 노트북 처리:")
    print(f"   전체: {stats['total_notebooks']}개")
    print(f"   정답 파일: {stats['solution_notebooks']}개")
    print(f"   문제 파일: {stats['problem_notebooks']}개")

    print("\n📝 청크 생성:")
    print(f"   전체: {stats['total_chunks']}개")

    print("\n📚 과목별:")
    for course, data in sorted(stats["by_course"].items()):
        print(f"   {course}: {data['notebooks']}개 노트북 → {data['chunks']}개 청크")

    print("\n⭐ 난이도별:")
    for difficulty, count in sorted(stats["by_difficulty"].items()):
        print(f"   {difficulty}: {count}개")

    if stats["failed_files"]:
        print(f"\n❌ 실패한 파일: {len(stats['failed_files'])}개")
        for fail in stats["failed_files"][:5]:
            print(f"   - {fail['file']}: {fail['error']}")

    print(f"\n{'=' * 80}")


def main() -> None:
    """메인 함수."""
    # 기본 경로 설정
    base_dir = Path(__file__).parent.parent.parent

    parser = argparse.ArgumentParser(
        description="Jupyter Notebook을 청킹하여 저장합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "document_chunks" / "notebook_chunks",
        help="청크 저장 디렉토리",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="청크 최대 토큰 수 (기본: 500)",
    )
    parser.add_argument(
        "--include-problems",
        action="store_true",
        help="문제 파일의 코드도 포함",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="전체 재처리 (기존 청크 무시)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="변경사항만 확인 (실제 처리 안함)",
    )

    args = parser.parse_args()

    input_dirs = [
        base_dir / "original_documents" / "practice",
        base_dir / "original_documents" / "home_work",
    ]

    # 처리 실행
    process_notebooks(
        input_dirs=input_dirs,
        output_dir=args.output_dir,
        solution_only=not args.include_problems,
        max_tokens=args.max_tokens,
        incremental=not args.recreate,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
