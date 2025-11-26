"""
전체 주간 미션 일괄 처리 스크립트.

weekly_mission/ 폴더의 문제 파일과 채점 기준표를 처리하여
document_chunks/mission_chunks/ 에 저장합니다.

⚠️ 정답 파일은 자동으로 제외됩니다.

사용법:
    # 증분 업데이트 (기본)
    python process_all_missions.py

    # 전체 재처리
    python process_all_missions.py --recreate

    # 변경사항만 확인 (dry-run)
    python process_all_missions.py --dry-run
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "document_processing"))

from mission.mission_loader import MissionLoader, MissionType
from mission.mission_chunker import MissionChunker
from common.versioning import VersionInfo, save_version_file, load_version_file


def process_all_missions(
    input_dir: Path,
    output_dir: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    verbose: bool = True,
    incremental: bool = True,
    dry_run: bool = False,
) -> dict:
    """
    모든 미션을 처리하고 청크를 저장합니다.

    Args:
        input_dir: 미션 파일이 있는 디렉토리
        output_dir: 청크 저장 디렉토리
        chunk_size: 청크 최대 토큰 수
        chunk_overlap: 청크 간 오버랩 토큰 수
        verbose: 상세 로그 출력
        incremental: True면 변경된 파일만 처리
        dry_run: True면 변경사항만 확인 (실제 처리 안함)

    Returns:
        처리 통계 딕셔너리
    """
    print("=" * 60)
    print("📋 주간 미션 일괄 처리 시작")
    print("=" * 60)
    print(f"📂 입력: {input_dir}")
    print(f"📁 출력: {output_dir}")
    print(f"🔧 청크 크기: {chunk_size} tokens (overlap: {chunk_overlap})")
    print("⚠️  정답 파일은 자동 제외됩니다")
    mode_str = "🔄 증분 업데이트" if incremental else "🔁 전체 재처리"
    if dry_run:
        mode_str += " (DRY-RUN)"
    print(f"📌 모드: {mode_str}")
    print()

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # 기존 버전 정보 로드 (증분 업데이트용)
    version_file = output_dir / "_version.json"
    existing_version = load_version_file(version_file) if incremental else None

    # 로더 및 청커 초기화
    loader = MissionLoader(verbose=verbose)
    chunker = MissionChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        include_code_hints=True,
    )

    # 미션 파일 로드 (정답 제외)
    print("📥 미션 파일 로드 중...")
    all_missions = loader.load_from_directory(input_dir, include_solutions=False)

    # 증분 업데이트: 변경된 파일만 필터링
    if existing_version and incremental:
        current_files = {m.file_path.name: m.file_path for m in all_missions}
        new_files, changed_files, deleted_files = existing_version.get_changed_files(current_files)

        print(f"\n📊 변경 감지 결과:")
        print(f"   🆕 새 파일: {len(new_files)}개")
        print(f"   ✏️  변경됨: {len(changed_files)}개")
        print(f"   🗑️  삭제됨: {len(deleted_files)}개")

        if dry_run:
            if new_files:
                print(f"\n   새 파일 목록: {new_files[:5]}{'...' if len(new_files) > 5 else ''}")
            if changed_files:
                print(
                    f"   변경 파일 목록: {changed_files[:5]}{'...' if len(changed_files) > 5 else ''}"
                )
            if deleted_files:
                print(
                    f"   삭제 파일 목록: {deleted_files[:5]}{'...' if len(deleted_files) > 5 else ''}"
                )
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

        missions = [m for m in all_missions if m.file_path.name in files_to_process]
        print(f"\n   → 처리 대상: {len(missions)}개 파일")
    else:
        missions = all_missions

    problems = [m for m in missions if m.mission_type == MissionType.PROBLEM]
    rubrics = [m for m in missions if m.mission_type == MissionType.RUBRIC]

    print(f"   📝 문제: {len(problems)}개")
    print(f"   📋 채점기준: {len(rubrics)}개")
    print()

    if not missions:
        print("⚠️ 처리할 미션이 없습니다.")
        return {}

    # 과목별로 그룹화
    missions_by_course: dict[str, list] = defaultdict(list)
    for mission in missions:
        course = mission.course if mission.course else "Unknown"
        missions_by_course[course].append(mission)

    # 통계
    stats = {
        "processed_at": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "total_problems": len(problems),
        "total_rubrics": len(rubrics),
        "total_chunks": 0,
        "by_course": {},
    }

    all_chunks: list[dict] = []

    # 과목별 처리
    for course, course_missions in sorted(missions_by_course.items()):
        print(f"📚 {course} ({len(course_missions)}개 파일)")

        course_chunks: list[dict] = []

        for mission in course_missions:
            # 청킹
            chunks = chunker.chunk_mission(mission)
            chunk_dicts = [c.to_dict() for c in chunks]
            course_chunks.extend(chunk_dicts)

            if verbose:
                type_icon = "📝" if mission.is_problem else "📋"
                print(f"   {type_icon} {mission.mission_name}: {len(chunks)} chunks")

        # 과목별 저장
        course_slug = course.lower().replace(" ", "_").replace("-", "_")
        course_file = output_dir / f"{course_slug}_chunks.json"

        course_data = {
            "course": course,
            "mission_count": len(course_missions),
            "chunk_count": len(course_chunks),
            "chunks": course_chunks,
        }

        with open(course_file, "w", encoding="utf-8") as f:
            json.dump(course_data, f, ensure_ascii=False, indent=2)

        print(f"   ✅ 저장: {course_file.name} ({len(course_chunks)} chunks)")
        print()

        # 통계 업데이트
        course_problems = [m for m in course_missions if m.is_problem]
        course_rubrics = [m for m in course_missions if m.is_rubric]

        stats["by_course"][course] = {
            "problem_count": len(course_problems),
            "rubric_count": len(course_rubrics),
            "chunk_count": len(course_chunks),
        }
        stats["total_chunks"] += len(course_chunks)
        all_chunks.extend(course_chunks)

    # 전체 청크 저장
    all_chunks_file = output_dir / "all_mission_chunks.json"
    with open(all_chunks_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_chunks": len(all_chunks),
                "chunks": all_chunks,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"📦 전체 저장: {all_chunks_file.name} ({len(all_chunks)} chunks)")

    # 요약 저장
    summary_file = output_dir / "_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 버전 정보 저장
    save_version_info_mission(missions, all_chunks, output_dir)

    # 결과 출력
    print()
    print("=" * 60)
    print("✅ 처리 완료")
    print("=" * 60)
    print(f"📝 문제 파일: {stats['total_problems']}개")
    print(f"📋 채점기준: {stats['total_rubrics']}개")
    print(f"🧩 총 청크: {stats['total_chunks']}개")
    print()
    print("📊 과목별 통계:")
    for course, course_stats in stats["by_course"].items():
        print(
            f"   • {course}: "
            f"{course_stats['problem_count']}문제 + {course_stats['rubric_count']}채점기준 "
            f"→ {course_stats['chunk_count']} chunks"
        )
    print()

    return stats


def save_version_info_mission(missions: list, all_chunks: list[dict], output_dir: Path) -> None:
    """
    버전 정보를 _version.json에 저장합니다.

    Args:
        missions: 처리된 미션 리스트
        all_chunks: 생성된 청크 리스트 (dict)
        output_dir: 출력 디렉토리
    """
    version_info = VersionInfo(
        total_chunks=len(all_chunks),
    )

    # 각 미션의 청크 수 계산
    chunks_by_file: dict[str, int] = {}
    for chunk in all_chunks:
        source_file = chunk.get("metadata", {}).get("source_file", "")
        if source_file:
            file_name = Path(source_file).name
            chunks_by_file[file_name] = chunks_by_file.get(file_name, 0) + 1

    # 원본 파일 정보 추가
    for mission in missions:
        file_name = mission.file_path.name
        chunk_count = chunks_by_file.get(file_name, 0)
        if chunk_count > 0:
            version_info.add_source_file(
                file_name=file_name,
                file_path=mission.file_path,
                chunk_count=chunk_count,
            )

    # 저장
    version_file = output_dir / "_version.json"
    save_version_file(version_file, version_info)
    print(f"\n📝 버전 정보 저장: {version_file.name}")


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(
        description="주간 미션을 청킹하여 저장합니다. (정답 파일 제외)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "original_documents" / "weekly_mission",
        help="미션 파일이 있는 디렉토리",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "document_chunks" / "mission_chunks",
        help="청크 저장 디렉토리",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="청크 최대 토큰 수 (기본: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="청크 간 오버랩 토큰 수 (기본: 100)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="간략한 출력",
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

    # 입력 디렉토리 확인
    if not args.input_dir.exists():
        print(f"❌ 입력 디렉토리가 없습니다: {args.input_dir}")
        sys.exit(1)

    # 처리 실행
    process_all_missions(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        incremental=not args.recreate,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
