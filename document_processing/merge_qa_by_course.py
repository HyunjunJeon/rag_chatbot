"""
과정별로 모든 기수의 Q&A를 하나의 JSON 파일로 병합하는 스크립트.

각 과정(level2_cv, level2_nlp 등)별로 모든 기수의 Q&A를 수집하고,
메타데이터와 함께 하나의 JSON 파일로 저장합니다.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def extract_date_from_filename(filename: str) -> str | None:
    """
    파일명에서 날짜를 추출합니다.

    매개변수:
        filename: 파일명 (예: "2021-09-06_qa.json")

    반환값:
        날짜 문자열 (예: "2021-09-06") 또는 None
    """
    try:
        # _qa.json 제거하고 날짜 부분만 추출
        date_str = filename.replace("_qa.json", "")
        # 날짜 형식 검증
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        return None


def load_qa_file(file_path: Path) -> dict[str, Any]:
    """
    Q&A JSON 파일을 로드합니다.

    매개변수:
        file_path: JSON 파일 경로

    반환값:
        파싱된 JSON 데이터
    """
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def collect_qa_by_course(
    processed_dir: Path | str,
) -> dict[str, dict[str, Any]]:
    """
    과정별로 모든 Q&A를 수집합니다.

    매개변수:
        processed_dir: 처리된 Q&A 디렉토리 경로

    반환값:
        과정명을 키로, Q&A 데이터와 메타데이터를 값으로 하는 딕셔너리
    """
    processed_dir = Path(processed_dir)
    course_data: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "qa_pairs": [],
            "generations": set(),
            "dates": [],
            "gen_stats": defaultdict(int),
            "source_files": [],
        }
    )

    print("=" * 80)
    print("📚 과정별 Q&A 수집 시작")
    print("=" * 80)

    # 기수별 디렉토리 순회
    for gen_dir in sorted(processed_dir.iterdir()):
        if not gen_dir.is_dir():
            continue

        generation = gen_dir.name
        print(f"\n📁 기수 {generation} 처리 중...")

        # 과정별 디렉토리 순회
        for course_dir in sorted(gen_dir.iterdir()):
            if not course_dir.is_dir():
                continue

            course_name = course_dir.name
            print(f"  📂 {course_name}")

            # JSON 파일 처리
            qa_files = sorted(course_dir.glob("*_qa.json"))
            file_count = 0

            for qa_file in qa_files:
                try:
                    data = load_qa_file(qa_file)
                    qa_pairs = data.get("qa_pairs", [])

                    if not qa_pairs:
                        continue

                    # 날짜 추출
                    date = extract_date_from_filename(qa_file.name)

                    # 각 Q&A 쌍에 메타데이터 추가
                    for qa_pair in qa_pairs:
                        enriched_qa = {
                            "generation": generation,
                            "date": date,
                            "source_file": qa_file.name,
                            "course": course_name,
                            "question": qa_pair["question"],
                            "answers": qa_pair["answers"],
                        }
                        course_data[course_name]["qa_pairs"].append(enriched_qa)

                    # 통계 업데이트
                    course_data[course_name]["generations"].add(generation)
                    if date:
                        course_data[course_name]["dates"].append(date)
                    course_data[course_name]["gen_stats"][generation] += len(qa_pairs)
                    course_data[course_name]["source_files"].append(
                        {
                            "generation": generation,
                            "filename": qa_file.name,
                            "qa_count": len(qa_pairs),
                        }
                    )

                    file_count += 1

                except Exception as e:
                    print(f"    ✗ {qa_file.name}: {e}")
                    continue

            if file_count > 0:
                print(f"    ✓ {file_count}개 파일 처리 완료")

    return dict(course_data)


def create_metadata(course_data: dict[str, Any]) -> dict[str, Any]:
    """
    과정별 메타데이터를 생성합니다.

    매개변수:
        course_data: 수집된 과정 데이터

    반환값:
        메타데이터 딕셔너리
    """
    qa_pairs = course_data["qa_pairs"]
    dates = sorted(course_data["dates"]) if course_data["dates"] else []

    metadata = {
        "total_qa_pairs": len(qa_pairs),
        "generations": sorted(course_data["generations"]),
        "generation_count": len(course_data["generations"]),
        "statistics": {
            "by_generation": dict(
                sorted(course_data["gen_stats"].items(), key=lambda x: x[0])
            )
        },
    }

    # 날짜 범위 추가
    if dates:
        metadata["date_range"] = {"start": dates[0], "end": dates[-1]}

    # 파일 통계 추가
    metadata["source_files"] = {
        "count": len(course_data["source_files"]),
        "files": course_data["source_files"],
    }

    return metadata


def sort_qa_pairs(qa_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Q&A 쌍을 날짜와 기수 순으로 정렬합니다.

    매개변수:
        qa_pairs: Q&A 쌍 리스트

    반환값:
        정렬된 Q&A 쌍 리스트
    """

    def sort_key(qa: dict[str, Any]) -> tuple:
        date = qa.get("date") or "9999-99-99"
        generation = qa.get("generation", "0")
        return (date, generation)

    return sorted(qa_pairs, key=sort_key)


def merge_qa_by_course(
    processed_dir: Path | str, output_dir: Path | str
) -> dict[str, int]:
    """
    과정별로 Q&A를 병합하여 저장합니다.

    매개변수:
        processed_dir: 처리된 Q&A 디렉토리 경로
        output_dir: 출력 디렉토리 경로

    반환값:
        과정명과 Q&A 개수를 담은 딕셔너리
    """
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 과정별 데이터 수집
    course_data = collect_qa_by_course(processed_dir)

    print("\n" + "=" * 80)
    print("💾 과정별 JSON 파일 생성 중...")
    print("=" * 80)

    stats: dict[str, int] = {}

    for course_name, data in sorted(course_data.items()):
        # Q&A 정렬
        sorted_qa = sort_qa_pairs(data["qa_pairs"])

        # 메타데이터 생성
        metadata = create_metadata(data)

        # 최종 데이터 구조
        output_data = {
            "course": course_name,
            "metadata": metadata,
            "qa_pairs": sorted_qa,
        }

        # JSON 파일로 저장
        output_file = output_dir / f"{course_name}_merged.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        qa_count = len(sorted_qa)
        stats[course_name] = qa_count

        print(f"\n✓ {course_name}_merged.json")
        print(f"  - Q&A 쌍: {qa_count}개")
        print(f"  - 기수: {metadata['generation_count']}개 ({', '.join(metadata['generations'])})")
        if "date_range" in metadata:
            print(
                f"  - 기간: {metadata['date_range']['start']} ~ {metadata['date_range']['end']}"
            )
        print(f"  - 파일 크기: {output_file.stat().st_size / 1024:.1f} KB")

    return stats


def generate_summary(stats: dict[str, int], output_dir: Path) -> None:
    """
    전체 통계 요약을 생성합니다.

    매개변수:
        stats: 과정별 Q&A 개수
        output_dir: 출력 디렉토리
    """
    total_qa = sum(stats.values())
    total_courses = len(stats)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_courses": total_courses,
        "total_qa_pairs": total_qa,
        "by_course": stats,
    }

    summary_file = output_dir / "_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("📊 전체 통계")
    print("=" * 80)
    print(f"총 과정 수: {total_courses}개")
    print(f"총 Q&A 쌍: {total_qa}개")
    print(f"\n요약 파일: {summary_file}")


def main() -> None:
    """메인 함수."""
    # 경로 설정
    processed_dir = Path(
        "/Users/jhj/Desktop/personal/naver_connect_chatbot/document_chunks/slack_qa_processed"
    )
    output_dir = Path(
        "/Users/jhj/Desktop/personal/naver_connect_chatbot/document_chunks/slack_qa_merged"
    )

    print("\n🚀 과정별 Q&A 병합 시작\n")
    print(f"입력: {processed_dir}")
    print(f"출력: {output_dir}\n")

    # 병합 실행
    stats = merge_qa_by_course(processed_dir, output_dir)

    # 요약 생성
    generate_summary(stats, output_dir)

    print("\n" + "=" * 80)
    print("✅ 모든 작업 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()

