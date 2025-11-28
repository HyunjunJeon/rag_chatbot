"""
모든 Slack JSON 파일을 일괄 처리하는 메인 스크립트.

전체 qa_dataset_from_slack 디렉토리의 모든 JSON 파일을 처리하여
Q&A 쌍을 추출하고 저장합니다.
"""

import sys
from pathlib import Path

from slack_qa_loader import SlackQALoader


def process_all_slack_data(
    input_base_dir: Path | str,
    output_base_dir: Path | str,
    exclude_bot_messages: bool = False,
) -> None:
    """
    모든 Slack JSON 파일을 일괄 처리합니다.

    매개변수:
        input_base_dir: 입력 기본 디렉토리 (qa_dataset_from_slack)
        output_base_dir: 출력 기본 디렉토리
        exclude_bot_messages: 봇 메시지 제외 여부
    """
    input_base_dir = Path(input_base_dir)
    output_base_dir = Path(output_base_dir)

    print("=" * 80)
    print("🚀 Slack Q&A 데이터 일괄 처리 시작")
    print("=" * 80)
    print(f"\n입력 디렉토리: {input_base_dir}")
    print(f"출력 디렉토리: {output_base_dir}")
    print(f"봇 메시지 제외: {exclude_bot_messages}\n")

    if not input_base_dir.exists():
        print(f"❌ 입력 디렉토리가 존재하지 않습니다: {input_base_dir}")
        return

    loader = SlackQALoader(exclude_bot_messages=exclude_bot_messages)

    # 기수별 디렉토리 탐색
    generation_dirs = sorted([d for d in input_base_dir.iterdir() if d.is_dir()])

    total_files = 0
    total_qa_pairs = 0
    failed_files = 0

    for gen_dir in generation_dirs:
        print(f"\n{'=' * 80}")
        print(f"📁 기수: {gen_dir.name}")
        print(f"{'=' * 80}")

        # 주제별 디렉토리 탐색
        topic_dirs = sorted([d for d in gen_dir.iterdir() if d.is_dir()])

        for topic_dir in topic_dirs:
            print(f"\n  📂 주제: {topic_dir.name}")
            print(f"  {'-' * 76}")

            # 출력 디렉토리 생성
            output_dir = output_base_dir / gen_dir.name / topic_dir.name
            output_dir.mkdir(parents=True, exist_ok=True)

            # JSON 파일 처리
            json_files = list(topic_dir.glob("*.json"))
            topic_qa_count = 0

            for json_file in sorted(json_files):
                try:
                    qa_pairs = loader.load_from_file(json_file)

                    if qa_pairs:
                        # 출력 파일 생성
                        output_file = output_dir / f"{json_file.stem}_qa.json"
                        loader.export_to_json(qa_pairs, output_file)

                        topic_qa_count += len(qa_pairs)
                        total_qa_pairs += len(qa_pairs)
                        print(f"    ✓ {json_file.name}: {len(qa_pairs)} Q&A 쌍")
                    else:
                        print(f"    ○ {json_file.name}: Q&A 없음")

                    total_files += 1

                except Exception as e:
                    print(f"    ✗ {json_file.name}: 오류 - {e}")
                    failed_files += 1
                    continue

            print(f"\n  소계: {topic_qa_count} Q&A 쌍")

    # 최종 통계
    print(f"\n{'=' * 80}")
    print("📊 최종 통계")
    print(f"{'=' * 80}")
    print(f"처리된 파일: {total_files}개")
    print(f"실패한 파일: {failed_files}개")
    print(f"추출된 Q&A 쌍: {total_qa_pairs}개")
    print(f"성공률: {(total_files - failed_files) / total_files * 100:.1f}%")
    print(f"\n출력 디렉토리: {output_base_dir}")
    print("=" * 80)


def main() -> None:
    """메인 함수."""
    # PROJECT_ROOT 기준 상대 경로 사용
    project_root = Path(__file__).parent.parent

    # 기본 경로 설정
    input_dir = project_root / "original_documents" / "qa_dataset_from_slack"
    output_dir = project_root / "document_chunks" / "slack_qa_processed"

    # 명령행 인자 처리
    exclude_bot = False
    if len(sys.argv) > 1 and sys.argv[1] == "--exclude-bot":
        exclude_bot = True
        print("\n⚠️  봇 메시지 제외 모드로 실행합니다.\n")

    # 처리 실행
    process_all_slack_data(input_dir, output_dir, exclude_bot_messages=exclude_bot)


if __name__ == "__main__":
    main()

