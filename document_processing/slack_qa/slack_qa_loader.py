"""
Slack 메시지 JSON 파일을 로드하고 Q&A 쌍을 추출하는 모듈.

이 모듈은 Slack에서 추출한 JSON 파일을 읽어서 시스템 메시지를 제외하고,
학생들의 질문과 답변 데이터만 추출합니다.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["SlackQALoader", "QAPair", "Message"]


@dataclass
class Message:
    """
    Slack 메시지를 표현하는 데이터 클래스.

    속성:
        text: 메시지 텍스트
        user: 사용자 ID
        timestamp: 메시지 타임스탬프
        is_bot: 봇 메시지 여부
        user_name: 사용자 실명 (있는 경우)
        metadata: 추가 메타데이터
    """

    text: str
    user: str
    timestamp: str
    is_bot: bool
    user_name: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class QAPair:
    """
    질문-답변 쌍을 표현하는 데이터 클래스.

    속성:
        question: 질문 메시지
        answers: 답변 메시지 리스트
    """

    question: Message
    answers: list[Message]


class SlackQALoader:
    """
    Slack JSON 파일을 로드하고 Q&A 쌍을 추출하는 클래스.

    이 클래스는 Slack 메시지 JSON 파일을 읽어서:
    1. 시스템 메시지 (채널 참여, 초대 등) 제거
    2. 스레드 구조 분석
    3. 질문-답변 쌍 추출

    예시:
        ```python
        loader = SlackQALoader()
        qa_pairs = loader.load_from_file("2021-10-08.json")

        for qa in qa_pairs:
            print(f"Q: {qa.question.text}")
            for answer in qa.answers:
                print(f"A: {answer.text}")
        ```
    """

    def __init__(self, exclude_bot_messages: bool = False) -> None:
        """
        SlackQALoader 초기화.

        매개변수:
            exclude_bot_messages: True면 봇 메시지를 제외, False면 포함 (기본값: False)
                                 최신 기수의 경우 답변 봇이 유용한 답변을 제공하므로 기본값은 False
        """
        self.exclude_bot_messages = exclude_bot_messages

    def load_from_file(self, file_path: Path | str) -> list[QAPair]:
        """
        JSON 파일에서 Q&A 쌍을 로드합니다.

        매개변수:
            file_path: JSON 파일 경로

        반환값:
            QAPair 객체의 리스트

        예외:
            FileNotFoundError: 파일이 존재하지 않는 경우
            json.JSONDecodeError: JSON 파싱 오류가 발생한 경우
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            messages = json.load(f)

        return self.extract_qa_pairs(messages)

    def load_from_directory(
        self, directory_path: Path | str, pattern: str = "*.json"
    ) -> dict[str, list[QAPair]]:
        """
        디렉토리 내의 모든 JSON 파일에서 Q&A 쌍을 로드합니다.

        매개변수:
            directory_path: 디렉토리 경로
            pattern: 파일 패턴 (기본값: "*.json")

        반환값:
            파일명을 키로, QAPair 리스트를 값으로 하는 딕셔너리
        """
        directory_path = Path(directory_path)
        results: dict[str, list[QAPair]] = {}

        for json_file in directory_path.glob(pattern):
            try:
                qa_pairs = self.load_from_file(json_file)
                if qa_pairs:  # 빈 리스트가 아닌 경우만 추가
                    results[json_file.name] = qa_pairs
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"파일 로드 실패 ({json_file.name}): {e}")
                continue

        return results

    def extract_qa_pairs(self, messages: list[dict[str, Any]]) -> list[QAPair]:
        """
        메시지 리스트에서 Q&A 쌍을 추출합니다.

        매개변수:
            messages: Slack 메시지 딕셔너리의 리스트

        반환값:
            QAPair 객체의 리스트
        """
        # 1. 유효한 메시지만 필터링
        valid_messages = [msg for msg in messages if self._is_valid_message(msg)]

        # 2. 스레드 구조 매핑
        thread_map = self._build_thread_map(valid_messages)

        # 3. Q&A 쌍 생성
        qa_pairs: list[QAPair] = []

        for thread_ts, thread_messages in thread_map.items():
            if not thread_messages:
                continue

            # 첫 번째 메시지가 질문
            question_msg = thread_messages[0]
            question = self._create_message(question_msg)

            # 나머지 메시지들이 답변
            answers = [self._create_message(msg) for msg in thread_messages[1:]]

            # 답변이 있는 경우만 추가 (혼잣말 제외)
            if answers:
                qa_pairs.append(QAPair(question=question, answers=answers))

        return qa_pairs

    def _is_valid_message(self, message: dict[str, Any]) -> bool:
        """
        메시지가 유효한지 확인합니다.

        유효한 메시지 조건:
        - type이 "message"
        - 시스템 메시지가 아님 (subtype이 없거나 특정 subtype이 아님)
        - user 필드가 존재
        - text 필드가 존재하고 비어있지 않음
        - 봇 메시지 제외 옵션이 활성화된 경우 bot_id가 없음

        매개변수:
            message: Slack 메시지 딕셔너리

        반환값:
            유효한 메시지면 True, 아니면 False
        """
        # 기본 필드 확인
        if message.get("type") != "message":
            return False

        # 텍스트가 없거나 비어있는 경우 제외
        text = message.get("text", "").strip()
        if not text:
            return False

        # 시스템 메시지 제외
        if self._is_system_message(message):
            return False

        # user 필드가 없으면 제외 (봇 메시지 포함 옵션이 꺼져있을 때만)
        if not message.get("user"):
            return False

        # 봇 메시지 제외 옵션 확인
        if self.exclude_bot_messages and message.get("bot_id"):
            return False

        return True

    def _is_system_message(self, message: dict[str, Any]) -> bool:
        """
        시스템 메시지인지 확인합니다.

        시스템 메시지 판별 기준:
        - subtype이 channel_join, channel_leave 등
        - inviter 필드 존재 (초대 메시지)
        - 채널 공지 (<!channel>, <!here> 등)

        매개변수:
            message: Slack 메시지 딕셔너리

        반환값:
            시스템 메시지면 True, 아니면 False
        """
        # subtype으로 시스템 메시지 판별
        system_subtypes = {
            "channel_join",
            "channel_leave",
            "channel_archive",
            "channel_unarchive",
            "channel_name",
            "channel_purpose",
            "channel_topic",
            "pinned_item",
            "unpinned_item",
        }

        subtype = message.get("subtype")
        if subtype in system_subtypes:
            return True

        # 초대 메시지 제외
        if message.get("inviter"):
            return True

        # 채널 공지 메시지 제외 (선택적)
        text = message.get("text", "")
        if text.startswith("<!channel>") or text.startswith("<!here>"):
            # 공지가 질문일 수도 있으므로 일단 허용
            # 필요시 True로 변경
            pass

        return False

    def _build_thread_map(
        self, messages: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """
        스레드 구조를 매핑합니다.

        thread_ts를 키로 하여 각 스레드에 속한 메시지들을 그룹화합니다.
        thread_ts가 없는 메시지는 독립 메시지로 간주하고 자신의 ts를 thread_ts로 사용합니다.

        매개변수:
            messages: 유효한 메시지 리스트

        반환값:
            thread_ts를 키로 하고, 해당 스레드의 메시지 리스트를 값으로 하는 딕셔너리
        """
        thread_map: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for message in messages:
            # thread_ts가 있으면 사용, 없으면 자신의 ts 사용
            thread_ts = message.get("thread_ts", message.get("ts"))

            if thread_ts:
                thread_map[thread_ts].append(message)

        # 각 스레드를 타임스탬프 순으로 정렬
        for thread_ts in thread_map:
            thread_map[thread_ts].sort(key=lambda msg: float(msg.get("ts", 0)))

        return dict(thread_map)

    def _create_message(self, message: dict[str, Any]) -> Message:
        """
        Slack 메시지 딕셔너리를 Message 객체로 변환합니다.

        매개변수:
            message: Slack 메시지 딕셔너리

        반환값:
            Message 객체
        """
        text = message.get("text", "").strip()
        user = message.get("user", "unknown")
        timestamp = message.get("ts", "")
        is_bot = bool(message.get("bot_id"))

        # 사용자 이름 추출
        user_name = None
        user_profile = message.get("user_profile", {})
        if user_profile:
            user_name = user_profile.get("real_name") or user_profile.get("display_name")

        # 메타데이터 수집
        metadata = {
            "edited": message.get("edited"),
            "reactions": message.get("reactions"),
            "reply_count": message.get("reply_count", 0),
        }

        # bot 메시지인 경우 봇 정보 추가
        if is_bot:
            bot_profile = message.get("bot_profile", {})
            metadata["bot_name"] = bot_profile.get("name")

        return Message(
            text=text,
            user=user,
            timestamp=timestamp,
            is_bot=is_bot,
            user_name=user_name,
            metadata=metadata,
        )

    def export_to_json(
        self, qa_pairs: list[QAPair], output_path: Path | str, indent: int = 2
    ) -> None:
        """
        Q&A 쌍을 JSON 파일로 저장합니다.

        매개변수:
            qa_pairs: QAPair 객체 리스트
            output_path: 출력 파일 경로
            indent: JSON 들여쓰기 (기본값: 2)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # QAPair를 딕셔너리로 변환
        data = {
            "qa_pairs": [
                {
                    "question": {
                        "text": qa.question.text,
                        "user": qa.question.user,
                        "user_name": qa.question.user_name,
                        "timestamp": qa.question.timestamp,
                        "is_bot": qa.question.is_bot,
                        "metadata": qa.question.metadata,
                    },
                    "answers": [
                        {
                            "text": answer.text,
                            "user": answer.user,
                            "user_name": answer.user_name,
                            "timestamp": answer.timestamp,
                            "is_bot": answer.is_bot,
                            "metadata": answer.metadata,
                        }
                        for answer in qa.answers
                    ],
                }
                for qa in qa_pairs
            ]
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)


def batch_process_directory(
    input_dir: Path | str,
    output_dir: Path | str,
    exclude_bot_messages: bool = False,
    recursive: bool = True,
) -> dict[str, int]:
    """
    디렉토리 내의 모든 JSON 파일을 일괄 처리합니다.

    매개변수:
        input_dir: 입력 디렉토리 경로
        output_dir: 출력 디렉토리 경로
        exclude_bot_messages: 봇 메시지 제외 여부
        recursive: 하위 디렉토리도 처리할지 여부

    반환값:
        처리 결과 통계 (파일명: Q&A 쌍 개수)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = SlackQALoader(exclude_bot_messages=exclude_bot_messages)
    stats: dict[str, int] = {}

    # 파일 패턴 결정
    pattern = "**/*.json" if recursive else "*.json"

    for json_file in input_dir.glob(pattern):
        if json_file.is_file():
            try:
                qa_pairs = loader.load_from_file(json_file)

                # Q&A 쌍이 있는 경우만 저장
                if qa_pairs:
                    # 출력 경로 생성 (입력 디렉토리 구조 유지)
                    relative_path = json_file.relative_to(input_dir)
                    output_file = output_dir / relative_path.parent / f"{relative_path.stem}_qa.json"

                    loader.export_to_json(qa_pairs, output_file)
                    stats[str(relative_path)] = len(qa_pairs)
                    print(f"✓ 처리 완료: {relative_path} ({len(qa_pairs)} Q&A 쌍)")
                else:
                    print(f"○ Q&A 없음: {json_file.relative_to(input_dir)}")

            except Exception as e:
                print(f"✗ 처리 실패: {json_file.relative_to(input_dir)} - {e}")
                continue

    return stats

