"""
Slack Q&A 데이터를 Qdrant VectorDB에 저장하는 스크립트.

질문-답변 페어 단위로 문서를 생성하고, 풍부한 메타데이터와 함께 저장합니다.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from openrouter_embeddings import OpenRouterEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

__all__ = ["QAVectorDBIngestion"]


class QAVectorDBIngestion:
    """
    Slack Q&A 데이터를 Qdrant VectorDB에 저장하는 클래스.

    예시:
        ```python
        ingestion = QAVectorDBIngestion(
            qdrant_url="http://localhost:6333",
            collection_name="slack_qa",
            embedding_model="jhgan/ko-sroberta-multitask"
        )

        ingestion.create_collection()
        ingestion.ingest_from_directory("document_chunks/slack_qa_merged/")
        ```
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "slack_qa",
        embedding_model: str = "qwen/qwen3-embedding-4b",
        batch_size: int = 100,
        embedding_batch_size: int = 16,
    ) -> None:
        """
        초기화.

        매개변수:
            qdrant_url: Qdrant 서버 URL
            collection_name: 컬렉션 이름
            embedding_model: 임베딩 모델 이름
            batch_size: VectorDB 배치 처리 크기
            embedding_batch_size: 임베딩 API 배치 크기
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size

        # Qdrant 클라이언트 초기화
        print(f"🔌 Qdrant 연결 중: {qdrant_url}")
        self.client = QdrantClient(url=qdrant_url)

        # 임베딩 모델 로드
        print(f"🤖 임베딩 모델 로드 중: {embedding_model}")
        self.embedding_model = OpenRouterEmbeddings(model=embedding_model)
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
        print(f"   벡터 차원: {self.vector_size}")

    def create_collection(self, recreate: bool = False) -> None:
        """
        Qdrant 컬렉션을 생성합니다.

        매개변수:
            recreate: True면 기존 컬렉션 삭제 후 재생성
        """
        print(f"\n📦 컬렉션 생성: {self.collection_name}")

        # 기존 컬렉션 확인
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)

        if collection_exists:
            if recreate:
                print(f"   기존 컬렉션 삭제 중...")
                self.client.delete_collection(self.collection_name)
            else:
                print(f"   ⚠️  컬렉션이 이미 존재합니다. recreate=True로 재생성할 수 있습니다.")
                return

        # 컬렉션 생성
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )
        print(f"   ✅ 컬렉션 생성 완료")

    def _generate_doc_id(self, qa_data: dict[str, Any]) -> str:
        """
        문서 고유 ID를 생성합니다.

        매개변수:
            qa_data: Q&A 데이터

        반환값:
            고유 ID 문자열
        """
        # 생성, 과정, 날짜, 타임스탬프를 조합하여 고유 ID 생성
        unique_str = (
            f"{qa_data['generation']}_"
            f"{qa_data['course']}_"
            f"{qa_data['date']}_"
            f"{qa_data['question']['timestamp']}_"
            f"{qa_data.get('answer_index', 0)}"
        )
        return hashlib.md5(unique_str.encode()).hexdigest()

    def _parse_course_name(self, course: str) -> tuple[str, str]:
        """
        과정명을 파싱하여 레벨과 주제를 추출합니다.

        매개변수:
            course: 과정명 (예: "level2_cv")

        반환값:
            (레벨, 주제) 튜플
        """
        parts = course.split("_", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return course, "unknown"

    def _create_embedding_text(self, qa_data: dict[str, Any]) -> str:
        """
        임베딩할 텍스트를 생성합니다.

        매개변수:
            qa_data: Q&A 데이터

        반환값:
            임베딩용 텍스트
        """
        course = qa_data["course"]
        generation = qa_data["generation"]
        question_text = qa_data["question"]["text"]
        answer_text = qa_data["answer"]["text"]
        answer_user = qa_data["answer"].get("user_name") or qa_data["answer"]["user"]

        # 구조화된 텍스트 생성
        text = f"""과정: {course}
기수: {generation}기

질문: {question_text}

답변: {answer_text}

작성자: {answer_user}"""

        return text

    def _create_payload(self, qa_data: dict[str, Any]) -> dict[str, Any]:
        """
        Qdrant payload를 생성합니다.

        매개변수:
            qa_data: Q&A 데이터

        반환값:
            payload 딕셔너리
        """
        course = qa_data["course"]
        course_level, course_topic = self._parse_course_name(course)
        date_str = qa_data["date"]

        # 날짜 파싱
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            year = date_obj.year
            year_month = date_obj.strftime("%Y-%m")
            timestamp = int(date_obj.timestamp())
        except (ValueError, TypeError):
            year = 0
            year_month = ""
            timestamp = 0

        # 반응 정보
        reactions = qa_data["answer"].get("metadata", {}).get("reactions") or []
        reaction_count = sum(r.get("count", 0) for r in reactions) if reactions else 0
        has_reactions = reaction_count > 0

        # Payload 생성
        payload = {
            # 필터링 핵심 필드
            "course": course,
            "course_level": course_level,
            "course_topic": course_topic,
            "generation": qa_data["generation"],
            # 시간 정보
            "date": date_str,
            "year": year,
            "year_month": year_month,
            "timestamp": timestamp,
            # 문서 타입
            "doc_type": "qa_pair",
            "has_bot_answer": qa_data["answer"]["is_bot"],
            # 품질 지표
            "has_reactions": has_reactions,
            "reaction_count": reaction_count,
            "answer_count": qa_data.get("answer_count", 1),
            "answer_index": qa_data.get("answer_index", 0),
            # 텍스트 필드
            "question_text": qa_data["question"]["text"],
            "answer_text": qa_data["answer"]["text"],
            "question_user": qa_data["question"].get("user_name")
            or qa_data["question"]["user"],
            "answer_user": qa_data["answer"].get("user_name") or qa_data["answer"]["user"],
            # 추적 정보
            "thread_id": qa_data.get("thread_id", ""),
            "qa_id": qa_data.get("qa_id", ""),
            "source_file": qa_data["source_file"],
        }

        return payload

    def _process_merged_file(self, file_path: Path) -> list[dict[str, Any]]:
        """
        병합된 JSON 파일을 처리하여 Q&A 페어 리스트를 생성합니다.

        매개변수:
            file_path: JSON 파일 경로

        반환값:
            처리된 Q&A 페어 리스트
        """
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        qa_pairs: list[dict[str, Any]] = []
        course = data["course"]

        for qa in data["qa_pairs"]:
            question = qa["question"]
            answers = qa["answers"]
            answer_count = len(answers)

            # 각 답변마다 별도 문서 생성
            for idx, answer in enumerate(answers):
                # Thread ID 생성
                thread_id = f"{qa['generation']}_{course}_{qa['date']}_{question['timestamp']}"

                # QA ID 생성
                qa_id = f"{thread_id}_a{idx}"

                qa_pair = {
                    "generation": qa["generation"],
                    "date": qa["date"],
                    "source_file": qa["source_file"],
                    "course": course,
                    "question": question,
                    "answer": answer,
                    "answer_count": answer_count,
                    "answer_index": idx,
                    "thread_id": thread_id,
                    "qa_id": qa_id,
                }

                qa_pairs.append(qa_pair)

        return qa_pairs

    def ingest_from_directory(self, directory: Path | str) -> dict[str, int]:
        """
        디렉토리 내의 모든 병합 파일을 처리하여 VectorDB에 저장합니다.

        매개변수:
            directory: 병합 파일이 있는 디렉토리 경로

        반환값:
            파일별 처리 개수 딕셔너리
        """
        directory = Path(directory)
        print(f"\n📂 디렉토리 처리: {directory}")

        # JSON 파일 찾기
        json_files = sorted(directory.glob("*_merged.json"))
        print(f"   발견된 파일: {len(json_files)}개")

        if not json_files:
            print("   ⚠️  처리할 파일이 없습니다.")
            return {}

        stats: dict[str, int] = {}
        all_points: list[PointStruct] = []

        # 모든 Q&A 페어 수집
        all_qa_pairs: list[tuple[Path, dict[str, Any]]] = []
        for json_file in json_files:
            try:
                qa_pairs = self._process_merged_file(json_file)
                stats[json_file.name] = len(qa_pairs)
                for qa_pair in qa_pairs:
                    all_qa_pairs.append((json_file, qa_pair))
            except Exception as e:
                print(f"\n   ✗ {json_file.name} 처리 실패: {e}")
                continue

        print(f"\n   총 Q&A 페어: {len(all_qa_pairs):,}개")

        # 배치로 임베딩 생성
        print(f"\n🔄 임베딩 생성 중... (배치 크기: {self.embedding_batch_size})")
        for i in tqdm(
            range(0, len(all_qa_pairs), self.embedding_batch_size),
            desc="임베딩 배치",
        ):
            batch = all_qa_pairs[i : i + self.embedding_batch_size]

            # 임베딩 텍스트 생성
            embedding_texts = [self._create_embedding_text(qa) for _, qa in batch]

            try:
                # 배치 임베딩 생성
                vectors = self.embedding_model.encode(
                    embedding_texts, batch_size=self.embedding_batch_size, convert_to_numpy=True
                )

                # 포인트 생성
                for (json_file, qa_pair), vector in zip(batch, vectors):
                    payload = self._create_payload(qa_pair)
                    point_id = self._generate_doc_id(qa_pair)
                    point = PointStruct(
                        id=point_id, vector=vector.tolist(), payload=payload
                    )
                    all_points.append(point)

            except Exception as e:
                print(f"\n   ✗ 배치 {i//self.embedding_batch_size + 1} 실패: {e}")
                # 배치 실패 시 개별 처리 시도
                for json_file, qa_pair in batch:
                    try:
                        embedding_text = self._create_embedding_text(qa_pair)
                        vector = self.embedding_model.encode(
                            embedding_text, convert_to_numpy=True
                        )
                        payload = self._create_payload(qa_pair)
                        point_id = self._generate_doc_id(qa_pair)
                        point = PointStruct(
                            id=point_id, vector=vector.tolist(), payload=payload
                        )
                        all_points.append(point)
                    except Exception as e2:
                        print(f"\n   ✗ 개별 처리도 실패: {e2}")
                        continue

        # 배치 저장
        print(f"\n💾 VectorDB에 저장 중... (총 {len(all_points)}개 문서)")
        self._batch_upsert(all_points)

        return stats

    def _batch_upsert(self, points: list[PointStruct]) -> None:
        """
        포인트를 배치로 나누어 저장합니다.

        매개변수:
            points: 저장할 포인트 리스트
        """
        total = len(points)
        batches = [points[i : i + self.batch_size] for i in range(0, total, self.batch_size)]

        for batch in tqdm(batches, desc="배치 저장"):
            self.client.upsert(collection_name=self.collection_name, points=batch)

        print(f"   ✅ {total}개 문서 저장 완료")

    def get_collection_info(self) -> None:
        """컬렉션 정보를 출력합니다."""
        try:
            info = self.client.get_collection(self.collection_name)
            print(f"\n📊 컬렉션 정보: {self.collection_name}")
            print(f"   벡터 개수: {info.points_count:,}")
            print(f"   벡터 차원: {self.vector_size}")
            print(f"   거리 측정: Cosine")
        except Exception as e:
            print(f"   ✗ 컬렉션 정보 조회 실패: {e}")

    def test_search(self, query: str, course: str | None = None, limit: int = 5) -> None:
        """
        테스트 검색을 수행합니다.

        매개변수:
            query: 검색 쿼리
            course: 과정 필터 (선택)
            limit: 결과 개수
        """
        print(f"\n🔍 테스트 검색")
        print(f"   쿼리: {query}")
        if course:
            print(f"   과정 필터: {course}")

        # 쿼리 임베딩
        query_vector = self.embedding_model.encode(query, convert_to_numpy=True).tolist()

        # 필터 생성
        query_filter = None
        if course:
            query_filter = Filter(
                must=[FieldCondition(key="course", match=MatchValue(value=course))]
            )

        # 검색 수행
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        ).points

        # 결과 출력
        print(f"\n   결과: {len(results)}개")
        for i, result in enumerate(results, 1):
            print(f"\n   [{i}] 유사도: {result.score:.3f}")
            print(f"       과정: {result.payload['course']} ({result.payload['generation']}기)")
            print(f"       날짜: {result.payload['date']}")
            print(f"       질문: {result.payload['question_text'][:100]}...")
            print(f"       답변: {result.payload['answer_text'][:100]}...")


def main() -> None:
    """메인 함수."""
    import argparse

    parser = argparse.ArgumentParser(description="Slack Q&A를 VectorDB에 저장")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/Users/jhj/Desktop/personal/naver_connect_chatbot/document_chunks/slack_qa_merged",
        help="입력 디렉토리",
    )
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333", help="Qdrant URL")
    parser.add_argument(
        "--collection", type=str, default="slack_qa", help="컬렉션 이름"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen/qwen3-embedding-4b",
        help="임베딩 모델",
    )
    parser.add_argument("--recreate", action="store_true", help="컬렉션 재생성")
    parser.add_argument("--test", action="store_true", help="테스트 검색 수행")

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 Slack Q&A VectorDB 저장 시작")
    print("=" * 80)

    # Ingestion 객체 생성
    ingestion = QAVectorDBIngestion(
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        embedding_model=args.model,
    )

    # 컬렉션 생성
    ingestion.create_collection(recreate=args.recreate)

    # 데이터 저장
    stats = ingestion.ingest_from_directory(args.input_dir)

    # 통계 출력
    print("\n" + "=" * 80)
    print("📊 처리 통계")
    print("=" * 80)
    total_docs = sum(stats.values())
    print(f"처리된 파일: {len(stats)}개")
    print(f"생성된 문서: {total_docs:,}개")

    # 컬렉션 정보
    ingestion.get_collection_info()

    # 테스트 검색
    if args.test:
        print("\n" + "=" * 80)
        print("🧪 테스트 검색")
        print("=" * 80)
        ingestion.test_search("GPU 메모리 부족 문제 해결", course="level2_cv")

    print("\n" + "=" * 80)
    print("✅ 모든 작업 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()

