"""
BM25 인덱스를 생성하고 관리하는 모듈.

Hybrid Search를 위한 BM25 검색 기능을 제공합니다.
"""

import json
import pickle
from pathlib import Path
from typing import Any

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi
from tqdm import tqdm

__all__ = ["BM25Indexer"]


class BM25Indexer:
    """
    BM25 인덱스를 생성하고 검색하는 클래스.

    질문과 답변을 분리하여 인덱싱하고, 가중치를 조절하여 검색할 수 있습니다.

    예시:
        ```python
        indexer = BM25Indexer()

        # 인덱스 생성
        indexer.build_index_from_directory(
            "document_chunks/slack_qa_merged/"
        )

        # 인덱스 저장
        indexer.save_index("bm25_index.pkl")

        # 검색
        results = indexer.search(
            query="GPU 메모리 부족",
            question_weight=0.7,
            answer_weight=0.3,
            limit=10
        )
        ```
    """

    def __init__(self) -> None:
        """초기화."""
        print("🔧 BM25Indexer 초기화 중...")

        # 한국어 형태소 분석기
        print("   Kiwi 형태소 분석기 로드 중...")
        self.kiwi = Kiwi()

        # BM25 인덱스 (질문, 답변, 통합)
        self.bm25_question: BM25Okapi | None = None
        self.bm25_answer: BM25Okapi | None = None
        self.bm25_combined: BM25Okapi | None = None

        # 문서 ID 매핑
        self.doc_ids: list[str] = []

        # 원본 문서 저장 (검색 결과 반환용)
        self.documents: list[dict[str, Any]] = []

        print("   ✅ 초기화 완료")

    def tokenize(self, text: str) -> list[str]:
        """
        텍스트를 토큰화합니다.

        매개변수:
            text: 토큰화할 텍스트

        반환값:
            토큰 리스트
        """
        # Kiwi로 형태소 분석
        tokens = self.kiwi.tokenize(text)

        # 명사, 동사, 형용사, 영어, 숫자만 추출
        result = []
        for token in tokens:
            if token.tag in ["NNG", "NNP", "VV", "VA", "SL", "SN"]:  # 명사, 동사, 형용사, 영어, 숫자
                result.append(token.form.lower())

        return result

    def _process_merged_file(self, file_path: Path) -> list[dict[str, Any]]:
        """
        병합된 JSON 파일을 처리합니다.

        매개변수:
            file_path: JSON 파일 경로

        반환값:
            처리된 문서 리스트
        """
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        documents: list[dict[str, Any]] = []
        course = data["course"]

        for qa in data["qa_pairs"]:
            question = qa["question"]
            answers = qa["answers"]

            # 각 답변마다 문서 생성
            for idx, answer in enumerate(answers):
                # 문서 ID 생성
                doc_id = (
                    f"{qa['generation']}_{course}_{qa['date']}_"
                    f"{question['timestamp']}_a{idx}"
                )

                # 토큰화
                question_tokens = self.tokenize(question["text"])
                answer_tokens = self.tokenize(answer["text"])
                combined_tokens = question_tokens + answer_tokens

                doc = {
                    "doc_id": doc_id,
                    "generation": qa["generation"],
                    "course": course,
                    "date": qa["date"],
                    "question_text": question["text"],
                    "answer_text": answer["text"],
                    "question_tokens": question_tokens,
                    "answer_tokens": answer_tokens,
                    "combined_tokens": combined_tokens,
                    "question_user": question.get("user_name") or question["user"],
                    "answer_user": answer.get("user_name") or answer["user"],
                    "is_bot": answer["is_bot"],
                }

                documents.append(doc)

        return documents

    def build_index_from_directory(self, directory: Path | str) -> None:
        """
        디렉토리의 모든 JSON 파일을 읽어 BM25 인덱스를 생성합니다.

        매개변수:
            directory: JSON 파일이 있는 디렉토리 경로
        """
        directory = Path(directory)
        print(f"\n📂 BM25 인덱스 생성: {directory}")

        # JSON 파일 찾기
        json_files = sorted(directory.glob("*_merged.json"))
        print(f"   발견된 파일: {len(json_files)}개")

        # 모든 문서 수집
        all_documents: list[dict[str, Any]] = []

        for json_file in tqdm(json_files, desc="파일 처리"):
            try:
                documents = self._process_merged_file(json_file)
                all_documents.extend(documents)
            except Exception as e:
                print(f"\n   ✗ {json_file.name} 처리 실패: {e}")
                continue

        print(f"\n   총 문서 수: {len(all_documents):,}개")

        # 문서 저장
        self.documents = all_documents
        self.doc_ids = [doc["doc_id"] for doc in all_documents]

        # 토큰 리스트 생성
        print("   BM25 인덱스 생성 중...")
        question_corpus = [doc["question_tokens"] for doc in all_documents]
        answer_corpus = [doc["answer_tokens"] for doc in all_documents]
        combined_corpus = [doc["combined_tokens"] for doc in all_documents]

        # BM25 인덱스 생성
        self.bm25_question = BM25Okapi(question_corpus)
        self.bm25_answer = BM25Okapi(answer_corpus)
        self.bm25_combined = BM25Okapi(combined_corpus)

        print("   ✅ 인덱스 생성 완료")

    def save_index(self, output_path: Path | str) -> None:
        """
        BM25 인덱스를 파일로 저장합니다.

        매개변수:
            output_path: 저장할 파일 경로
        """
        output_path = Path(output_path)
        print(f"\n💾 BM25 인덱스 저장: {output_path}")

        data = {
            "bm25_question": self.bm25_question,
            "bm25_answer": self.bm25_answer,
            "bm25_combined": self.bm25_combined,
            "doc_ids": self.doc_ids,
            "documents": self.documents,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        file_size = output_path.stat().st_size / 1024 / 1024
        print(f"   ✅ 저장 완료: {file_size:.2f} MB")

    def load_index(self, index_path: Path | str) -> None:
        """
        저장된 BM25 인덱스를 로드합니다.

        매개변수:
            index_path: 인덱스 파일 경로
        """
        index_path = Path(index_path)
        print(f"\n📥 BM25 인덱스 로드: {index_path}")

        if not index_path.exists():
            raise FileNotFoundError(f"인덱스 파일이 없습니다: {index_path}")

        with open(index_path, "rb") as f:
            data = pickle.load(f)

        self.bm25_question = data["bm25_question"]
        self.bm25_answer = data["bm25_answer"]
        self.bm25_combined = data["bm25_combined"]
        self.doc_ids = data["doc_ids"]
        self.documents = data["documents"]

        print(f"   ✅ 로드 완료: {len(self.documents):,}개 문서")

    def search(
        self,
        query: str,
        question_weight: float = 0.7,
        answer_weight: float = 0.3,
        limit: int = 10,
        course_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        BM25로 검색합니다.

        매개변수:
            query: 검색 쿼리
            question_weight: 질문 가중치 (0~1)
            answer_weight: 답변 가중치 (0~1)
            limit: 반환할 문서 개수
            course_filter: 과정 필터 (선택)

        반환값:
            검색 결과 리스트
        """
        if self.bm25_question is None or self.bm25_answer is None:
            raise ValueError("인덱스가 로드되지 않았습니다. build_index() 또는 load_index()를 먼저 실행하세요.")

        # 쿼리 토큰화
        query_tokens = self.tokenize(query)

        # BM25 점수 계산
        question_scores = self.bm25_question.get_scores(query_tokens)
        answer_scores = self.bm25_answer.get_scores(query_tokens)

        # 가중치 적용
        combined_scores = (
            question_weight * question_scores + answer_weight * answer_scores
        )

        # 과정 필터 적용
        if course_filter:
            for i, doc in enumerate(self.documents):
                if doc["course"] != course_filter:
                    combined_scores[i] = -1  # 제외

        # 상위 문서 선택
        import numpy as np

        top_indices = np.argsort(combined_scores)[::-1][:limit]

        # 결과 생성
        results = []
        for idx in top_indices:
            if combined_scores[idx] > 0:  # 점수가 0보다 큰 것만
                doc = self.documents[idx].copy()
                doc["bm25_score"] = float(combined_scores[idx])
                results.append(doc)

        return results

    def get_stats(self) -> dict[str, Any]:
        """
        인덱스 통계를 반환합니다.

        반환값:
            통계 딕셔너리
        """
        if not self.documents:
            return {"total_documents": 0}

        # 과정별 통계
        course_counts: dict[str, int] = {}
        for doc in self.documents:
            course = doc["course"]
            course_counts[course] = course_counts.get(course, 0) + 1

        return {
            "total_documents": len(self.documents),
            "unique_courses": len(course_counts),
            "by_course": course_counts,
        }


def main() -> None:
    """메인 함수."""
    import argparse

    parser = argparse.ArgumentParser(description="BM25 인덱스 생성")
    # PROJECT_ROOT 기준 상대 경로 사용
    project_root = Path(__file__).parent.parent

    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(project_root / "document_chunks" / "slack_qa_merged"),
        help="입력 디렉토리 (기본값: document_chunks/slack_qa_merged)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(project_root / "document_chunks" / "bm25_index.pkl"),
        help="출력 파일 (기본값: document_chunks/bm25_index.pkl)",
    )
    parser.add_argument("--test", action="store_true", help="테스트 검색 수행")

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 BM25 인덱스 생성 시작")
    print("=" * 80)

    # 인덱서 생성
    indexer = BM25Indexer()

    # 인덱스 빌드
    indexer.build_index_from_directory(args.input_dir)

    # 통계 출력
    stats = indexer.get_stats()
    print(f"\n📊 인덱스 통계")
    print(f"   총 문서: {stats['total_documents']:,}개")
    print(f"   과정 수: {stats['unique_courses']}개")

    # 저장
    indexer.save_index(args.output)

    # 테스트 검색
    if args.test:
        print("\n" + "=" * 80)
        print("🧪 테스트 검색")
        print("=" * 80)

        test_queries = [
            ("GPU 메모리 부족", "level2_cv"),
            ("데이터 증강 기법", "level2_cv"),
            ("optimizer 선택", None),
        ]

        for query, course in test_queries:
            print(f"\n🔍 쿼리: {query}")
            if course:
                print(f"   과정: {course}")

            results = indexer.search(
                query=query,
                question_weight=0.7,
                answer_weight=0.3,
                limit=3,
                course_filter=course,
            )

            for i, result in enumerate(results, 1):
                print(f"\n   [{i}] BM25 점수: {result['bm25_score']:.3f}")
                print(f"       과정: {result['course']} ({result['generation']}기)")
                print(f"       질문: {result['question_text'][:80]}...")
                print(f"       답변: {result['answer_text'][:80]}...")

    print("\n" + "=" * 80)
    print("✅ 모든 작업 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()

