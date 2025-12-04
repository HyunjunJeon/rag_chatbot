"""
모든 데이터 소스를 통합하여 VectorDB + BM25에 저장하는 스크립트.

Slack Q&A + Notebook + PDF + Weekly Mission 청크를 하나의 Collection에 저장합니다.

확장 가능한 구조:
- DocumentLoader 기본 클래스를 상속하여 새 데이터 소스 추가
- 현재 지원: notebook, slack_qa, pdf, weekly_mission

인덱스:
- VectorDB (Qdrant): Dense 임베딩 검색
- BM25 (Kiwi): Sparse 키워드 검색
"""

import argparse
import hashlib
import json
import re
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_naver import ClovaXEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

# 프로젝트 루트 및 .env 로드
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# 버전 관리 모듈 로드
from document_processing.common.versioning import (
    SCHEMA_VERSION,
    PIPELINE_VERSION,
    CORPUS_VERSION,
    get_current_timestamp,
)

sys.path.insert(0, str(PROJECT_ROOT / "app"))

try:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kiwi_bm25_retriever",
        PROJECT_ROOT
        / "app"
        / "naver_connect_chatbot"
        / "rag"
        / "retriever"
        / "kiwi_bm25_retriever.py",
    )
    kiwi_module = importlib.util.module_from_spec(spec)
    sys.modules["kiwi_bm25_retriever"] = kiwi_module
    spec.loader.exec_module(kiwi_module)

    KiwiBM25Retriever = kiwi_module.KiwiBM25Retriever
    get_default_important_pos = kiwi_module.get_default_important_pos
    BM25_AVAILABLE = True
except Exception as e:
    print(f"⚠️ KiwiBM25Retriever 로드 실패: {e}")
    KiwiBM25Retriever = None
    get_default_important_pos = None
    BM25_AVAILABLE = False

__all__ = ["UnifiedVectorDBIngestion", "DocumentLoader"]


# =============================================================================
# 데이터 소스별 Loader (확장 가능)
# =============================================================================


class DocumentLoader(ABC):
    """
    데이터 소스별 Document Loader 기본 클래스.

    새 데이터 소스 추가 시 이 클래스를 상속하여 구현합니다.

    예시:
        ```python
        class PDFLoader(DocumentLoader):
            doc_type = "pdf"
            default_dir = "document_chunks/pdf_chunks"

            def load(self, directory: Path) -> list[Document]:
                # PDF 청크 로드 로직
                ...
        ```
    """

    doc_type: str = "unknown"  # 문서 타입 (필터링용)
    default_dir: str = ""  # 기본 디렉토리

    @abstractmethod
    def load(self, directory: Path) -> list[Document]:
        """Document 리스트를 로드합니다."""
        pass


class NotebookLoader(DocumentLoader):
    """노트북 청크 Loader."""

    doc_type = "notebook"
    default_dir = "document_chunks/notebook_chunks"

    def load(self, directory: Path) -> list[Document]:
        """Notebook 청크를 Document로 변환."""
        print(f"\n📓 Notebook 청크 로드: {directory}")

        documents: list[Document] = []
        chunk_files = sorted(directory.glob("*_chunks.json"))
        print(f"   파일: {len(chunk_files)}개")

        for chunk_file in chunk_files:
            if chunk_file.name in ("_summary.json", "all_notebook_chunks.json"):
                continue

            try:
                with open(chunk_file, encoding="utf-8") as f:
                    data = json.load(f)

                for chunk in data.get("chunks", []):
                    doc = self._to_document(chunk)
                    documents.append(doc)

            except Exception as e:
                print(f"   ✗ {chunk_file.name}: {e}")

        print(f"   총 문서: {len(documents):,}개")
        return documents

    def _to_document(self, chunk: dict[str, Any]) -> Document:
        """NotebookChunk dict를 Document로 변환."""
        metadata = chunk.get("metadata", {})

        doc_metadata = {
            "doc_id": chunk["id"],
            "doc_type": self.doc_type,
            "source_file": metadata.get("source_file", ""),
            "course": metadata.get("course", ""),
            "topic": metadata.get("topic", ""),
            "subcourse": metadata.get("subcourse", ""),
            "difficulty": metadata.get("difficulty", ""),
            "file_type": metadata.get("file_type", ""),
            "keywords": metadata.get("keywords", []),
        }

        return Document(page_content=chunk["content"], metadata=doc_metadata)


class SlackQALoader(DocumentLoader):
    """
    Slack Q&A Loader.

    품질 필터링:
    - 이모지 코드 제거 (:emoji:)
    - Slack 멘션 제거 (<@USER_ID>)
    - 짧은 답변 필터링
    - 행정적 내용 필터링
    """

    doc_type = "slack_qa"
    default_dir = "document_chunks/slack_qa_auto_filtered"

    # 필터링 설정
    MIN_ANSWER_LENGTH = 50  # 답변 최소 길이
    MIN_QUESTION_LENGTH = 15  # 질문 최소 길이

    # 행정적 키워드 (이 키워드가 포함되면 제외)
    ADMIN_KEYWORDS = [
        "출석",
        "지각",
        "조퇴",
        "팀빌딩",
        "팀 배정",
        "팀원 모집",
        "제출 기한",
        "마감일",
        "마감 일정",
        "zoom 링크",
        "줌 링크",
        "오피스아워",
        "오피스 아워",
        "피어세션",
    ]

    # 단순 응답 패턴 (이것만 있으면 제외)
    SIMPLE_RESPONSE_PATTERNS = [
        r"^감사합니다\.?!?$",
        r"^네\.?!?$",
        r"^네네\.?!?$",
        r"^알겟습니다\.?!?$",
        r"^확인했습니다\.?!?$",
        r"^확인했어요\.?!?$",
        r"^해결되었습니다\.?!?$",
        r"^해결됐습니다\.?!?$",
    ]

    def __init__(self) -> None:
        """SlackQALoader 초기화."""
        self._stats = {
            "total_qa": 0,
            "filtered_emoji": 0,
            "filtered_short": 0,
            "filtered_admin": 0,
            "filtered_simple": 0,
            "passed": 0,
        }

    def load(self, directory: Path) -> list[Document]:
        """Slack Q&A를 Document로 변환 (필터링 적용)."""
        print(f"\n💬 Slack Q&A 로드: {directory}")

        documents: list[Document] = []
        json_files = sorted(directory.glob("*_merged.json"))
        print(f"   파일: {len(json_files)}개")

        for json_file in json_files:
            if json_file.name == "_summary.json":
                continue

            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                course = data["course"]

                for qa in data["qa_pairs"]:
                    docs = self._qa_to_documents(qa, course)
                    documents.extend(docs)

            except Exception as e:
                print(f"   ✗ {json_file.name}: {e}")

        # 필터링 통계 출력
        print(f"   필터링 결과:")
        print(f"      전체 Q&A: {self._stats['total_qa']:,}개")
        print(f"      짧은 답변 제외: {self._stats['filtered_short']:,}개")
        print(f"      행정 내용 제외: {self._stats['filtered_admin']:,}개")
        print(f"      단순 응답 제외: {self._stats['filtered_simple']:,}개")
        print(f"   → 최종 문서: {len(documents):,}개")
        return documents

    def _clean_text(self, text: str) -> str:
        """
        텍스트 정제:
        - Slack 이모지 코드 제거 (:emoji:)
        - Slack 멘션 제거 (<@USER_ID>)
        - 연속 공백 정리
        """
        # Slack 이모지 코드 제거: :emoji_name:
        text = re.sub(r":[a-zA-Z0-9_+-]+:", "", text)
        # Slack 멘션 제거: <@U1234567>
        text = re.sub(r"<@[A-Z0-9]+>", "", text)
        # Slack 채널 링크 제거: <#C1234567|channel-name>
        text = re.sub(r"<#[A-Z0-9]+\|[^>]+>", "", text)
        # 연속 공백 정리
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _should_filter(self, question_text: str, answer_text: str) -> str | None:
        """
        필터링 여부 판단.

        Returns:
            필터링 사유 (필터링 안할 경우 None)
        """
        # 1. 짧은 답변 필터링
        if len(answer_text) < self.MIN_ANSWER_LENGTH:
            return "short_answer"

        # 2. 짧은 질문 필터링
        if len(question_text) < self.MIN_QUESTION_LENGTH:
            return "short_question"

        # 3. 행정적 키워드 필터링
        combined = (question_text + " " + answer_text).lower()
        for keyword in self.ADMIN_KEYWORDS:
            if keyword in combined:
                return "admin_content"

        # 4. 단순 응답 패턴 필터링
        for pattern in self.SIMPLE_RESPONSE_PATTERNS:
            if re.match(pattern, answer_text.strip()):
                return "simple_response"

        return None

    def _qa_to_documents(self, qa: dict[str, Any], course: str) -> list[Document]:
        """QA 페어를 Document 리스트로 변환 (필터링 적용)."""
        documents: list[Document] = []

        question = qa["question"]
        answers = qa["answers"]
        generation = qa["generation"]
        date = qa["date"]

        # 질문 텍스트 정제
        clean_question = self._clean_text(question["text"])

        for idx, answer in enumerate(answers):
            self._stats["total_qa"] += 1

            # 답변 텍스트 정제
            clean_answer = self._clean_text(answer["text"])

            # 필터링 체크
            filter_reason = self._should_filter(clean_question, clean_answer)
            if filter_reason:
                if filter_reason in ("short_answer", "short_question"):
                    self._stats["filtered_short"] += 1
                elif filter_reason == "admin_content":
                    self._stats["filtered_admin"] += 1
                elif filter_reason == "simple_response":
                    self._stats["filtered_simple"] += 1
                continue

            self._stats["passed"] += 1

            doc_id = f"slack_{generation}_{course}_{date}_{question['timestamp']}_a{idx}"

            content = (
                f"[과정: {course}] [기수: {generation}기]\n\n"
                f"질문: {clean_question}\n\n"
                f"답변: {clean_answer}"
            )

            metadata = {
                "doc_id": doc_id,
                "doc_type": self.doc_type,
                "course": course,
                "generation": generation,
                "date": date,
                "question_text": clean_question,
                "answer_text": clean_answer,
                "question_user": question.get("user_name") or question["user"],
                "answer_user": answer.get("user_name") or answer["user"],
                "is_bot": answer["is_bot"],
                "source_file": qa.get("source_file", ""),
            }

            documents.append(Document(page_content=content, metadata=metadata))

        return documents


class PDFLoader(DocumentLoader):
    """PDF 강의 자료 Loader."""

    doc_type = "pdf"
    default_dir = "document_chunks/pdf_chunks"

    def load(self, directory: Path) -> list[Document]:
        """PDF 청크를 Document로 변환."""
        print(f"\n📄 PDF 청크 로드: {directory}")

        documents: list[Document] = []
        chunk_files = sorted(directory.glob("*_chunks.json"))
        print(f"   파일: {len(chunk_files)}개")

        for chunk_file in chunk_files:
            if chunk_file.name in ("_summary.json", "all_pdf_chunks.json"):
                continue

            try:
                with open(chunk_file, encoding="utf-8") as f:
                    data = json.load(f)

                for chunk in data.get("chunks", []):
                    doc = self._to_document(chunk)
                    documents.append(doc)

            except Exception as e:
                print(f"   ✗ {chunk_file.name}: {e}")

        print(f"   총 문서: {len(documents):,}개")
        return documents

    def _to_document(self, chunk: dict[str, Any]) -> Document:
        """PDFChunk dict를 Document로 변환."""
        metadata = chunk.get("metadata", {})

        doc_metadata = {
            "doc_id": chunk["id"],
            "doc_type": self.doc_type,
            "source_file": metadata.get("source_file", ""),
            "course": metadata.get("course", ""),
            "lecture_num": metadata.get("lecture_num", ""),
            "lecture_title": metadata.get("lecture_title", ""),
            "instructor": metadata.get("instructor", ""),
            "page_start": metadata.get("page_start", 0),
            "page_end": metadata.get("page_end", 0),
        }

        return Document(page_content=chunk["content"], metadata=doc_metadata)


class WeeklyMissionLoader(DocumentLoader):
    """주간 미션 Loader."""

    doc_type = "weekly_mission"
    default_dir = "document_chunks/mission_chunks"

    def load(self, directory: Path) -> list[Document]:
        """주간 미션 청크를 Document로 변환."""
        print(f"\n🎯 주간 미션 청크 로드: {directory}")

        documents: list[Document] = []
        chunk_files = sorted(directory.glob("*_chunks.json"))
        print(f"   파일: {len(chunk_files)}개")

        for chunk_file in chunk_files:
            if chunk_file.name in ("_summary.json", "all_mission_chunks.json"):
                continue

            try:
                with open(chunk_file, encoding="utf-8") as f:
                    data = json.load(f)

                for chunk in data.get("chunks", []):
                    doc = self._to_document(chunk)
                    documents.append(doc)

            except Exception as e:
                print(f"   ✗ {chunk_file.name}: {e}")

        print(f"   총 문서: {len(documents):,}개")
        return documents

    def _to_document(self, chunk: dict[str, Any]) -> Document:
        """MissionChunk dict를 Document로 변환."""
        metadata = chunk.get("metadata", {})

        doc_metadata = {
            "doc_id": chunk["id"],
            "doc_type": self.doc_type,
            "source_file": metadata.get("source_file", ""),
            "course": metadata.get("course", ""),
            "week": metadata.get("week", ""),
            "mission_name": metadata.get("mission_name", ""),
            "instructor": metadata.get("instructor", ""),
            "chunk_type": metadata.get("chunk_type", ""),  # problem / rubric
        }

        return Document(page_content=chunk["content"], metadata=doc_metadata)


class LectureTranscriptLoader(DocumentLoader):
    """강의 녹취록 Loader."""

    doc_type = "lecture_transcript"
    default_dir = "document_chunks/lecture_transcript_chunks"

    def load(self, directory: Path) -> list[Document]:
        """강의 녹취록 청크를 Document로 변환."""
        print(f"\n🎙️ 강의 녹취록 로드: {directory}")

        documents: list[Document] = []
        chunk_files = sorted(directory.glob("*_chunks.json"))
        print(f"   파일: {len(chunk_files)}개")

        for chunk_file in chunk_files:
            if chunk_file.name in ("_summary.json",):
                continue

            try:
                with open(chunk_file, encoding="utf-8") as f:
                    data = json.load(f)

                for chunk in data.get("chunks", []):
                    doc = self._to_document(chunk)
                    documents.append(doc)

            except Exception as e:
                print(f"   ✗ {chunk_file.name}: {e}")

        print(f"   총 문서: {len(documents):,}개")
        return documents

    def _to_document(self, chunk: dict[str, Any]) -> Document:
        """LectureTranscriptChunk dict를 Document로 변환."""
        metadata = chunk.get("metadata", {})

        doc_metadata = {
            "doc_id": chunk["id"],
            "doc_type": self.doc_type,
            "source_file": metadata.get("source_file", ""),
            "lecture_name": metadata.get("lecture_name", ""),
            "course": metadata.get("course", ""),
            "lecture_num": metadata.get("lecture_num", ""),
            "lecture_title": metadata.get("lecture_title", ""),
            "chunk_idx": metadata.get("chunk_idx", 0),
            "total_chunks": metadata.get("total_chunks", 1),
        }

        return Document(page_content=chunk["content"], metadata=doc_metadata)


# =============================================================================
# Loader 레지스트리
# =============================================================================

# 지원하는 모든 Loader (여기에 추가)
REGISTERED_LOADERS: dict[str, type[DocumentLoader]] = {
    "notebook": NotebookLoader,
    "slack_qa": SlackQALoader,
    "pdf": PDFLoader,
    "weekly_mission": WeeklyMissionLoader,
    "lecture_transcript": LectureTranscriptLoader,
}


# =============================================================================
# 통합 인제스트
# =============================================================================


class UnifiedVectorDBIngestion:
    """
    모든 데이터 소스를 통합하여 VectorDB에 저장하는 클래스.

    확장 방법:
    1. DocumentLoader 상속 클래스 생성
    2. REGISTERED_LOADERS에 등록
    3. ingest_all() 호출 시 자동 포함

    예시:
        ```python
        ingestion = UnifiedVectorDBIngestion(
            qdrant_url="http://localhost:6333",
            collection_name="naver_connect_docs",
        )

        ingestion.create_collection(recreate=True)
        ingestion.ingest_all(sources=["notebook", "slack_qa"])
        ```
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "naver_connect_docs",
        embedding_model: str = "bge-m3",
        batch_size: int = 100,
        embedding_batch_size: int = 16,
    ) -> None:
        """
        초기화.

        매개변수:
            qdrant_url: Qdrant 서버 URL
            collection_name: 커렉션 이름 (통합)
            embedding_model: NAVER 임베딩 모델 (bge-m3 등)
            batch_size: VectorDB 배치 크기
            embedding_batch_size: 임베딩 API 배치 크기
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size

        # Qdrant 클라이언트
        print(f"🔌 Qdrant 연결: {qdrant_url}")
        self.client = QdrantClient(url=qdrant_url)

        # NAVER Embedding V2 (ClovaXEmbeddings)
        print(f"🤖 NAVER Embedding: {embedding_model}")
        self.embeddings = ClovaXEmbeddings(model=embedding_model)
        # bge-m3: 1024차원
        self.vector_size = 1024
        print(f"   벡터 차원: {self.vector_size}")

    def create_collection(self, recreate: bool = False) -> None:
        """컬렉션 생성."""
        print(f"\n📦 컬렉션: {self.collection_name}")

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists:
            if recreate:
                print("   기존 컬렉션 삭제...")
                self.client.delete_collection(self.collection_name)
            else:
                print("   ⚠️ 이미 존재. recreate=True로 재생성 가능")
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )
        print("   ✅ 생성 완료")

    # =========================================================================
    # 통합 인제스트 (Loader 패턴 사용)
    # =========================================================================

    def ingest_all(
        self,
        sources: list[str] | None = None,
        custom_dirs: dict[str, Path | str] | None = None,
        build_bm25: bool = True,
        bm25_output_dir: Path | str | None = None,
    ) -> dict[str, int]:
        """
        등록된 데이터 소스를 VectorDB + BM25에 저장합니다.

        매개변수:
            sources: 로드할 소스 목록 (None이면 구현된 모든 소스)
                     예: ["notebook", "slack_qa"]
            custom_dirs: 소스별 커스텀 디렉토리
                     예: {"notebook": "/path/to/chunks"}
            build_bm25: BM25 인덱스 생성 여부
            bm25_output_dir: BM25 인덱스 저장 경로

        반환값:
            소스별 문서 수

        예시:
            ```python
            # 모든 소스 로드 (VectorDB + BM25)
            ingestion.ingest_all()

            # 특정 소스만, BM25 없이
            ingestion.ingest_all(sources=["notebook"], build_bm25=False)
            ```
        """
        # 기본값: 모든 구현된 소스
        if sources is None:
            sources = ["notebook", "slack_qa", "pdf", "weekly_mission", "lecture_transcript"]

        if custom_dirs is None:
            custom_dirs = {}

        base_dir = Path(__file__).parent.parent / "document_chunks"

        all_documents: list[Document] = []
        stats: dict[str, int] = {}

        print(f"\n📋 로드할 소스: {sources}")

        for source_name in sources:
            if source_name not in REGISTERED_LOADERS:
                print(f"\n⚠️ 알 수 없는 소스: {source_name}")
                stats[source_name] = 0
                continue

            # Loader 인스턴스 생성
            loader_class = REGISTERED_LOADERS[source_name]
            loader = loader_class()

            # 디렉토리 결정
            if source_name in custom_dirs:
                directory = Path(custom_dirs[source_name])
            else:
                directory = base_dir / loader.default_dir.split("/")[-1]

            # 로드
            if directory.exists():
                docs = loader.load(directory)
                all_documents.extend(docs)
                stats[source_name] = len(docs)
            else:
                print(f"\n⚠️ 디렉토리 없음: {directory}")
                stats[source_name] = 0

        # 통계 출력
        print(f"\n{'=' * 60}")
        print(f"📊 로드 완료: 총 {len(all_documents):,}개")
        for source, count in stats.items():
            print(f"   - {source}: {count:,}개")
        print(f"{'=' * 60}")

        if not all_documents:
            print("\n⚠️ 저장할 문서가 없습니다.")
            return stats

        # 1. VectorDB에 저장
        self._ingest_documents(all_documents)

        # 2. BM25 인덱스 생성
        if build_bm25:
            self._build_bm25_index(all_documents, bm25_output_dir)

        stats["total"] = len(all_documents)
        return stats

    def _build_bm25_index(
        self,
        documents: list[Document],
        output_dir: Path | str | None = None,
    ) -> None:
        """
        Kiwi 형태소 분석기 기반 BM25 인덱스를 생성합니다.

        매개변수:
            documents: Document 리스트
            output_dir: 인덱스 저장 경로
        """
        if not BM25_AVAILABLE:
            print("\n⚠️ KiwiBM25Retriever를 사용할 수 없습니다. BM25 인덱스 생성 건너뜀.")
            return

        if output_dir is None:
            output_dir = PROJECT_ROOT / "sparse_index" / "unified_bm25"
        else:
            output_dir = Path(output_dir)

        print(f"\n{'=' * 60}")
        print("🔍 BM25 인덱스 생성 (Kiwi 형태소 분석)")
        print(f"{'=' * 60}")
        print(f"   문서 수: {len(documents):,}개")
        print(f"   저장 위치: {output_dir}")

        try:
            retriever = KiwiBM25Retriever.from_documents(
                documents=documents,
                k=10,
                model_type="knlm",
                typos=None,
                important_pos=get_default_important_pos(),
                load_default_dict=True,
                load_typo_dict=False,
                load_multi_dict=False,
                num_workers=0,
                auto_save=True,
                save_path=output_dir,
                save_user_dict=True,
            )

            print("\n   ✅ BM25 인덱스 생성 완료")
            print(f"   총 문서: {len(retriever.docs):,}개")

            # 간단한 테스트
            test_query = "PyTorch Tensor 생성"
            results = retriever.invoke(test_query)
            if results:
                print(f"\n   🧪 테스트 검색: '{test_query}'")
                print(f"      상위 결과: {results[0].page_content[:100]}...")

        except Exception as e:
            print(f"\n   ✗ BM25 인덱스 생성 실패: {e}")

    def _split_long_documents(
        self,
        documents: list[Document],
        max_chars: int = 1500,
        overlap: int = 150,
    ) -> list[Document]:
        """긴 문서를 RecursiveCharacterTextSplitter로 분할합니다."""
        # 한국어에 적합한 구분자 설정
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", "。", "? ", "! ", ", ", " ", ""],
            length_function=len,
        )

        result: list[Document] = []
        split_count = 0

        for doc in documents:
            if len(doc.page_content) <= max_chars:
                result.append(doc)
                continue

            # 긴 문서 분할
            split_count += 1
            chunks = text_splitter.split_text(doc.page_content)

            for i, chunk in enumerate(chunks):
                new_metadata = doc.metadata.copy()
                new_metadata["chunk_idx"] = i
                new_metadata["total_chunks"] = len(chunks)
                new_metadata["original_doc_id"] = new_metadata.get("doc_id", "")
                new_metadata["doc_id"] = f"{new_metadata.get('doc_id', '')}__chunk_{i}"

                result.append(Document(page_content=chunk, metadata=new_metadata))

        if split_count > 0:
            print(
                f"   ✂️ {split_count}개 문서 → {len(result) - len(documents) + split_count}개 청크로 분할"
            )

        return result

    def _ingest_documents(self, documents: list[Document]) -> None:
        """Document 리스트를 VectorDB에 저장 (무한 재시도, 절대 멈추지 않음)."""
        # 긴 문서 분할 (1500자 청크)
        max_chunk_chars = 1500
        print(f"\n📄 문서 분할 중... (최대 {max_chunk_chars}자/청크)")
        documents = self._split_long_documents(documents, max_chars=max_chunk_chars)
        print(f"   총 문서: {len(documents):,}개")

        # NAVER API rate limit이 엄격하므로 작은 배치 + 충분한 지연
        batch_size = min(self.embedding_batch_size, 2)  # 최대 2개씩 (안전하게)
        delay_between_batches = 5.0  # 배치 간 5초 대기

        print("\n🔄 임베딩 생성 중... (NAVER Embedding V2)")
        print(f"   배치 크기: {batch_size}, 배치 간 지연: {delay_between_batches}s")
        print("   ⚠️ 무한 재시도 모드: 절대 멈추지 않음")

        all_points: list[PointStruct] = []
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for batch_idx, i in enumerate(range(0, len(documents), batch_size)):
            batch = documents[i : i + batch_size]
            texts = [doc.page_content for doc in batch]

            attempt = 0
            while True:  # 무한 재시도
                try:
                    vectors = self.embeddings.embed_documents(texts)

                    for doc, vector in zip(batch, vectors):
                        point_id = self._generate_point_id(doc)
                        payload = self._create_payload(doc)
                        point = PointStruct(
                            id=point_id,
                            vector=vector,
                            payload=payload,
                        )
                        all_points.append(point)

                    # 성공! 진행률 출력
                    progress = (batch_idx + 1) / total_batches * 100
                    print(
                        f"\r   [{batch_idx + 1}/{total_batches}] {progress:.1f}% - {len(all_points)}개 완료",
                        end="",
                        flush=True,
                    )

                    # 다음 배치 전 대기
                    time.sleep(delay_between_batches)
                    break  # 성공 시 다음 배치로

                except Exception as e:
                    attempt += 1
                    error_msg = str(e)

                    if "429" in error_msg or "rate" in error_msg.lower():
                        # Rate limit - 충분히 긴 대기 (최대 120초)
                        wait_time = min(120, (2 ** min(attempt, 6)) * 2)
                        print(f"\n   ⏳ Rate limit, {wait_time}s 대기 (시도 {attempt})")
                        time.sleep(wait_time)
                    else:
                        # 기타 오류 - 30초 대기 후 재시도
                        print(f"\n   ⚠️ 오류: {error_msg[:100]}, 30s 후 재시도 (시도 {attempt})")
                        time.sleep(30)

        print(f"\n\n💾 저장 중... ({len(all_points):,}개)")
        self._batch_upsert(all_points)
        print("✅ 모든 임베딩 완료!")

    def _generate_point_id(self, doc: Document) -> str:
        """Document에서 고유 ID 생성."""
        doc_id = doc.metadata.get("doc_id", "")
        if doc_id:
            return hashlib.md5(doc_id.encode()).hexdigest()
        # fallback: content hash
        return hashlib.md5(doc.page_content[:500].encode()).hexdigest()

    def _create_payload(self, doc: Document) -> dict[str, Any]:
        """Document에서 payload 생성 (라인리지 필드 포함)."""
        payload = doc.metadata.copy()

        # keywords가 리스트면 문자열로 변환 (Qdrant 호환)
        if "keywords" in payload and isinstance(payload["keywords"], list):
            payload["keywords_list"] = payload["keywords"]
            payload["keywords"] = ", ".join(payload["keywords"])

        # content 저장 (검색 결과에서 바로 접근)
        payload["content"] = doc.page_content

        # 라인리지 필드 추가 (버전 추적용)
        payload["schema_version"] = SCHEMA_VERSION
        payload["pipeline_version"] = PIPELINE_VERSION
        payload["corpus_version"] = CORPUS_VERSION
        payload["ingested_at"] = get_current_timestamp()

        # pipeline_trace가 없으면 기본값 추가
        if "pipeline_trace" not in payload:
            doc_type = payload.get("doc_type", "unknown")
            payload["pipeline_trace"] = [
                f"{doc_type}_loaded",
                "filtered_v2",
                "chunked",
                "ingested",
            ]

        return payload

    def _batch_upsert(self, points: list[PointStruct]) -> None:
        """배치 저장."""
        batches = [points[i : i + self.batch_size] for i in range(0, len(points), self.batch_size)]

        for batch in tqdm(batches, desc="저장"):
            self.client.upsert(collection_name=self.collection_name, points=batch)

        print(f"   ✅ {len(points):,}개 저장 완료")

    def get_collection_info(self) -> None:
        """컬렉션 정보 출력."""
        try:
            info = self.client.get_collection(self.collection_name)
            print(f"\n📊 컬렉션: {self.collection_name}")
            print(f"   문서 수: {info.points_count:,}")
            print(f"   벡터 차원: {self.vector_size}")
        except Exception as e:
            print(f"   ✗ 조회 실패: {e}")

    def test_search(self, query: str, doc_type: str | None = None, limit: int = 5) -> None:
        """테스트 검색."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        print(f"\n🔍 검색: {query}")
        if doc_type:
            print(f"   필터: doc_type={doc_type}")

        query_vector = self.embeddings.embed_query(query)

        query_filter = None
        if doc_type:
            query_filter = Filter(
                must=[FieldCondition(key="doc_type", match=MatchValue(value=doc_type))]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        ).points

        print(f"\n   결과: {len(results)}개")
        for i, r in enumerate(results, 1):
            print(f"\n   [{i}] 유사도: {r.score:.3f}")
            print(f"       타입: {r.payload.get('doc_type')}")
            print(f"       과목: {r.payload.get('course')}")
            content = r.payload.get("content", "")[:150]
            print(f"       내용: {content}...")


def main() -> None:
    """메인 함수."""
    parser = argparse.ArgumentParser(description="통합 VectorDB + BM25 인제스트")
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant URL",
    )
    parser.add_argument(
        "--collection",
        default="naver_connect_docs",
        help="컬렉션 이름",
    )
    parser.add_argument(
        "--model",
        default="bge-m3",
        help="NAVER 임베딩 모델 (bge-m3 등)",
    )
    parser.add_argument("--recreate", action="store_true", help="컬렉션 재생성")
    parser.add_argument("--test", action="store_true", help="테스트 검색")
    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        default=None,
        help="로드할 소스 (notebook, slack_qa, pdf, weekly_mission)",
    )
    parser.add_argument(
        "--no-bm25",
        action="store_true",
        help="BM25 인덱스 생성 건너뛰기",
    )
    parser.add_argument(
        "--bm25-dir",
        type=str,
        default="./sparse_index/unified_bm25",
        help="BM25 인덱스 저장 경로",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 통합 인덱스 생성 (VectorDB + BM25)")
    print("=" * 80)
    print(f"   지원 소스: {list(REGISTERED_LOADERS.keys())}")
    print(f"   BM25 사용: {'❌ 비활성화' if args.no_bm25 else '✅ 활성화'}")

    ingestion = UnifiedVectorDBIngestion(
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        embedding_model=args.model,
    )

    ingestion.create_collection(recreate=args.recreate)

    stats = ingestion.ingest_all(
        sources=args.sources,
        build_bm25=not args.no_bm25,
        bm25_output_dir=args.bm25_dir,
    )

    print("\n" + "=" * 80)
    print("📊 결과")
    print("=" * 80)
    for source, count in stats.items():
        print(f"   {source}: {count:,}개")

    ingestion.get_collection_info()

    if args.test:
        print("\n" + "=" * 80)
        print("🧪 테스트 검색 (VectorDB)")
        print("=" * 80)
        ingestion.test_search("PyTorch Tensor 생성 방법", doc_type="notebook")
        ingestion.test_search("GPU 메모리 부족", doc_type="slack_qa")

    print("\n" + "=" * 80)
    print("✅ 완료!")
    print("=" * 80)
    print("\n생성된 인덱스:")
    print(f"   📦 VectorDB: {args.collection} (Qdrant)")
    if not args.no_bm25:
        bm25_path = args.bm25_dir or "./sparse_index/unified_bm25"
        print(f"   🔍 BM25: {bm25_path} (Kiwi)")


if __name__ == "__main__":
    main()
