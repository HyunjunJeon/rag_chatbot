"""
PDF 청킹 모듈.

LangChain의 RecursiveCharacterTextSplitter를 사용하여 PDF를 RAG 시스템용 청크로 분할합니다.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .pdf_loader import ParsedPDF, PDFPage
from ..common.versioning import create_chunk_version_metadata
from ..common.filters import (
    is_toc_page,
    remove_copyright_notices,
    remove_headers_footers,
)


@dataclass
class PDFChunk:
    """
    PDF에서 추출한 하나의 청크를 나타내는 클래스.

    Attributes:
        id: 고유 식별자
        content: 청크 내용
        metadata: 청크 메타데이터
    """

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """
        대략적인 토큰 수 추정.

        한글은 약 1.5 글자당 1 토큰, 영문은 약 4 글자당 1 토큰으로 계산.
        """
        korean_chars = len(re.findall(r"[가-힣]", self.content))
        other_chars = len(self.content) - korean_chars
        return int(korean_chars / 1.5 + other_chars / 4)

    @property
    def char_count(self) -> int:
        """문자 수."""
        return len(self.content)

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "token_estimate": self.token_estimate,
            "char_count": self.char_count,
        }


class PDFChunker:
    """
    PDF를 RAG용 청크로 분할하는 클래스.

    LangChain의 RecursiveCharacterTextSplitter를 사용하여
    의미 단위로 텍스트를 분할합니다.

    청킹 전략:
    1. 전체 PDF 텍스트를 하나로 결합
    2. RecursiveCharacterTextSplitter로 의미 단위 분할
    3. 각 청크에 메타데이터 부여

    예시:
        ```python
        chunker = PDFChunker(chunk_size=1000, chunk_overlap=100)
        chunks = chunker.chunk_pdf(parsed_pdf)

        for chunk in chunks:
            print(f"[{chunk.id}] {chunk.content[:100]}...")
        ```
    """

    # 한국어에 적합한 구분자 (우선순위 순)
    DEFAULT_SEPARATORS = [
        "\n\n\n",  # 페이지/섹션 구분
        "\n\n",  # 단락 구분
        "\n",  # 줄바꿈
        ".\n",  # 문장 끝 + 줄바꿈
        ". ",  # 문장 끝
        "。",  # 한/중/일 마침표
        "? ",  # 물음표
        "! ",  # 느낌표
        ";\n",  # 세미콜론 + 줄바꿈
        "; ",  # 세미콜론
        ", ",  # 쉼표
        " ",  # 공백
        "",  # 문자 단위 (최후 수단)
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
        length_function: str = "tokens",
        enable_filtering: bool = True,
    ) -> None:
        """
        PDFChunker 초기화.

        Args:
            chunk_size: 청크 최대 크기 (토큰 또는 문자)
            chunk_overlap: 청크 간 오버랩 크기
            separators: 텍스트 분할 구분자 리스트
            length_function: 길이 측정 방식 ("tokens" 또는 "chars")
            enable_filtering: 필터링 활성화 (목차/저작권/헤더푸터 제거)
        """
        self.enable_filtering = enable_filtering
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.length_function = length_function

        # 토큰 기반이면 문자 수로 변환 (한글 1.5자/토큰, 영문 4자/토큰 → 평균 2.5자/토큰)
        if length_function == "tokens":
            char_size = int(chunk_size * 2.5)
            char_overlap = int(chunk_overlap * 2.5)
        else:
            char_size = chunk_size
            char_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=char_size,
            chunk_overlap=char_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_pdf(self, pdf: ParsedPDF) -> list[PDFChunk]:
        """
        PDF를 청크로 분할합니다.

        Args:
            pdf: 파싱된 PDF

        Returns:
            PDFChunk 리스트
        """
        # 빈 PDF 체크
        if not pdf.non_empty_pages:
            return []

        # 필터링: 목차 페이지 제외
        pages_to_process = pdf.pages
        if self.enable_filtering:
            pages_to_process = [page for page in pdf.pages if not is_toc_page(page.text)]

        if not pages_to_process:
            return []

        # 전체 텍스트 결합 (페이지 구분자 포함)
        full_text = self._combine_pages(pages_to_process)

        # 필터링: 저작권 고지 및 헤더/푸터 제거
        if self.enable_filtering:
            full_text = remove_copyright_notices(full_text)
            full_text = remove_headers_footers(full_text)

        if not full_text.strip():
            return []

        # RecursiveCharacterTextSplitter로 분할
        text_chunks = self.text_splitter.split_text(full_text)

        # PDFChunk 객체로 변환
        chunks: list[PDFChunk] = []
        for idx, text in enumerate(text_chunks):
            chunk = self._create_chunk(
                pdf=pdf,
                content=text,
                chunk_idx=idx,
                total_chunks=len(text_chunks),
            )
            chunks.append(chunk)

        return chunks

    def chunk_pdfs(self, pdfs: list[ParsedPDF]) -> list[PDFChunk]:
        """
        여러 PDF를 일괄 청킹합니다.

        Args:
            pdfs: ParsedPDF 리스트

        Returns:
            모든 PDF의 PDFChunk 리스트
        """
        all_chunks: list[PDFChunk] = []

        for pdf in pdfs:
            chunks = self.chunk_pdf(pdf)
            all_chunks.extend(chunks)

        return all_chunks

    def _combine_pages(self, pages: list[PDFPage]) -> str:
        """
        페이지들을 하나의 텍스트로 결합.

        페이지 구분을 위해 충분한 줄바꿈을 추가합니다.

        Args:
            pages: PDFPage 리스트

        Returns:
            결합된 텍스트
        """
        texts: list[str] = []

        for page in pages:
            if page.is_empty:
                continue

            # 페이지 텍스트 정리
            text = page.text_content.strip()

            # 페이지 번호 주석 추가 (검색 시 참조용)
            page_text = f"[Page {page.page_num}]\n{text}"
            texts.append(page_text)

        return "\n\n\n".join(texts)

    def _create_chunk(
        self,
        pdf: ParsedPDF,
        content: str,
        chunk_idx: int,
        total_chunks: int,
    ) -> PDFChunk:
        """
        청크 객체를 생성합니다.

        Args:
            pdf: 원본 PDF
            content: 청크 내용
            chunk_idx: 청크 인덱스
            total_chunks: 전체 청크 수

        Returns:
            PDFChunk 객체
        """
        # 청크 ID 생성
        chunk_id = self._generate_chunk_id(pdf, chunk_idx)

        # 메타데이터 구성
        metadata = {
            "source_file": str(pdf.file_path),
            "file_name": pdf.file_path.name,
            "course": pdf.course,
            "lecture_num": pdf.lecture_num,
            "topic": pdf.topic,
            "instructor": pdf.instructor,
            "chunk_idx": chunk_idx,
            "total_chunks": total_chunks,
            "doc_type": "pdf",
        }

        # 버전 메타데이터 추가
        version_meta = create_chunk_version_metadata(
            source_file=pdf.file_path,
            include_hash=True,
        )
        metadata.update(version_meta)

        return PDFChunk(
            id=chunk_id,
            content=content,
            metadata=metadata,
        )

    def _generate_chunk_id(self, pdf: ParsedPDF, chunk_idx: int) -> str:
        """
        청크 고유 ID를 생성합니다.

        형식: pdf_{course}_{lecture_num}_{chunk_idx}_{hash}

        Args:
            pdf: 원본 PDF
            chunk_idx: 청크 인덱스

        Returns:
            고유 ID 문자열
        """
        # 과목명 정규화 (공백 → 언더스코어, 소문자)
        course_slug = pdf.course.lower().replace(" ", "_").replace("-", "_")
        if not course_slug:
            course_slug = "unknown"

        # 짧은 해시 (파일 경로 기반)
        hash_input = f"{pdf.file_path}_{chunk_idx}".encode()
        short_hash = hashlib.md5(hash_input).hexdigest()[:6]

        lecture_str = f"lecture{pdf.lecture_num:02d}" if pdf.lecture_num else "lecture00"

        return f"pdf_{course_slug}_{lecture_str}_c{chunk_idx:03d}_{short_hash}"


# =============================================================================
# CLI 테스트
# =============================================================================

if __name__ == "__main__":
    import sys
    from .pdf_loader import PDFLoader

    if len(sys.argv) < 2:
        print("사용법: python -m pdf.pdf_chunker <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # PDF 로드
    loader = PDFLoader(verbose=True)
    pdf = loader.load_from_file(pdf_path)

    # 청킹 (1000 토큰)
    chunker = PDFChunker(chunk_size=1000, chunk_overlap=100)
    chunks = chunker.chunk_pdf(pdf)

    print(f"\n{'=' * 60}")
    print(f"📄 파일: {pdf.file_path.name}")
    print(f"📚 과목: {pdf.course} | 📖 {pdf.lecture_num}강 | 📝 {pdf.topic}")
    print(f"📃 페이지: {pdf.total_pages}개 → 🧩 청크: {len(chunks)}개")
    print(f"{'=' * 60}\n")

    # 청크 미리보기
    for i, chunk in enumerate(chunks[:5]):
        print(f"--- Chunk {i} ({chunk.token_estimate} tokens) ---")
        preview = chunk.content[:300]
        print(preview)
        if len(chunk.content) > 300:
            print("...")
        print()
