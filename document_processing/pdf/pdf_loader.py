"""
PDF 파일을 로드하고 파싱하는 모듈.

PyMuPDF4LLM을 사용하여 PDF를 마크다운 형식으로 변환하고,
파일명 패턴에서 메타데이터(과목, 강 번호, 주제, 마스터명)를 추출합니다.
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "document_processing"))

from base import parse_pdf


@dataclass
class PDFPage:
    """
    PDF의 단일 페이지(슬라이드)를 나타내는 클래스.

    Attributes:
        page_num: 페이지 번호 (1-indexed)
        text_content: 마크다운 형식의 텍스트 내용
        images: 페이지에 포함된 이미지 정보 리스트
        tables: 페이지에 포함된 테이블 정보 리스트
        metadata: 페이지 메타데이터 (원본 pymupdf4llm 결과)
    """

    page_num: int
    text_content: str
    images: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """페이지가 비어있는지 확인 (텍스트 없음)."""
        return not self.text_content.strip()

    @property
    def char_count(self) -> int:
        """텍스트 문자 수."""
        return len(self.text_content)

    @property
    def has_images(self) -> bool:
        """이미지 포함 여부."""
        return len(self.images) > 0

    @property
    def has_tables(self) -> bool:
        """테이블 포함 여부."""
        return len(self.tables) > 0


@dataclass
class ParsedPDF:
    """
    파싱된 PDF 문서를 나타내는 클래스.

    Attributes:
        file_path: PDF 파일 경로
        pages: 파싱된 페이지 리스트
        course: 과목명 (예: "PyTorch", "AI Math")
        lecture_num: 강 번호 (예: 7)
        topic: 주제명 (예: "Linear Regression1")
        instructor: 마스터명 (예: "오영석")
        metadata: 추가 메타데이터
    """

    file_path: Path
    pages: list[PDFPage]
    course: str = ""
    lecture_num: int = 0
    topic: str = ""
    instructor: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_pages(self) -> int:
        """전체 페이지 수."""
        return len(self.pages)

    @property
    def non_empty_pages(self) -> list[PDFPage]:
        """비어있지 않은 페이지만 반환."""
        return [p for p in self.pages if not p.is_empty]

    @property
    def total_chars(self) -> int:
        """전체 문자 수."""
        return sum(p.char_count for p in self.pages)

    def get_full_text(self) -> str:
        """전체 텍스트를 하나의 문자열로 반환."""
        return "\n\n".join(p.text_content for p in self.pages if not p.is_empty)


class PDFLoader:
    """
    PDF 파일을 로드하고 파싱하는 클래스.

    PyMuPDF4LLM을 사용하여 PDF를 마크다운으로 변환합니다.
    파일명 및 폴더명 패턴에서 메타데이터를 자동 추출합니다.

    예시:
        ```python
        loader = PDFLoader()
        pdf = loader.load_from_file("path/to/lecture.pdf")

        print(f"과목: {pdf.course}")
        print(f"강: {pdf.lecture_num}강")
        print(f"주제: {pdf.topic}")

        for page in pdf.pages:
            print(f"[Page {page.page_num}] {page.text_content[:100]}...")
        ```
    """

    # 파일명 패턴: [과목] (n강) 주제.pdf
    # 예: [PyTorch] (7강) Linear Regression1.pdf
    FILENAME_PATTERN = re.compile(
        r"\[(?P<course>[^\]]+)\]\s*\((?P<num>\d+)강\)\s*(?P<topic>.+)\.pdf$",
        re.IGNORECASE,
    )

    # 폴더명 패턴: 번호. 과목명 (마스터명 마스터)
    # 예: 01. PyTorch (오영석 마스터)
    FOLDER_PATTERN = re.compile(r"^\d+\.\s*(?P<course>.+?)\s*\((?P<instructor>\w+)\s*마스터\)$")

    def __init__(self, verbose: bool = False) -> None:
        """
        PDFLoader 초기화.

        Args:
            verbose: 상세 로그 출력 여부
        """
        self.verbose = verbose

    def load_from_file(self, file_path: Path | str) -> ParsedPDF:
        """
        단일 PDF 파일을 로드하고 파싱합니다.

        Args:
            file_path: PDF 파일 경로

        Returns:
            ParsedPDF 객체

        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            ValueError: PDF 파일이 아닌 경우
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"PDF 파일이 아닙니다: {file_path.suffix}")

        if self.verbose:
            print(f"📄 PDF 로드: {file_path.name}")

        # PyMuPDF4LLM으로 파싱
        raw_result = parse_pdf(file_path)

        # 페이지별로 PDFPage 객체 생성
        pages = self._parse_pages(raw_result)

        # 메타데이터 추출
        course, lecture_num, topic = self._extract_from_filename(file_path.name)
        instructor = self._extract_instructor(file_path)

        return ParsedPDF(
            file_path=file_path,
            pages=pages,
            course=course,
            lecture_num=lecture_num,
            topic=topic,
            instructor=instructor,
            metadata={
                "source_file": str(file_path),
                "file_name": file_path.name,
            },
        )

    def load_from_directory(
        self,
        directory: Path | str,
        recursive: bool = True,
    ) -> list[ParsedPDF]:
        """
        디렉토리 내 모든 PDF 파일을 로드합니다.

        Args:
            directory: PDF 파일이 있는 디렉토리
            recursive: 하위 폴더 포함 여부

        Returns:
            ParsedPDF 객체 리스트
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {directory}")

        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = sorted(directory.glob(pattern))

        if self.verbose:
            print(f"📂 디렉토리: {directory}")
            print(f"   PDF 파일: {len(pdf_files)}개")

        pdfs: list[ParsedPDF] = []

        for pdf_file in pdf_files:
            try:
                pdf = self.load_from_file(pdf_file)
                pdfs.append(pdf)
            except Exception as e:
                print(f"   ⚠️ 로드 실패: {pdf_file.name} - {e}")

        return pdfs

    def _parse_pages(self, raw_result: list[dict[str, Any]]) -> list[PDFPage]:
        """
        PyMuPDF4LLM 결과를 PDFPage 객체 리스트로 변환.

        Args:
            raw_result: pymupdf4llm.to_markdown(page_chunks=True) 결과

        Returns:
            PDFPage 리스트
        """
        pages: list[PDFPage] = []

        for idx, page_data in enumerate(raw_result):
            # 텍스트 추출
            text_content = page_data.get("text", "")

            # 이미지 정보 (있는 경우)
            images = page_data.get("images", [])

            # 테이블 정보 (있는 경우)
            tables = page_data.get("tables", [])

            page = PDFPage(
                page_num=idx + 1,  # 1-indexed
                text_content=text_content,
                images=images,
                tables=tables,
                metadata=page_data.get("metadata", {}),
            )
            pages.append(page)

        return pages

    def _extract_from_filename(self, filename: str) -> tuple[str, int, str]:
        """
        파일명에서 과목, 강 번호, 주제를 추출.

        예시:
            "[PyTorch] (7강) Linear Regression1.pdf"
            → ("PyTorch", 7, "Linear Regression1")

        Args:
            filename: PDF 파일명

        Returns:
            (course, lecture_num, topic) 튜플
        """
        match = self.FILENAME_PATTERN.search(filename)

        if match:
            course = match.group("course").strip()
            lecture_num = int(match.group("num"))
            topic = match.group("topic").strip()
            return course, lecture_num, topic

        # 패턴 매칭 실패 시 파일명에서 최대한 추출
        name_without_ext = Path(filename).stem
        return "", 0, name_without_ext

    def _extract_instructor(self, file_path: Path) -> str:
        """
        폴더명에서 마스터(강사)명을 추출.

        예시:
            "01. PyTorch (오영석 마스터)/[PyTorch] (1강) Intro.pdf"
            → "오영석"

        Args:
            file_path: PDF 파일 경로

        Returns:
            마스터명 (추출 실패 시 빈 문자열)
        """
        # 상위 폴더들 확인
        for parent in file_path.parents:
            match = self.FOLDER_PATTERN.match(parent.name)
            if match:
                return match.group("instructor")

        return ""


# =============================================================================
# CLI 테스트
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python pdf_loader.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    loader = PDFLoader(verbose=True)

    try:
        pdf = loader.load_from_file(pdf_path)

        print(f"\n{'=' * 60}")
        print(f"📄 파일: {pdf.file_path.name}")
        print(f"📚 과목: {pdf.course}")
        print(f"📖 강: {pdf.lecture_num}강")
        print(f"📝 주제: {pdf.topic}")
        print(f"👨‍🏫 마스터: {pdf.instructor}")
        print(f"📃 페이지: {pdf.total_pages}개")
        print(f"📊 문자 수: {pdf.total_chars:,}자")
        print(f"{'=' * 60}\n")

        # 첫 3 페이지 미리보기
        for page in pdf.pages[:3]:
            print(f"--- Page {page.page_num} ---")
            preview = page.text_content[:500]
            print(preview)
            if len(page.text_content) > 500:
                print("...")
            print()

    except Exception as e:
        print(f"❌ 오류: {e}")
        sys.exit(1)
