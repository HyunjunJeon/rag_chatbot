"""
Jupyter Notebook 파일을 로드하고 파싱하는 모듈.

.ipynb 파일의 JSON 구조를 분석하여 셀 타입별로 분류하고,
RAG에 유용한 콘텐츠를 추출합니다.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class CellType(str, Enum):
    """셀 타입 열거형."""

    MARKDOWN = "markdown"
    CODE = "code"
    RAW = "raw"


class FileType(str, Enum):
    """파일 타입 (문제/정답) 열거형."""

    PROBLEM = "문제"
    SOLUTION = "정답"
    UNKNOWN = "알수없음"


class Difficulty(str, Enum):
    """난이도 열거형."""

    BASIC = "기본"
    ADVANCED = "심화"
    UNKNOWN = "알수없음"


@dataclass
class NotebookCell:
    """
    단일 노트북 셀을 나타내는 클래스.

    Attributes:
        cell_type: 셀 타입 (markdown, code, raw)
        source: 셀의 소스 코드/텍스트
        outputs: 코드 셀의 출력 결과
        cell_index: 노트북 내 셀 인덱스
        metadata: 셀 메타데이터
    """

    cell_type: CellType
    source: str
    outputs: list[dict[str, Any]] = field(default_factory=list)
    cell_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """셀이 비어있는지 확인."""
        return not self.source.strip()

    @property
    def is_heading(self) -> bool:
        """마크다운 셀이 헤딩으로 시작하는지 확인."""
        if self.cell_type != CellType.MARKDOWN:
            return False
        return bool(re.match(r"^#{1,6}\s+", self.source.strip()))

    @property
    def heading_level(self) -> int | None:
        """헤딩 레벨 반환 (1-6). 헤딩이 아니면 None."""
        if not self.is_heading:
            return None
        match = re.match(r"^(#{1,6})\s+", self.source.strip())
        return len(match.group(1)) if match else None

    @property
    def heading_text(self) -> str | None:
        """헤딩 텍스트 반환. 헤딩이 아니면 None."""
        if not self.is_heading:
            return None
        match = re.match(r"^#{1,6}\s+(.+)$", self.source.strip().split("\n")[0])
        return match.group(1) if match else None

    def get_output_text(self, max_lines: int = 50) -> str:
        """
        출력 결과를 텍스트로 변환.

        Args:
            max_lines: 최대 출력 라인 수

        Returns:
            출력 텍스트 (없으면 빈 문자열)
        """
        if not self.outputs:
            return ""

        text_parts: list[str] = []

        for output in self.outputs:
            output_type = output.get("output_type", "")

            if output_type == "stream":
                # stdout/stderr 출력
                text = output.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                text_parts.append(text)

            elif output_type == "execute_result":
                # 실행 결과
                data = output.get("data", {})
                if "text/plain" in data:
                    text = data["text/plain"]
                    if isinstance(text, list):
                        text = "".join(text)
                    text_parts.append(text)

            elif output_type == "error":
                # 에러 메시지
                ename = output.get("ename", "Error")
                evalue = output.get("evalue", "")
                traceback = output.get("traceback", [])
                # ANSI 코드 제거
                error_text = f"{ename}: {evalue}"
                text_parts.append(error_text)

        combined = "\n".join(text_parts)
        lines = combined.split("\n")

        if len(lines) > max_lines:
            return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines}줄 생략)"

        return combined


@dataclass
class ParsedNotebook:
    """
    파싱된 노트북을 나타내는 클래스.

    Attributes:
        file_path: 노트북 파일 경로
        cells: 파싱된 셀 리스트
        file_type: 문제/정답 구분
        difficulty: 난이도 (기본/심화)
        course: 과목명
        topic: 주제명
        metadata: 노트북 메타데이터
    """

    file_path: Path
    cells: list[NotebookCell]
    file_type: FileType = FileType.UNKNOWN
    difficulty: Difficulty = Difficulty.UNKNOWN
    course: str = ""
    topic: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def markdown_cells(self) -> list[NotebookCell]:
        """마크다운 셀만 반환."""
        return [c for c in self.cells if c.cell_type == CellType.MARKDOWN]

    @property
    def code_cells(self) -> list[NotebookCell]:
        """코드 셀만 반환."""
        return [c for c in self.cells if c.cell_type == CellType.CODE]

    @property
    def non_empty_cells(self) -> list[NotebookCell]:
        """비어있지 않은 셀만 반환."""
        return [c for c in self.cells if not c.is_empty]

    def get_title(self) -> str | None:
        """노트북 제목 추출 (첫 번째 H1 헤딩)."""
        for cell in self.markdown_cells:
            if cell.heading_level == 1:
                return cell.heading_text
        return None


class NotebookLoader:
    """
    Jupyter Notebook 파일을 로드하고 파싱하는 클래스.

    이 클래스는 .ipynb 파일을 읽어서:
    1. JSON 구조 파싱
    2. 셀 타입별 분류 (markdown, code, raw)
    3. 문제/정답 파일 구분
    4. 과목/주제/난이도 메타데이터 추출

    예시:
        ```python
        loader = NotebookLoader()
        notebook = loader.load_from_file("path/to/notebook.ipynb")

        for cell in notebook.cells:
            print(f"[{cell.cell_type}] {cell.source[:50]}...")
        ```
    """

    # 파일명에서 문제/정답 구분을 위한 패턴
    PROBLEM_PATTERNS = [r"\(문제\)", r"_문제", r"문제\.ipynb$"]
    SOLUTION_PATTERNS = [r"\(정답\)", r"_정답", r"정답\.ipynb$", r"\(해설\)", r"_해설"]

    # 난이도 패턴
    BASIC_PATTERNS = [r"\(기본", r"기본-", r"기본_"]
    ADVANCED_PATTERNS = [r"\(심화", r"심화-", r"심화_"]

    def __init__(self) -> None:
        """NotebookLoader 초기화."""
        pass

    def load_from_file(self, file_path: Path | str) -> ParsedNotebook:
        """
        .ipynb 파일을 로드하고 파싱합니다.

        Args:
            file_path: 노트북 파일 경로

        Returns:
            ParsedNotebook 객체

        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            json.JSONDecodeError: JSON 파싱 오류가 발생한 경우
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        if not file_path.suffix == ".ipynb":
            raise ValueError(f"지원하지 않는 파일 형식입니다: {file_path.suffix}")

        with open(file_path, encoding="utf-8") as f:
            notebook_data = json.load(f)

        # 셀 파싱
        cells = self._parse_cells(notebook_data)

        # 메타데이터 추출
        file_type = self._detect_file_type(file_path)
        difficulty = self._detect_difficulty(file_path)
        course, topic = self._extract_course_topic(file_path)

        return ParsedNotebook(
            file_path=file_path,
            cells=cells,
            file_type=file_type,
            difficulty=difficulty,
            course=course,
            topic=topic,
            metadata={
                "nbformat": notebook_data.get("nbformat"),
                "nbformat_minor": notebook_data.get("nbformat_minor"),
                "kernel": notebook_data.get("metadata", {})
                .get("kernelspec", {})
                .get("display_name"),
            },
        )

    def load_from_directory(
        self,
        directory_path: Path | str,
        recursive: bool = True,
        solution_only: bool = False,
    ) -> list[ParsedNotebook]:
        """
        디렉토리 내의 모든 노트북 파일을 로드합니다.

        Args:
            directory_path: 디렉토리 경로
            recursive: 하위 디렉토리도 탐색할지 여부
            solution_only: 정답 파일만 로드할지 여부

        Returns:
            ParsedNotebook 객체 리스트
        """
        directory_path = Path(directory_path)
        notebooks: list[ParsedNotebook] = []

        pattern = "**/*.ipynb" if recursive else "*.ipynb"

        for ipynb_file in sorted(directory_path.glob(pattern)):
            try:
                notebook = self.load_from_file(ipynb_file)

                # 정답 파일만 필터링
                if solution_only and notebook.file_type != FileType.SOLUTION:
                    continue

                notebooks.append(notebook)

            except Exception as e:
                print(f"⚠️ 파일 로드 실패: {ipynb_file.name} - {e}")
                continue

        return notebooks

    def _parse_cells(self, notebook_data: dict[str, Any]) -> list[NotebookCell]:
        """
        노트북 JSON에서 셀을 파싱합니다.

        Args:
            notebook_data: 노트북 JSON 데이터

        Returns:
            NotebookCell 객체 리스트
        """
        cells: list[NotebookCell] = []
        raw_cells = notebook_data.get("cells", [])

        for idx, raw_cell in enumerate(raw_cells):
            cell_type_str = raw_cell.get("cell_type", "raw")

            try:
                cell_type = CellType(cell_type_str)
            except ValueError:
                cell_type = CellType.RAW

            # source 처리 (리스트 또는 문자열)
            source = raw_cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)

            # outputs 처리 (코드 셀만)
            outputs = raw_cell.get("outputs", []) if cell_type == CellType.CODE else []

            cell = NotebookCell(
                cell_type=cell_type,
                source=source,
                outputs=outputs,
                cell_index=idx,
                metadata=raw_cell.get("metadata", {}),
            )
            cells.append(cell)

        return cells

    def _detect_file_type(self, file_path: Path) -> FileType:
        """파일명에서 문제/정답 구분."""
        filename = file_path.name

        for pattern in self.SOLUTION_PATTERNS:
            if re.search(pattern, filename, re.IGNORECASE):
                return FileType.SOLUTION

        for pattern in self.PROBLEM_PATTERNS:
            if re.search(pattern, filename, re.IGNORECASE):
                return FileType.PROBLEM

        return FileType.UNKNOWN

    def _detect_difficulty(self, file_path: Path) -> Difficulty:
        """파일 경로에서 난이도 추출."""
        path_str = str(file_path)

        for pattern in self.ADVANCED_PATTERNS:
            if re.search(pattern, path_str, re.IGNORECASE):
                return Difficulty.ADVANCED

        for pattern in self.BASIC_PATTERNS:
            if re.search(pattern, path_str, re.IGNORECASE):
                return Difficulty.BASIC

        return Difficulty.UNKNOWN

    def _extract_course_topic(self, file_path: Path) -> tuple[str, str]:
        """
        파일 경로에서 과목명과 주제명을 추출합니다.

        경로 예시:
        - practice/01. AI Core/01. PyTorch/(기본-2) Linear Regression/...
        - home_work/3. MRC/(1강-실습) Looking into KorQuAD.ipynb

        Returns:
            (과목명, 주제명) 튜플
        """
        parts = file_path.parts
        course = ""
        topic = ""

        # 경로에서 과목명과 주제명 추출
        for i, part in enumerate(parts):
            # 숫자로 시작하는 부분 (01. PyTorch, 3. MRC 등)
            if re.match(r"^\d+[\.\s]", part):
                # 숫자 prefix 제거
                cleaned = re.sub(r"^\d+[\.\s]+", "", part)
                # 마스터명 제거 (괄호 안의 내용)
                cleaned = re.sub(r"\s*\([^)]+마스터\)", "", cleaned)

                if not course:
                    course = cleaned
                else:
                    topic = cleaned

            # 괄호로 시작하는 부분 (실습명)
            elif re.match(r"^\(", part):
                if not topic:
                    # 괄호 내용 정리
                    topic = part

        # topic이 파일명인 경우 처리
        if not topic and file_path.stem:
            topic = file_path.stem

        return course, topic


def main() -> None:
    """테스트 및 데모용 메인 함수."""
    print("=" * 80)
    print("📓 NotebookLoader 테스트")
    print("=" * 80)

    loader = NotebookLoader()

    # PROJECT_ROOT 기준 상대 경로 사용
    project_root = Path(__file__).parent.parent.parent

    # 테스트 경로
    test_dirs = [
        project_root / "original_documents" / "practice",
        project_root / "original_documents" / "home_work",
    ]

    for test_dir in test_dirs:
        if test_dir.exists():
            print(f"\n📁 {test_dir.name} 폴더 스캔 중...")
            notebooks = loader.load_from_directory(test_dir, recursive=True)

            print(f"   발견된 노트북: {len(notebooks)}개")

            for nb in notebooks[:3]:  # 처음 3개만 표시
                print(f"\n   📔 {nb.file_path.name}")
                print(f"      과목: {nb.course}")
                print(f"      주제: {nb.topic}")
                print(f"      타입: {nb.file_type.value}")
                print(f"      난이도: {nb.difficulty.value}")
                print(
                    f"      셀 수: {len(nb.cells)} (MD: {len(nb.markdown_cells)}, Code: {len(nb.code_cells)})"
                )
                title = nb.get_title()
                if title:
                    print(f"      제목: {title}")


if __name__ == "__main__":
    main()
