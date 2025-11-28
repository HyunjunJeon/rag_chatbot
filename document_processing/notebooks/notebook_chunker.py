"""
Jupyter Notebook 청킹 모듈.

파싱된 노트북에서 RAG 시스템을 위한 의미있는 청크를 생성합니다.
섹션 단위로 관련 셀들을 그룹화하여 컨텍스트를 유지합니다.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .notebook_loader import (
    CellType,
    FileType,
    NotebookCell,
    ParsedNotebook,
)
from ..common.versioning import create_chunk_version_metadata
from ..common.filters import remove_copyright_notices


@dataclass
class NotebookChunk:
    """
    노트북에서 추출한 하나의 청크를 나타내는 클래스.

    Attributes:
        id: 고유 식별자
        content: 청크 내용 (마크다운 + 코드 조합)
        metadata: 청크 메타데이터
    """

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """대략적인 토큰 수 추정 (한글 기준 글자수 / 1.5)."""
        # 한글은 대략 1.5 글자당 1 토큰, 영문은 4 글자당 1 토큰
        korean_chars = len(re.findall(r"[가-힣]", self.content))
        other_chars = len(self.content) - korean_chars
        return int(korean_chars / 1.5 + other_chars / 4)

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "token_estimate": self.token_estimate,
        }


class NotebookChunker:
    """
    노트북을 RAG용 청크로 분할하는 클래스.

    청킹 전략:
    1. 섹션 기반 분할 (H2/H3 헤딩 기준)
    2. 관련 셀 그룹화 (마크다운 + 코드 + 출력)
    3. 크기 제한 (max_tokens)
    4. 메타데이터 보존

    예시:
        ```python
        chunker = NotebookChunker(max_tokens=500)
        chunks = chunker.chunk_notebook(parsed_notebook)

        for chunk in chunks:
            print(f"[{chunk.id}] {chunk.content[:100]}...")
        ```
    """

    # 의미없는 코드 패턴 (문제 파일의 빈 코드)
    EMPTY_CODE_PATTERNS = [
        r"^\s*#\s*TODO",
        r"^\s*pass\s*$",
        r"^\s*\.\.\.\s*$",
        r"^#.*작성.*코드",
        r"^#.*여기에.*작성",
        r"^#.*your\s+code",
    ]

    # import만 있는 코드 패턴
    IMPORT_ONLY_PATTERN = r"^(\s*(import|from)\s+.+\n?)+$"

    def __init__(
        self,
        max_tokens: int = 500,
        min_tokens: int = 50,
        include_outputs: bool = True,
        max_output_lines: int = 30,
        solution_only: bool = True,
        enable_filtering: bool = True,
    ) -> None:
        """
        NotebookChunker 초기화.

        Args:
            max_tokens: 청크 최대 토큰 수
            min_tokens: 청크 최소 토큰 수 (이보다 작으면 다음 섹션과 병합)
            include_outputs: 코드 출력 포함 여부
            max_output_lines: 출력 최대 라인 수
            solution_only: 정답 파일만 코드 포함
            enable_filtering: 필터링 활성화 (저작권 고지 제거)
        """
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.include_outputs = include_outputs
        self.max_output_lines = max_output_lines
        self.solution_only = solution_only
        self.enable_filtering = enable_filtering

    def chunk_notebook(self, notebook: ParsedNotebook) -> list[NotebookChunk]:
        """
        노트북을 청크로 분할합니다.

        Args:
            notebook: 파싱된 노트북

        Returns:
            NotebookChunk 리스트
        """
        # 빈 노트북 체크
        if not notebook.non_empty_cells:
            return []

        # 섹션별로 셀 그룹화
        sections = self._split_into_sections(notebook)

        # 각 섹션을 청크로 변환
        chunks: list[NotebookChunk] = []
        for section_idx, section_cells in enumerate(sections):
            section_chunks = self._create_chunks_from_section(
                notebook=notebook,
                section_cells=section_cells,
                section_idx=section_idx,
            )
            chunks.extend(section_chunks)

        # 너무 작은 청크 병합
        chunks = self._merge_small_chunks(chunks)

        return chunks

    def chunk_notebooks(self, notebooks: list[ParsedNotebook]) -> list[NotebookChunk]:
        """
        여러 노트북을 일괄 청킹합니다.

        Args:
            notebooks: ParsedNotebook 리스트

        Returns:
            모든 노트북의 NotebookChunk 리스트
        """
        all_chunks: list[NotebookChunk] = []

        for notebook in notebooks:
            chunks = self.chunk_notebook(notebook)
            all_chunks.extend(chunks)

        return all_chunks

    def _split_into_sections(self, notebook: ParsedNotebook) -> list[list[NotebookCell]]:
        """
        노트북을 섹션별로 분할합니다.

        H2 또는 H3 헤딩을 기준으로 분할합니다.

        Returns:
            섹션별 셀 리스트
        """
        sections: list[list[NotebookCell]] = []
        current_section: list[NotebookCell] = []

        for cell in notebook.cells:
            # 새 섹션 시작 조건: H1, H2, H3 헤딩
            if cell.is_heading and cell.heading_level and cell.heading_level <= 3:
                # 이전 섹션 저장
                if current_section:
                    sections.append(current_section)
                current_section = [cell]
            else:
                current_section.append(cell)

        # 마지막 섹션 저장
        if current_section:
            sections.append(current_section)

        return sections

    def _create_chunks_from_section(
        self,
        notebook: ParsedNotebook,
        section_cells: list[NotebookCell],
        section_idx: int,
    ) -> list[NotebookChunk]:
        """
        하나의 섹션에서 청크를 생성합니다.

        Args:
            notebook: 원본 노트북
            section_cells: 섹션 내 셀들
            section_idx: 섹션 인덱스

        Returns:
            NotebookChunk 리스트
        """
        chunks: list[NotebookChunk] = []
        content_parts: list[str] = []
        cell_indices: list[int] = []
        cell_types: list[str] = []

        for cell in section_cells:
            # 빈 셀 스킵
            if cell.is_empty:
                continue

            # 셀 타입별 처리
            if cell.cell_type == CellType.MARKDOWN:
                content_parts.append(cell.source.strip())
                cell_indices.append(cell.cell_index)
                cell_types.append("markdown")

            elif cell.cell_type == CellType.CODE:
                # 코드 포함 조건 확인
                if self._should_include_code(cell, notebook.file_type):
                    # 의미없는 코드 필터링
                    if not self._is_empty_code(cell.source):
                        code_block = f"```python\n{cell.source.strip()}\n```"
                        content_parts.append(code_block)
                        cell_indices.append(cell.cell_index)
                        cell_types.append("code")

                        # 출력 포함
                        if self.include_outputs:
                            output_text = cell.get_output_text(self.max_output_lines)
                            if output_text.strip():
                                output_block = f"```\n# Output:\n{output_text.strip()}\n```"
                                content_parts.append(output_block)
                                cell_types.append("output")

            # 토큰 제한 체크 - 너무 커지면 청크 생성
            combined = "\n\n".join(content_parts)
            estimated_tokens = self._estimate_tokens(combined)

            if estimated_tokens > self.max_tokens and len(content_parts) > 1:
                # 마지막 부분 제외하고 청크 생성
                chunk_content = "\n\n".join(content_parts[:-1])
                chunk = self._create_chunk(
                    notebook=notebook,
                    content=chunk_content,
                    section_idx=section_idx,
                    chunk_idx=len(chunks),
                    cell_indices=cell_indices[:-1],
                    cell_types=cell_types[:-1],
                )
                chunks.append(chunk)

                # 마지막 부분으로 새 시작
                content_parts = [content_parts[-1]]
                cell_indices = [cell_indices[-1]] if cell_indices else []
                cell_types = [cell_types[-1]] if cell_types else []

        # 남은 내용으로 청크 생성
        if content_parts:
            chunk_content = "\n\n".join(content_parts)
            chunk = self._create_chunk(
                notebook=notebook,
                content=chunk_content,
                section_idx=section_idx,
                chunk_idx=len(chunks),
                cell_indices=cell_indices,
                cell_types=cell_types,
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        notebook: ParsedNotebook,
        content: str,
        section_idx: int,
        chunk_idx: int,
        cell_indices: list[int],
        cell_types: list[str],
    ) -> NotebookChunk:
        """
        청크 객체를 생성합니다.

        Args:
            notebook: 원본 노트북
            content: 청크 내용
            section_idx: 섹션 인덱스
            chunk_idx: 청크 인덱스 (섹션 내)
            cell_indices: 포함된 셀 인덱스들
            cell_types: 포함된 셀 타입들 (미사용, 하위 호환성 유지)

        Returns:
            NotebookChunk 객체
        """
        # 고유 ID 생성
        chunk_id = self._generate_chunk_id(notebook, section_idx, chunk_idx)

        # 필터링: 저작권 고지 제거
        if self.enable_filtering:
            content = remove_copyright_notices(content)

        # 상대 경로 생성 (original_documents 기준)
        source_path = str(notebook.file_path)
        if "original_documents" in source_path:
            relative_path = source_path.split("original_documents/")[-1]
        else:
            relative_path = notebook.file_path.name

        # subcourse 추출 (topic에서 더 세부적인 정보)
        subcourse = self._extract_subcourse(notebook.file_path, notebook.topic)

        # 키워드 추출
        keywords = self._extract_keywords(content, notebook)

        # Content 앞에 키워드 헤더 추가 (임베딩 품질 향상)
        if keywords:
            keywords_header = f"[키워드: {', '.join(keywords)}]\n\n"
            enriched_content = keywords_header + content
        else:
            enriched_content = content

        # 메타데이터 구성
        metadata = {
            "doc_type": "notebook",
            "source_file": relative_path,
            "course": notebook.course,
            "topic": notebook.topic,
            "subcourse": subcourse,
            "difficulty": notebook.difficulty.value,
            "file_type": notebook.file_type.value,
            "keywords": keywords,
            "section_idx": section_idx,
            "chunk_idx": chunk_idx,
            "cell_range": [min(cell_indices), max(cell_indices)] if cell_indices else [],
        }

        # 버전 메타데이터 추가
        version_meta = create_chunk_version_metadata(
            source_file=notebook.file_path,
            include_hash=True,
        )
        metadata.update(version_meta)

        return NotebookChunk(id=chunk_id, content=enriched_content, metadata=metadata)

    def _generate_chunk_id(self, notebook: ParsedNotebook, section_idx: int, chunk_idx: int) -> str:
        """청크 고유 ID 생성."""
        # 파일 경로 기반 prefix
        parts = []
        if notebook.course:
            parts.append(self._slugify(notebook.course))
        if notebook.topic:
            parts.append(self._slugify(notebook.topic))

        prefix = "_".join(parts) if parts else "notebook"

        # 섹션/청크 인덱스 추가
        base_id = f"{prefix}_s{section_idx:02d}_c{chunk_idx:02d}"

        # 짧은 해시 추가 (충돌 방지)
        content_hash = hashlib.md5(str(notebook.file_path).encode()).hexdigest()[:6]

        return f"{base_id}_{content_hash}"

    def _slugify(self, text: str) -> str:
        """텍스트를 URL/ID 친화적으로 변환."""
        # 한글 유지, 특수문자 제거
        text = re.sub(r"[^\w가-힣\s-]", "", text)
        text = re.sub(r"[-\s]+", "_", text)
        return text.lower().strip("_")[:30]

    def _should_include_code(self, cell: NotebookCell, file_type: FileType) -> bool:
        """
        코드 셀을 포함할지 결정합니다.

        문제 파일의 코드는 대부분 빈 템플릿이므로 제외합니다.
        """
        if not self.solution_only:
            return True

        # 정답 파일만 코드 포함
        return file_type == FileType.SOLUTION

    def _is_empty_code(self, source: str) -> bool:
        """의미없는 빈 코드인지 확인."""
        source = source.strip()

        # 빈 문자열
        if not source:
            return True

        # 빈 코드 패턴 체크
        for pattern in self.EMPTY_CODE_PATTERNS:
            if re.search(pattern, source, re.MULTILINE | re.IGNORECASE):
                return True

        # import만 있는 코드
        if re.match(self.IMPORT_ONLY_PATTERN, source, re.MULTILINE):
            # import만 있으면 제외
            return True

        return False

    def _estimate_tokens(self, text: str) -> int:
        """토큰 수 추정."""
        korean_chars = len(re.findall(r"[가-힣]", text))
        other_chars = len(text) - korean_chars
        return int(korean_chars / 1.5 + other_chars / 4)

    def _extract_subcourse(self, file_path: Path, topic: str) -> str:
        """
        파일 경로에서 세부 과목(subcourse)을 추출합니다.

        예시:
        - "(기본-1) Tensor의 생성, 조작, 연산" → "Tensor의 생성, 조작, 연산"
        - "(심화-1) BERT4Rec" → "BERT4Rec"
        """
        # 폴더명에서 추출 시도
        for part in file_path.parts:
            # (기본-1) 또는 (심화-1) 패턴
            match = re.match(r"^\([기본심화]-?\d*\)\s*(.+)$", part)
            if match:
                return match.group(1).strip()

        # 파일명에서 추출 시도
        stem = file_path.stem
        # (정답), (문제) 제거
        stem = re.sub(r"\s*\((정답|문제|해설)\)\s*", "", stem)
        # (기본-1), (심화-1) 패턴 매칭
        match = re.match(r"^\([기본심화]-?\d*\)\s*(.+)$", stem)
        if match:
            return match.group(1).strip()

        # topic이 있으면 topic 사용
        if topic and topic != file_path.stem:
            return topic

        return ""

    def _extract_keywords(self, content: str, notebook: ParsedNotebook) -> list[str]:
        """
        청크 내용에서 주요 키워드를 추출합니다.

        추출 대상:
        - 주요 기술 용어 (PyTorch, Tensor, BERT 등)
        - 영문 대문자로 된 약어 (CUDA, GPU, MLP 등)
        - 코드에서 자주 등장하는 클래스/함수명
        """
        keywords: set[str] = set()

        # 1. 과목/주제에서 키워드 추출
        if notebook.course:
            keywords.add(notebook.course)
        if notebook.topic:
            # 괄호 제거하고 추가
            topic_clean = re.sub(r"^\([^)]+\)\s*", "", notebook.topic)
            if topic_clean:
                keywords.add(topic_clean)

        # 2. 기술 용어 패턴 매칭
        tech_patterns = [
            r"\b(PyTorch|TensorFlow|Keras|NumPy|Pandas)\b",
            r"\b(Tensor|Matrix|Vector|Array)\b",
            r"\b(BERT|GPT|Transformer|Attention|RNN|LSTM|GRU|CNN)\b",
            r"\b(Linear|Conv2d|BatchNorm|Dropout|Softmax|ReLU)\b",
            r"\b(Adam|SGD|Optimizer|Loss|CrossEntropy)\b",
            r"\b(GPU|CUDA|CPU)\b",
            r"\b(Embedding|Tokenizer|Encoder|Decoder)\b",
            r"\b(FAISS|BM25|TF-IDF|Retrieval|Retriever)\b",
            r"\b(KorQuAD|SQuAD|ODQA|MRC)\b",
            r"\b(RecSys|Recommendation|Collaborative|Filtering)\b",
            r"\b(Segmentation|Detection|Classification)\b",
            r"\b(VAE|GAN|Diffusion|Contrastive)\b",
            r"\b(HCCF|SASRec|BERT4Rec|DeepFM|NCF)\b",
        ]

        for pattern in tech_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                # 원래 케이스 유지
                keywords.add(match)

        # 3. 영문 대문자 약어 추출 (3글자 이상)
        acronyms = re.findall(r"\b([A-Z]{3,})\b", content)
        for acronym in acronyms:
            if acronym not in {"TODO", "NOTE", "FIXME", "OUTPUT"}:
                keywords.add(acronym)

        # 4. 중복 제거 및 정렬 (최대 8개)
        keywords_list = sorted(keywords, key=lambda x: (-len(x), x.lower()))
        return keywords_list[:8]

    def _merge_small_chunks(self, chunks: list[NotebookChunk]) -> list[NotebookChunk]:
        """
        너무 작은 청크를 이전 청크와 병합합니다.

        Args:
            chunks: 원본 청크 리스트

        Returns:
            병합된 청크 리스트
        """
        if not chunks:
            return chunks

        merged: list[NotebookChunk] = []

        for chunk in chunks:
            if not merged:
                merged.append(chunk)
                continue

            # 현재 청크가 너무 작고, 이전 청크와 같은 파일인 경우 병합
            if chunk.token_estimate < self.min_tokens and merged[-1].metadata.get(
                "source_file"
            ) == chunk.metadata.get("source_file"):
                # 이전 청크에 병합
                prev = merged[-1]
                new_content = f"{prev.content}\n\n{chunk.content}"

                # 병합 후 크기 체크
                if self._estimate_tokens(new_content) <= self.max_tokens * 1.2:
                    # 메타데이터 업데이트 (키워드 병합)
                    new_metadata = prev.metadata.copy()
                    prev_keywords = set(prev.metadata.get("keywords", []))
                    chunk_keywords = set(chunk.metadata.get("keywords", []))
                    new_metadata["keywords"] = list(prev_keywords | chunk_keywords)[:8]

                    merged[-1] = NotebookChunk(
                        id=prev.id,
                        content=new_content,
                        metadata=new_metadata,
                    )
                    continue

            merged.append(chunk)

        return merged


def main() -> None:
    """테스트 및 데모용 메인 함수."""
    from .notebook_loader import NotebookLoader

    print("=" * 80)
    print("📝 NotebookChunker 테스트")
    print("=" * 80)

    loader = NotebookLoader()
    chunker = NotebookChunker(max_tokens=500, solution_only=True)

    # PROJECT_ROOT 기준 상대 경로 사용
    project_root = Path(__file__).parent.parent.parent

    # 테스트 경로
    test_dirs = [
        project_root / "original_documents" / "practice",
        project_root / "original_documents" / "home_work",
    ]

    for test_dir in test_dirs:
        if test_dir.exists():
            print(f"\n📁 {test_dir.name} 폴더 처리 중...")

            # 정답 파일만 로드
            notebooks = loader.load_from_directory(test_dir, recursive=True, solution_only=True)
            print(f"   정답 노트북: {len(notebooks)}개")

            # 청킹
            all_chunks = chunker.chunk_notebooks(notebooks)
            print(f"   생성된 청크: {len(all_chunks)}개")

            # 샘플 출력
            for chunk in all_chunks[:2]:
                print(f"\n   📄 {chunk.id}")
                print(f"      과목: {chunk.metadata.get('course')}")
                print(f"      주제: {chunk.metadata.get('topic')}")
                print(f"      세부과목: {chunk.metadata.get('subcourse')}")
                print(f"      토큰: ~{chunk.token_estimate}")
                print(f"      키워드: {chunk.metadata.get('keywords')}")
                print(f"      내용 미리보기: {chunk.content[:100]}...")


if __name__ == "__main__":
    main()
