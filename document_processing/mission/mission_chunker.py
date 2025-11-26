"""
주간 미션 청킹 모듈.

문제 파일에서 학습 목표, 문제 설명, 힌트만 추출합니다.
⚠️ 정답 코드는 제외하여 힌트 형태로만 제공합니다.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .mission_loader import ParsedMission, MissionType
from ..common.versioning import create_chunk_version_metadata


@dataclass
class MissionChunk:
    """
    미션에서 추출한 청크.

    Attributes:
        id: 고유 식별자
        content: 청크 내용
        metadata: 메타데이터
    """

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """토큰 수 추정."""
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


class MissionChunker:
    """
    미션을 RAG용 청크로 분할하는 클래스.

    문제 파일에서는 마크다운(설명/힌트)만 추출하고,
    코드는 제외하여 정답 유출을 방지합니다.

    예시:
        ```python
        chunker = MissionChunker(chunk_size=1000)
        chunks = chunker.chunk_mission(parsed_mission)

        for chunk in chunks:
            print(f"[{chunk.id}] {chunk.content[:100]}...")
        ```
    """

    # 한국어에 적합한 구분자
    DEFAULT_SEPARATORS = [
        "\n\n\n",
        "\n\n",
        "\n",
        ". ",
        "。",
        "? ",
        "! ",
        ", ",
        " ",
        "",
    ]

    # base64 이미지 패턴 (마크다운 이미지 및 인라인 base64)
    BASE64_IMAGE_PATTERNS = [
        # 마크다운 이미지: ![alt](data:image/...;base64,...)
        r"!\[[^\]]*\]\(data:image\/[^;]+;base64,[A-Za-z0-9+/=\s]+\)",
        # HTML 이미지 태그: <img src="data:image/...;base64,...">
        r"<img[^>]*src=['\"]data:image\/[^;]+;base64,[A-Za-z0-9+/=\s]+['\"][^>]*>",
        # 순수 base64 데이터 블록 (100자 이상 연속)
        r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{100,}={0,2}(?![A-Za-z0-9+/])",
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        include_code_hints: bool = True,
    ) -> None:
        """
        MissionChunker 초기화.

        Args:
            chunk_size: 청크 최대 토큰 수
            chunk_overlap: 청크 간 오버랩 토큰 수
            include_code_hints: 코드 주석/TODO를 힌트로 포함할지 여부
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.include_code_hints = include_code_hints

        # 토큰 → 문자 변환 (평균 2.5자/토큰)
        char_size = int(chunk_size * 2.5)
        char_overlap = int(chunk_overlap * 2.5)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=char_size,
            chunk_overlap=char_overlap,
            separators=self.DEFAULT_SEPARATORS,
            length_function=len,
        )

    def chunk_mission(self, mission: ParsedMission) -> list[MissionChunk]:
        """
        미션을 청크로 분할합니다.

        Args:
            mission: 파싱된 미션

        Returns:
            MissionChunk 리스트
        """
        if mission.mission_type == MissionType.PROBLEM:
            return self._chunk_problem(mission)
        elif mission.mission_type == MissionType.RUBRIC:
            return self._chunk_rubric(mission)
        else:
            # 정답 파일은 빈 리스트 반환
            return []

    def chunk_missions(self, missions: list[ParsedMission]) -> list[MissionChunk]:
        """여러 미션을 일괄 청킹합니다."""
        all_chunks: list[MissionChunk] = []

        for mission in missions:
            chunks = self.chunk_mission(mission)
            all_chunks.extend(chunks)

        return all_chunks

    def _chunk_problem(self, mission: ParsedMission) -> list[MissionChunk]:
        """
        문제 노트북을 청킹합니다.

        마크다운 셀(설명/힌트)만 추출하고 코드는 제외합니다.
        """
        if not mission.notebook:
            return []

        notebook = mission.notebook

        # 마크다운 셀만 추출
        markdown_texts: list[str] = []

        for cell in notebook.cells:
            if cell.cell_type.value == "markdown" and not cell.is_empty:
                # base64 이미지 제거
                cleaned_source = self._remove_base64_images(cell.source)
                if cleaned_source.strip():
                    markdown_texts.append(cleaned_source)

            # 코드 셀에서 힌트 추출 (선택적)
            elif self.include_code_hints and cell.cell_type.value == "code":
                hints = self._extract_code_hints(cell.source)
                if hints:
                    markdown_texts.append(hints)

        if not markdown_texts:
            return []

        # 전체 텍스트 결합
        full_text = "\n\n".join(markdown_texts)

        # 메타데이터 헤더 추가
        header = self._create_header(mission)
        full_text = f"{header}\n\n{full_text}"

        # RecursiveCharacterTextSplitter로 분할
        text_chunks = self.text_splitter.split_text(full_text)

        # MissionChunk 객체로 변환
        chunks: list[MissionChunk] = []
        for idx, text in enumerate(text_chunks):
            chunk = self._create_chunk(
                mission=mission,
                content=text,
                chunk_idx=idx,
                total_chunks=len(text_chunks),
                chunk_type="problem",
            )
            chunks.append(chunk)

        return chunks

    def _chunk_rubric(self, mission: ParsedMission) -> list[MissionChunk]:
        """채점 기준표를 청킹합니다."""
        if not mission.raw_text:
            return []

        # 메타데이터 헤더 추가
        header = f"[채점 기준표] {mission.course} - {mission.mission_name}"
        full_text = f"{header}\n\n{mission.raw_text}"

        # 분할
        text_chunks = self.text_splitter.split_text(full_text)

        # 청크 생성
        chunks: list[MissionChunk] = []
        for idx, text in enumerate(text_chunks):
            chunk = self._create_chunk(
                mission=mission,
                content=text,
                chunk_idx=idx,
                total_chunks=len(text_chunks),
                chunk_type="rubric",
            )
            chunks.append(chunk)

        return chunks

    def _remove_base64_images(self, text: str) -> str:
        """
        텍스트에서 base64 인코딩된 이미지 데이터를 제거합니다.

        제거 대상:
        - 마크다운 이미지: ![alt](data:image/png;base64,...)
        - HTML 이미지: <img src="data:image/png;base64,...">
        - 순수 base64 데이터 블록
        """
        result = text

        for pattern in self.BASE64_IMAGE_PATTERNS:
            result = re.sub(pattern, "[Image]", result, flags=re.DOTALL)

        return result

    def _extract_code_hints(self, code: str) -> str:
        """
        코드 셀에서 힌트(주석, TODO)만 추출합니다.

        정답 코드는 제외하고 힌트성 주석만 추출합니다.
        """
        hints: list[str] = []

        for line in code.split("\n"):
            line = line.strip()

            # TODO 주석
            if re.match(r"^#\s*TODO", line, re.IGNORECASE):
                hints.append(f"💡 힌트: {line[1:].strip()}")

            # 힌트/가이드 주석
            elif re.match(r"^#\s*(힌트|Hint|가이드|Guide)", line, re.IGNORECASE):
                hints.append(f"💡 {line[1:].strip()}")

            # 설명 주석 (긴 주석)
            elif line.startswith("#") and len(line) > 20:
                # 단순 코드 주석 제외 (import, 변수명 등)
                comment = line[1:].strip()
                if not re.match(r"^(import|from|def|class|\w+\s*=)", comment):
                    hints.append(f"📝 {comment}")

        return "\n".join(hints) if hints else ""

    def _create_header(self, mission: ParsedMission) -> str:
        """미션 메타데이터 헤더를 생성합니다."""
        parts = []

        if mission.course:
            parts.append(f"과목: {mission.course}")
        if mission.week:
            parts.append(f"주차: {mission.week}")
        if mission.mission_name:
            parts.append(f"미션: {mission.mission_name}")
        if mission.instructor:
            parts.append(f"마스터: {mission.instructor}")

        if parts:
            return f"[주간 미션] {' | '.join(parts)}"
        return "[주간 미션]"

    def _create_chunk(
        self,
        mission: ParsedMission,
        content: str,
        chunk_idx: int,
        total_chunks: int,
        chunk_type: str,
    ) -> MissionChunk:
        """청크 객체를 생성합니다."""
        chunk_id = self._generate_chunk_id(mission, chunk_idx, chunk_type)

        metadata = {
            "source_file": str(mission.file_path),
            "file_name": mission.file_path.name,
            "course": mission.course,
            "week": mission.week,
            "mission_name": mission.mission_name,
            "instructor": mission.instructor,
            "chunk_idx": chunk_idx,
            "total_chunks": total_chunks,
            "chunk_type": chunk_type,
            "doc_type": "weekly_mission",
        }

        # 버전 메타데이터 추가
        version_meta = create_chunk_version_metadata(
            source_file=mission.file_path,
            include_hash=True,
        )
        metadata.update(version_meta)

        return MissionChunk(
            id=chunk_id,
            content=content,
            metadata=metadata,
        )

    def _generate_chunk_id(
        self,
        mission: ParsedMission,
        chunk_idx: int,
        chunk_type: str,
    ) -> str:
        """청크 ID를 생성합니다."""
        course_slug = mission.course.lower().replace(" ", "_").replace("-", "_")
        if not course_slug:
            course_slug = "unknown"

        week_str = mission.week if mission.week else "w0"

        hash_input = f"{mission.file_path}_{chunk_idx}".encode()
        short_hash = hashlib.md5(hash_input).hexdigest()[:6]

        return f"mission_{course_slug}_{week_str}_{chunk_type}_c{chunk_idx:03d}_{short_hash}"


# =============================================================================
# CLI 테스트
# =============================================================================

if __name__ == "__main__":
    import sys
    from .mission_loader import MissionLoader

    if len(sys.argv) < 2:
        print("사용법: python -m mission.mission_chunker <directory>")
        sys.exit(1)

    directory = sys.argv[1]

    loader = MissionLoader(verbose=True)
    missions = loader.load_from_directory(directory)

    chunker = MissionChunker(chunk_size=1000, chunk_overlap=100)

    print(f"\n{'=' * 60}")
    print(f"📂 {len(missions)}개 미션 로드")

    all_chunks = chunker.chunk_missions(missions)
    print(f"🧩 {len(all_chunks)}개 청크 생성")
    print("=" * 60)

    # 미리보기
    for chunk in all_chunks[:3]:
        print(f"\n--- {chunk.id} ({chunk.token_estimate}t) ---")
        print(chunk.content[:400])
        if len(chunk.content) > 400:
            print("...")
