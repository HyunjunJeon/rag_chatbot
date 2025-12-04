"""
강의 녹취록 청킹 모듈.

processed_combined 디렉토리의 JSON 파일(강의 녹취록)을
RAG 시스템을 위한 청크로 분할합니다.

파일명이 강의명이므로 메타데이터로 중요하게 활용합니다.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..common.versioning import create_chunk_version_metadata


@dataclass
class LectureTranscriptChunk:
    """
    강의 녹취록에서 추출한 청크.

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
        """토큰 수 추정 (한글 기준)."""
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


@dataclass
class ParsedTranscript:
    """
    파싱된 강의 녹취록.

    Attributes:
        file_path: 원본 파일 경로
        lecture_name: 강의명 (파일명에서 추출)
        source_file: 원본 소스 파일명
        text: 전체 녹취록 텍스트
        course: 과목명 (추출)
        lecture_num: 강의 번호 (추출)
        lecture_title: 강의 제목 (추출)
    """

    file_path: Path
    lecture_name: str
    source_file: str
    text: str
    course: str = ""
    lecture_num: str = ""
    lecture_title: str = ""


class LectureTranscriptChunker:
    """
    강의 녹취록을 RAG용 청크로 분할하는 클래스.

    청킹 전략:
    1. RecursiveCharacterTextSplitter로 의미 단위 분할
    2. 파일명에서 과목/강의번호/제목 메타데이터 추출
    3. 컨텍스트 헤더 추가 (과목, 강의명)

    예시:
        ```python
        chunker = LectureTranscriptChunker(chunk_size=1000)
        chunks = chunker.process_file(Path("(1강) CV.json"))

        for chunk in chunks:
            print(f"[{chunk.id}] {chunk.content[:100]}...")
        ```
    """

    # 한국어에 적합한 구분자 (문장 단위 분할 우선)
    DEFAULT_SEPARATORS = [
        "\n\n",  # 문단
        "\n",  # 줄바꿈
        "다. ",  # 한국어 문장 종결
        ". ",  # 영어 문장 종결
        "요. ",  # 존댓말 종결
        "죠. ",  # 구어체 종결
        ", ",  # 쉼표
        " ",  # 공백
        "",  # 글자
    ]

    # 과목명 패턴들 (파일명에서 추출)
    COURSE_PATTERNS = [
        # [과목] 형식: [RecSys 이론], [AI Math], [MRC]
        r"^\[([^\]]+)\]",
        # (과목) 형식 (강의번호 제외)
        r"^\((?!\d+강)([^)]+)\)",
    ]

    # 강의 번호 패턴들
    LECTURE_NUM_PATTERNS = [
        r"\((\d+강)\)",  # (1강)
        r"\((\d+-\d+강)\)",  # (8-1강)
        r"(\d+강)[\s_]",  # 1강 또는 1강_
        r"_(\d+강)_",  # _10강_
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        min_chunk_size: int = 100,
    ) -> None:
        """
        LectureTranscriptChunker 초기화.

        Args:
            chunk_size: 청크 최대 토큰 수 (기본 1000)
            chunk_overlap: 청크 간 오버랩 토큰 수 (기본 150)
            min_chunk_size: 최소 청크 크기 (이보다 작으면 이전 청크에 병합)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # 토큰 → 문자 변환 (한글 기준 약 2자/토큰)
        char_size = int(chunk_size * 2)
        char_overlap = int(chunk_overlap * 2)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=char_size,
            chunk_overlap=char_overlap,
            separators=self.DEFAULT_SEPARATORS,
            length_function=len,
        )

    def process_file(self, file_path: Path) -> list[LectureTranscriptChunk]:
        """
        단일 JSON 파일을 처리하여 청크로 분할합니다.

        Args:
            file_path: JSON 파일 경로

        Returns:
            LectureTranscriptChunk 리스트
        """
        # JSON 파싱
        transcript = self._load_transcript(file_path)
        if not transcript or not transcript.text.strip():
            return []

        # 청킹
        chunks = self._chunk_transcript(transcript)

        return chunks

    def process_directory(
        self,
        directory: Path,
        verbose: bool = True,
    ) -> list[LectureTranscriptChunk]:
        """
        디렉토리 내 모든 JSON 파일을 처리합니다.

        Args:
            directory: 처리할 디렉토리
            verbose: 진행 상황 출력 여부

        Returns:
            모든 LectureTranscriptChunk 리스트
        """
        json_files = sorted(directory.glob("*.json"))

        if verbose:
            print(f"\n📁 {directory.name}: {len(json_files)}개 파일")

        all_chunks: list[LectureTranscriptChunk] = []
        skipped = 0

        for json_file in json_files:
            chunks = self.process_file(json_file)
            if chunks:
                all_chunks.extend(chunks)
                if verbose:
                    print(f"   ✓ {json_file.stem[:50]}... → {len(chunks)}청크")
            else:
                skipped += 1
                if verbose:
                    print(f"   ✗ {json_file.stem[:50]}... (빈 파일)")

        if verbose:
            print(f"\n   총: {len(all_chunks)}청크 ({skipped}개 스킵)")

        return all_chunks

    def _load_transcript(self, file_path: Path) -> ParsedTranscript | None:
        """JSON 파일을 로드하고 파싱합니다."""
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            lecture_name = data.get("lecture_name", file_path.stem)
            source_file = data.get("source_file", "")
            text = data.get("text", "")

            # 텍스트가 너무 짧으면 스킵
            if len(text) < 50:
                return None

            # 메타데이터 추출
            course, lecture_num, lecture_title = self._extract_metadata(lecture_name)

            return ParsedTranscript(
                file_path=file_path,
                lecture_name=lecture_name,
                source_file=source_file,
                text=text,
                course=course,
                lecture_num=lecture_num,
                lecture_title=lecture_title,
            )

        except Exception as e:
            print(f"   ⚠️ 파싱 오류 {file_path.name}: {e}")
            return None

    def _extract_metadata(self, lecture_name: str) -> tuple[str, str, str]:
        """
        강의명에서 과목, 강의번호, 제목을 추출합니다.

        예시:
        - "[RecSys 이론] (2강) 추천 시스템 Basic 2"
          → ("RecSys 이론", "2강", "추천 시스템 Basic 2")
        - "(1강) Introduction to Computer Vision"
          → ("Computer Vision", "1강", "Introduction to Computer Vision")
        - "GenAI CV part1"
          → ("GenAI CV", "", "GenAI CV part1")

        Returns:
            (과목명, 강의번호, 강의제목) 튜플
        """
        course = ""
        lecture_num = ""
        lecture_title = lecture_name

        # 1. [과목] 패턴 추출
        for pattern in self.COURSE_PATTERNS:
            match = re.search(pattern, lecture_name)
            if match:
                course = match.group(1).strip()
                # 과목명 제거한 나머지가 제목
                lecture_title = lecture_name[match.end() :].strip()
                break

        # 2. 강의 번호 추출
        for pattern in self.LECTURE_NUM_PATTERNS:
            match = re.search(pattern, lecture_name)
            if match:
                lecture_num = match.group(1)
                # 강의번호 제거
                lecture_title = re.sub(pattern, "", lecture_title).strip()
                break

        # 3. 과목이 없으면 강의 제목에서 추론
        if not course:
            course = self._infer_course(lecture_name)

        # 제목 정리 (앞뒤 특수문자 제거)
        lecture_title = re.sub(r"^[\s\-_]+|[\s\-_]+$", "", lecture_title)

        return course, lecture_num, lecture_title

    def _infer_course(self, lecture_name: str) -> str:
        """
        파일명에서 과목을 추론합니다.

        키워드 기반 매핑:
        """
        lecture_lower = lecture_name.lower()

        # 키워드 → 과목 매핑
        course_keywords = {
            "computer vision": "Computer Vision",
            "cv": "Computer Vision",
            "segmentation": "Computer Vision",
            "detection": "Computer Vision",
            "3d understanding": "Computer Vision",
            "3d human": "Computer Vision",
            "generative model": "Generative Model",
            "genai": "Generative AI",
            "diffusion": "Generative Model",
            "vae": "Generative Model",
            "nlp": "NLP",
            "transformer": "NLP",
            "bert": "NLP",
            "language model": "NLP",
            "tokenization": "NLP",
            "word embedding": "NLP",
            "seq2seq": "NLP",
            "attention": "NLP",
            "mrc": "MRC",
            "passage retrieval": "MRC",
            "recsys": "RecSys",
            "recommender": "RecSys",
            "recommendation": "RecSys",
            "collaborative filtering": "RecSys",
            "pytorch": "PyTorch",
            "tensor": "PyTorch",
            "neural network": "Deep Learning Basic",
            "back propagation": "Deep Learning Basic",
            "linear regression": "Machine Learning",
            "classification": "Machine Learning",
            "ai math": "AI Math",
            "선형대수": "AI Math",
            "벡터": "AI Math",
            "행렬": "AI Math",
            "경사하강법": "AI Math",
            "확률론": "AI Math",
            "통계학": "AI Math",
            "ml lifecycle": "ML Engineering",
            "streamlit": "ML Engineering",
            "데이터 증강": "Data Engineering",
            "전처리": "Data Engineering",
            "경진대회": "Competition",
            "object detection": "Object Detection",
        }

        for keyword, course in course_keywords.items():
            if keyword in lecture_lower:
                return course

        # 파일명 앞부분에서 추출 시도 (예: "GenAI CV part1" → "GenAI CV")
        # part/Part로 분리
        part_match = re.split(r"\s+part\s*\d*", lecture_name, flags=re.IGNORECASE)
        if len(part_match) > 1 and part_match[0].strip():
            return part_match[0].strip()

        return "기타"

    def _chunk_transcript(self, transcript: ParsedTranscript) -> list[LectureTranscriptChunk]:
        """녹취록을 청크로 분할합니다."""
        # 텍스트 전처리
        text = self._preprocess_text(transcript.text)

        # 컨텍스트 헤더 생성
        header = self._create_header(transcript)

        # RecursiveCharacterTextSplitter로 분할
        text_chunks = self.text_splitter.split_text(text)

        # 청크 객체로 변환
        chunks: list[LectureTranscriptChunk] = []
        for idx, chunk_text in enumerate(text_chunks):
            # 첫 청크에는 헤더 추가
            if idx == 0:
                content = f"{header}\n\n{chunk_text}"
            else:
                # 이후 청크에는 간략한 참조만
                short_header = f"[{transcript.course}] {transcript.lecture_name}"
                content = f"{short_header}\n\n{chunk_text}"

            chunk = self._create_chunk(
                transcript=transcript,
                content=content,
                chunk_idx=idx,
                total_chunks=len(text_chunks),
            )
            chunks.append(chunk)

        # 작은 청크 병합
        chunks = self._merge_small_chunks(chunks)

        return chunks

    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리."""
        # 연속 공백 정리
        text = re.sub(r" {2,}", " ", text)
        # 연속 줄바꿈 정리 (3개 이상 → 2개)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 앞뒤 공백 제거
        text = text.strip()
        return text

    def _create_header(self, transcript: ParsedTranscript) -> str:
        """컨텍스트 헤더를 생성합니다."""
        parts = []

        if transcript.course:
            parts.append(f"과목: {transcript.course}")
        if transcript.lecture_num:
            parts.append(f"강의: {transcript.lecture_num}")
        if transcript.lecture_title:
            parts.append(f"제목: {transcript.lecture_title}")

        if parts:
            return f"[강의 녹취록] {' | '.join(parts)}"
        return f"[강의 녹취록] {transcript.lecture_name}"

    def _create_chunk(
        self,
        transcript: ParsedTranscript,
        content: str,
        chunk_idx: int,
        total_chunks: int,
    ) -> LectureTranscriptChunk:
        """청크 객체를 생성합니다."""
        chunk_id = self._generate_chunk_id(transcript, chunk_idx)

        metadata = {
            "doc_type": "lecture_transcript",
            "source_file": str(transcript.file_path.name),
            "lecture_name": transcript.lecture_name,
            "course": transcript.course,
            "lecture_num": transcript.lecture_num,
            "lecture_title": transcript.lecture_title,
            "chunk_idx": chunk_idx,
            "total_chunks": total_chunks,
        }

        # 버전 메타데이터 추가
        version_meta = create_chunk_version_metadata(
            source_file=transcript.file_path,
            include_hash=True,
        )
        metadata.update(version_meta)

        return LectureTranscriptChunk(
            id=chunk_id,
            content=content,
            metadata=metadata,
        )

    def _generate_chunk_id(self, transcript: ParsedTranscript, chunk_idx: int) -> str:
        """청크 고유 ID 생성."""
        # 과목 슬러그
        course_slug = self._slugify(transcript.course) if transcript.course else "etc"

        # 강의 슬러그
        lecture_slug = self._slugify(transcript.lecture_name)[:30]

        # 파일 해시
        hash_input = f"{transcript.file_path}_{chunk_idx}".encode()
        short_hash = hashlib.md5(hash_input).hexdigest()[:6]

        return f"transcript_{course_slug}_{lecture_slug}_c{chunk_idx:03d}_{short_hash}"

    def _slugify(self, text: str) -> str:
        """텍스트를 ID 친화적으로 변환."""
        # 한글 유지, 특수문자 제거
        text = re.sub(r"[^\w가-힣\s-]", "", text)
        text = re.sub(r"[-\s]+", "_", text)
        return text.lower().strip("_")

    def _merge_small_chunks(
        self, chunks: list[LectureTranscriptChunk]
    ) -> list[LectureTranscriptChunk]:
        """작은 청크를 이전 청크에 병합합니다."""
        if not chunks or len(chunks) < 2:
            return chunks

        merged: list[LectureTranscriptChunk] = []
        min_chars = self.min_chunk_size * 2  # 토큰 → 문자

        for chunk in chunks:
            if not merged:
                merged.append(chunk)
                continue

            # 현재 청크가 너무 작으면 이전 청크에 병합
            if chunk.char_count < min_chars:
                prev = merged[-1]
                new_content = f"{prev.content}\n\n{chunk.content}"

                # 병합 후 크기 체크 (너무 크지 않으면 병합)
                if len(new_content) <= self.chunk_size * 2 * 1.3:
                    merged[-1] = LectureTranscriptChunk(
                        id=prev.id,
                        content=new_content,
                        metadata=prev.metadata,
                    )
                    continue

            merged.append(chunk)

        return merged


# =============================================================================
# CLI 테스트
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python -m lecture_transcript.lecture_transcript_chunker <directory>")
        sys.exit(1)

    directory = Path(sys.argv[1])
    if not directory.exists():
        print(f"디렉토리가 존재하지 않습니다: {directory}")
        sys.exit(1)

    chunker = LectureTranscriptChunker(chunk_size=1000, chunk_overlap=150)

    print("=" * 70)
    print("🎙️  강의 녹취록 청킹")
    print("=" * 70)

    all_chunks = chunker.process_directory(directory, verbose=True)

    print("\n" + "=" * 70)
    print(f"✅ 총 {len(all_chunks)}개 청크 생성")

    # 샘플 출력
    if all_chunks:
        print("\n📄 샘플 청크:")
        for chunk in all_chunks[:2]:
            print(f"\n--- {chunk.id} ---")
            print(f"과목: {chunk.metadata.get('course')}")
            print(f"강의: {chunk.metadata.get('lecture_name')}")
            print(f"토큰: ~{chunk.token_estimate}")
            print(f"내용: {chunk.content[:300]}...")
