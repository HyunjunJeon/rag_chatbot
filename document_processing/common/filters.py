"""
청크 품질 필터링 유틸리티.

노이즈가 많은 콘텐츠를 제거하여 RAG 품질을 향상시킵니다.
"""

import re
from dataclasses import dataclass


# =============================================================================
# 필터링 패턴 정의
# =============================================================================

# 저작권 고지 패턴
COPYRIGHT_PATTERNS = [
    r"Copyright\s*©?\s*\d{4}",
    r"All\s+rights?\s+reserved\.?",
    r"무단\s*전재\s*및\s*재배포\s*금지",
    r"저작권\s*[:\s]\s*네이버",
    r"ⓒ\s*\d{4}",
    r"본\s+자료는?\s+.*\s+저작물",
]

# 목차 관련 패턴
TOC_PATTERNS = [
    r"^목\s*차\s*$",
    r"^Table\s+of\s+Contents?\s*$",
    r"^Contents?\s*$",
    r"^\d+\.\s+.+\s*\.{3,}\s*\d+$",  # "1. Introduction .......... 3"
]

# 헤더/푸터 패턴 (페이지 번호 등)
HEADER_FOOTER_PATTERNS = [
    r"^\s*-?\s*\d+\s*-?\s*$",  # 페이지 번호만
    r"^\s*page\s+\d+\s*$",
    r"^\s*\d+\s*/\s*\d+\s*$",  # "3 / 10" 형식
]

# import만 있는 코드 패턴
IMPORT_ONLY_PATTERNS = [
    r"^\s*(import\s+\w+|from\s+\w+\s+import\s+.+)\s*$",
]

# 단순 출력/확인 코드 패턴
TRIVIAL_CODE_PATTERNS = [
    r"^\s*print\s*\(\s*['\"].*['\"]\s*\)\s*$",  # print("hello")
    r"^\s*#\s*%%\s*$",  # Jupyter cell magic만
    r"^\s*pass\s*$",
]


# =============================================================================
# 필터링 함수
# =============================================================================


def contains_copyright(text: str) -> bool:
    """텍스트에 저작권 고지가 포함되어 있는지 확인."""
    for pattern in COPYRIGHT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return True
    return False


def remove_copyright_notices(text: str) -> str:
    """저작권 고지를 제거합니다."""
    result = text
    for pattern in COPYRIGHT_PATTERNS:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE | re.MULTILINE)
    # 연속된 빈 줄 정리
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def is_toc_page(text: str) -> bool:
    """목차 페이지인지 확인."""
    lines = text.strip().split("\n")

    # 첫 몇 줄에 목차 표시가 있는지 확인
    for line in lines[:5]:
        for pattern in TOC_PATTERNS:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                return True

    # 목차 형식의 줄이 많은지 확인 (점선 + 페이지 번호)
    toc_line_count = sum(1 for line in lines if re.match(r".+\.{3,}\s*\d+$", line.strip()))

    return toc_line_count > len(lines) * 0.5  # 50% 이상이면 목차


def remove_headers_footers(text: str) -> str:
    """헤더/푸터(페이지 번호 등)를 제거합니다."""
    lines = text.split("\n")
    filtered_lines = []

    for line in lines:
        is_header_footer = False
        for pattern in HEADER_FOOTER_PATTERNS:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                is_header_footer = True
                break

        if not is_header_footer:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def is_import_only_code(code: str) -> bool:
    """코드가 import 문만 포함하는지 확인."""
    lines = [line.strip() for line in code.strip().split("\n") if line.strip()]

    if not lines:
        return True

    # 모든 줄이 import 또는 빈 줄/주석인지 확인
    for line in lines:
        if not line or line.startswith("#"):
            continue

        is_import = any(re.match(pattern, line) for pattern in IMPORT_ONLY_PATTERNS)

        if not is_import:
            return False

    return True


def is_trivial_code(code: str) -> bool:
    """코드가 의미 없는 단순 코드인지 확인."""
    lines = [line.strip() for line in code.strip().split("\n") if line.strip()]

    if not lines:
        return True

    # 모든 줄이 trivial 패턴에 매치되는지 확인
    for line in lines:
        if not line or line.startswith("#"):
            continue

        is_trivial = any(re.match(pattern, line) for pattern in TRIVIAL_CODE_PATTERNS)

        if not is_trivial:
            return False

    return True


def estimate_content_quality(text: str) -> float:
    """
    콘텐츠 품질 점수를 추정합니다 (0.0 ~ 1.0).

    낮은 점수 요인:
    - 너무 짧음
    - 특수문자/숫자 비율이 높음
    - 반복 패턴이 많음
    """
    if not text or len(text.strip()) < 20:
        return 0.0

    score = 1.0

    # 길이 기반 점수
    text_len = len(text.strip())
    if text_len < 50:
        score *= 0.5
    elif text_len < 100:
        score *= 0.7

    # 특수문자 비율
    special_chars = len(re.findall(r"[^\w\s가-힣]", text))
    special_ratio = special_chars / text_len
    if special_ratio > 0.3:
        score *= 0.5

    # 숫자만 있는 줄 비율
    lines = text.strip().split("\n")
    number_only_lines = sum(1 for line in lines if re.match(r"^\s*[\d\s.,]+\s*$", line))
    if lines and number_only_lines / len(lines) > 0.5:
        score *= 0.5

    return min(max(score, 0.0), 1.0)


# =============================================================================
# 통합 필터 클래스
# =============================================================================


@dataclass
class FilterResult:
    """필터링 결과."""

    passed: bool
    reason: str = ""
    cleaned_text: str = ""


class ContentFilter:
    """콘텐츠 필터."""

    def __init__(
        self,
        remove_copyright: bool = True,
        remove_toc: bool = True,
        remove_headers_footers: bool = True,
        remove_import_only: bool = True,
        min_quality_score: float = 0.3,
    ):
        self.remove_copyright = remove_copyright
        self.remove_toc = remove_toc
        self.remove_headers_footers_flag = remove_headers_footers
        self.remove_import_only = remove_import_only
        self.min_quality_score = min_quality_score

        # 통계
        self.stats = {
            "total": 0,
            "filtered_copyright": 0,
            "filtered_toc": 0,
            "filtered_import_only": 0,
            "filtered_low_quality": 0,
            "passed": 0,
        }

    def filter_text(self, text: str, is_code: bool = False) -> FilterResult:
        """
        텍스트를 필터링합니다.

        Args:
            text: 필터링할 텍스트
            is_code: 코드 여부

        Returns:
            FilterResult 객체
        """
        self.stats["total"] += 1

        # 코드 필터링
        if is_code and self.remove_import_only:
            if is_import_only_code(text):
                self.stats["filtered_import_only"] += 1
                return FilterResult(passed=False, reason="import_only")

        # 목차 필터링
        if self.remove_toc and is_toc_page(text):
            self.stats["filtered_toc"] += 1
            return FilterResult(passed=False, reason="toc_page")

        # 텍스트 정제
        cleaned = text

        # 저작권 고지 제거
        if self.remove_copyright:
            if contains_copyright(cleaned):
                self.stats["filtered_copyright"] += 1
            cleaned = remove_copyright_notices(cleaned)

        # 헤더/푸터 제거
        if self.remove_headers_footers_flag:
            cleaned = remove_headers_footers(cleaned)

        # 품질 점수 확인
        quality = estimate_content_quality(cleaned)
        if quality < self.min_quality_score:
            self.stats["filtered_low_quality"] += 1
            return FilterResult(passed=False, reason="low_quality", cleaned_text=cleaned)

        self.stats["passed"] += 1
        return FilterResult(passed=True, cleaned_text=cleaned)

    def print_stats(self) -> None:
        """필터링 통계를 출력합니다."""
        print(f"\n📊 필터링 통계:")
        print(f"   전체: {self.stats['total']}개")
        print(f"   저작권 제거: {self.stats['filtered_copyright']}개")
        print(f"   목차 제외: {self.stats['filtered_toc']}개")
        print(f"   import만 제외: {self.stats['filtered_import_only']}개")
        print(f"   저품질 제외: {self.stats['filtered_low_quality']}개")
        print(f"   → 통과: {self.stats['passed']}개")
