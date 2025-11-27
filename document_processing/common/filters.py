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

# [NEW] IP(Intellectual Property) 경고 패턴
IP_WARNING_PATTERNS = [
    r"본\s*자료는?\s*(내부|교육용)\s*(전용|목적)",
    r"외부\s*(유출|공유|배포)\s*(금지|불가)",
    r"(비공개|기밀|Confidential)",
    r"(교육생|수강생)\s*외\s*열람\s*금지",
    r"2차\s*(배포|가공|수정)\s*금지",
    r"무단\s*(복제|전재|배포)",
]

# [NEW] 이미지 플레이스홀더 패턴
IMAGE_PLACEHOLDER_PATTERNS = [
    r"\[이미지\]",
    r"\[그림\s*\d*\]",
    r"\[Figure\s*\d*\]",
    r"\[Image\s*\d*\]",
    r"\[사진\s*\d*\]",
    r"\[표\s*\d*\]",
    r"\[Table\s*\d*\]",
    r"\[도표\s*\d*\]",
    r"<image>",
    r"<그림>",
    r"\[화면\s*캡처\]",
    r"\[스크린샷\]",
]

# [NEW] 슬라이드 메타 정보 패턴 (강의 자료)
SLIDE_META_PATTERNS = [
    r"^\s*\d+\s*/\s*\d+\s*$",  # "15 / 30"
    r"^\s*Slide\s*\d+\s*$",
    r"^\s*슬라이드\s*\d+\s*$",
    r"네이버\s*부스트캠프",
    r"Naver\s*Boost\s*Camp",
    r"^(Day|Week)\s*\d+",
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

# [NEW] 진단/로깅 출력 패턴 (노트북 출력에서 제거)
DIAGNOSTIC_OUTPUT_PATTERNS = [
    r"^\s*\d+/\d+\s*\[=*>*\.*\]\s*-",  # Keras/TF progress bar: "1/10 [=====>....] - 2s"
    r"^Epoch\s+\d+/\d+",  # Epoch progress
    r"^\s*\d+it\s*\[\d+:\d+",  # tqdm progress
    r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s*[:\-]",  # Log levels
    r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}",  # Timestamps
    r"^(Downloading|Loading|Saving):\s*\d+%",  # Download progress
    r"^\s*loss:\s*[\d.]+",  # Training metrics
    r"^Step\s+\d+",  # Training steps
    r"^Iteration\s+\d+",
    r"^\s*\[\d+\]\s*loss:",  # LightGBM/XGBoost logs
    r"^\s*\[I\s+\d{4}",  # IPython/Jupyter internal logs
]

# [NEW] 불필요한 출력 패턴 (제거 대상)
UNNECESSARY_OUTPUT_PATTERNS = [
    r"^<[a-zA-Z_][a-zA-Z0-9_.]*\s+object\s+at\s+0x[0-9a-fA-F]+>$",  # Object repr
    r"^tensor\(\[[\d\s.,\-e]+\]\)$",  # PyTorch tensor short repr
    r"^array\(\[[\d\s.,\-e]+\]\)$",  # NumPy array short repr
    r"^\s*dtype\s*=",  # dtype info
    r"^<matplotlib\.",  # Matplotlib objects
    r"^Text\(0,\s*\d+,",  # Matplotlib text objects
    r"^<AxesSubplot:",  # Matplotlib axes
    r"^\s*Name:\s*\w+,\s*Length:",  # Pandas series info
    r"^\s*\.\.\.\s*$",  # Ellipsis only
]

# [NEW] 코드 셀 품질 평가를 위한 의미 있는 패턴
MEANINGFUL_CODE_PATTERNS = [
    r"^\s*def\s+\w+",  # Function definition
    r"^\s*class\s+\w+",  # Class definition
    r"^\s*@\w+",  # Decorators
    r"^\s*(if|for|while|with|try)\s+",  # Control flow
    r"\.\w+\(",  # Method calls
    r"=\s*\w+\(",  # Variable assignment with call
    r"return\s+",  # Return statements
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


def remove_ip_warnings(text: str) -> str:
    """[NEW] IP 경고 문구를 제거합니다."""
    result = text
    for pattern in IP_WARNING_PATTERNS:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE | re.MULTILINE)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def remove_image_placeholders(text: str) -> str:
    """[NEW] 이미지 플레이스홀더를 제거합니다."""
    result = text
    for pattern in IMAGE_PLACEHOLDER_PATTERNS:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def remove_slide_meta(text: str) -> str:
    """[NEW] 슬라이드 메타 정보를 제거합니다."""
    lines = text.split("\n")
    filtered_lines = []

    for line in lines:
        is_slide_meta = False
        for pattern in SLIDE_META_PATTERNS:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                is_slide_meta = True
                break

        if not is_slide_meta:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def clean_pdf_text(text: str) -> str:
    """
    [NEW] PDF 텍스트에 대한 통합 클린업을 수행합니다.

    다음 순서로 클린업:
    1. 저작권 고지 제거
    2. IP 경고 제거
    3. 이미지 플레이스홀더 제거
    4. 슬라이드 메타 정보 제거
    5. 헤더/푸터 제거

    Args:
        text: 클린업할 PDF 텍스트

    Returns:
        클린업된 텍스트
    """
    result = text

    # 1. 저작권 고지 제거
    result = remove_copyright_notices(result)

    # 2. IP 경고 제거
    result = remove_ip_warnings(result)

    # 3. 이미지 플레이스홀더 제거
    result = remove_image_placeholders(result)

    # 4. 슬라이드 메타 정보 제거
    result = remove_slide_meta(result)

    # 5. 헤더/푸터 제거
    result = remove_headers_footers(result)

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


def is_diagnostic_output(output: str) -> bool:
    """[NEW] 출력이 진단/로깅 출력인지 확인."""
    lines = output.strip().split("\n")

    # 모든 줄이 진단 패턴인지 확인
    diagnostic_lines = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue

        for pattern in DIAGNOSTIC_OUTPUT_PATTERNS:
            if re.match(pattern, line):
                diagnostic_lines += 1
                break

    # 50% 이상이 진단 출력이면 제거 대상
    total_lines = len([l for l in lines if l.strip()])
    return total_lines > 0 and diagnostic_lines / total_lines > 0.5


def is_unnecessary_output(output: str) -> bool:
    """[NEW] 출력이 불필요한 출력인지 확인."""
    text = output.strip()

    # 단일 줄 불필요 출력 체크
    for pattern in UNNECESSARY_OUTPUT_PATTERNS:
        if re.match(pattern, text):
            return True

    return False


def clean_notebook_output(output: str) -> str:
    """
    [NEW] 노트북 출력을 클린업합니다.

    진단 로그, 불필요한 객체 표현 등을 제거합니다.

    Args:
        output: 클린업할 노트북 출력

    Returns:
        클린업된 출력 (빈 문자열일 수 있음)
    """
    if not output or not output.strip():
        return ""

    # 불필요한 출력이면 빈 문자열 반환
    if is_unnecessary_output(output):
        return ""

    # 진단 출력이면 빈 문자열 반환
    if is_diagnostic_output(output):
        return ""

    # 줄 단위로 필터링
    lines = output.split("\n")
    filtered_lines = []

    for line in lines:
        # 진단 패턴에 매치되는 줄 제거
        is_diagnostic = False
        for pattern in DIAGNOSTIC_OUTPUT_PATTERNS:
            if re.match(pattern, line.strip()):
                is_diagnostic = True
                break

        # 불필요 패턴에 매치되는 줄 제거
        is_unnecessary = False
        for pattern in UNNECESSARY_OUTPUT_PATTERNS:
            if re.match(pattern, line.strip()):
                is_unnecessary = True
                break

        if not is_diagnostic and not is_unnecessary:
            filtered_lines.append(line)

    result = "\n".join(filtered_lines)

    # 연속된 빈 줄 정리
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def has_meaningful_code(code: str) -> bool:
    """
    [NEW] 코드에 의미 있는 로직이 포함되어 있는지 확인.

    함수 정의, 클래스 정의, 제어 흐름 등이 있으면 의미 있는 코드로 판단합니다.

    Args:
        code: 확인할 코드

    Returns:
        의미 있는 코드 여부
    """
    for pattern in MEANINGFUL_CODE_PATTERNS:
        if re.search(pattern, code, re.MULTILINE):
            return True

    return False


def should_keep_code_cell(code: str, has_output: bool = False) -> tuple[bool, str]:
    """
    [NEW] 코드 셀을 유지해야 하는지 판단합니다.

    Args:
        code: 코드 셀 내용
        has_output: 의미 있는 출력이 있는지 여부

    Returns:
        (should_keep: bool, reason: str)
    """
    # 빈 코드
    if not code or not code.strip():
        return False, "empty_code"

    # import만 있는 코드
    if is_import_only_code(code):
        return False, "import_only"

    # trivial 코드
    if is_trivial_code(code):
        return False, "trivial_code"

    # 의미 있는 코드가 있으면 유지
    if has_meaningful_code(code):
        return True, "meaningful_code"

    # 출력이 있으면 유지 (결과를 보여주는 코드)
    if has_output:
        return True, "has_output"

    # 그 외의 경우, 길이가 충분하면 유지
    if len(code.strip()) > 50:
        return True, "sufficient_length"

    return False, "low_value"


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
