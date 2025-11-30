import json
from pathlib import Path
from typing import Any

try:
    import pymupdf4llm
except ImportError:
    print("Could not import pymupdf4llm, please install with `pip install pymupdf4llm`.")
    raise


def _convert_pymupdf_objects_to_serializable(obj: Any) -> Any:
    """
    PyMuPDF 객체를 JSON 직렬화 가능한 형태로 바꾸기 위해 재귀적으로 변환합니다.

    매개변수:
        obj: 변환할 객체 (dict, list, Rect 등)

    반환값:
        JSON 직렬화 가능한 객체
    """
    # Rect 객체 처리
    if hasattr(obj, "__class__") and obj.__class__.__name__ == "Rect":
        # Rect를 (x0, y0, x1, y1) 튜플로 변환
        return tuple(obj)

    # 딕셔너리 처리
    elif isinstance(obj, dict):
        return {key: _convert_pymupdf_objects_to_serializable(value) for key, value in obj.items()}

    # 리스트 처리
    elif isinstance(obj, list):
        return [_convert_pymupdf_objects_to_serializable(item) for item in obj]

    # 튜플 처리 (내부에 Rect가 있을 수 있음)
    elif isinstance(obj, tuple):
        return tuple(_convert_pymupdf_objects_to_serializable(item) for item in obj)

    # 기타 타입은 그대로 반환
    else:
        return obj

def parse_pdf(pdf_path: Path | str) -> dict:
    """PDF를 파싱하여 JSON 형식으로 반환합니다."""
    if isinstance(pdf_path, Path):
        pdf_name = pdf_path.name.replace('.pdf', '')
    else:
        pdf_name = Path(pdf_path).name.replace('.pdf', '')

    image_dir = Path(f"./images/{pdf_name}")
    image_dir.mkdir(parents=True, exist_ok=True)

    pymupdf4llm_result = pymupdf4llm.to_markdown(
        str(pdf_path),
        show_progress=True,
        # ==========================================
        # 이미지 관련 설정
        # ==========================================
        write_images=True,  # 이미지를 파일로 저장
        image_path=str(image_dir),  # 이미지 저장 경로
        image_format="png",  # PNG 포맷 사용
        image_size_limit=0.05,  # 최소 이미지 크기 (페이지의 5%)
        dpi=150,  # 이미지 해상도
        ignore_images=False,  # 이미지 포함
        # ==========================================
        # 텍스트 추출 품질 향상
        # ==========================================
        force_text=True,  # 이미지 배경이 있어도 텍스트 출력
        detect_bg_color=True,  # 배경색 감지로 정확도 향상
        ignore_alpha=False,  # 투명한 텍스트는 제외
        fontsize_limit=3,  # 최소 폰트 크기
        # ==========================================
        # 테이블 감지 설정
        # ==========================================
        table_strategy="lines_strict",  # 엄격한 테이블 감지 전략
        # ==========================================
        # 그래픽 및 코드 처리
        # ==========================================
        ignore_graphics=False,  # 벡터 그래픽 포함
        ignore_code=False,  # 코드 형식 유지
        # ==========================================
        # 페이지 및 레이아웃 설정
        # ==========================================
        page_chunks=True,  # 전체 문서 = JSON 으로 리턴
        page_separators=True,  # 페이지 구분자 사용
        margins=0,  # 여백 포함
        page_width=800,  # 페이지 너비 (default=612)
        # ==========================================
        # 추가 설정
        # ==========================================
        extract_words=False,  # 기본 텍스트 추출만 사용
        use_glyphs=False,  # Unicode 사용
        embed_images=False,  # base64 인코딩 사용 안함 (write_images와 배타적)
    )

    return pymupdf4llm_result

def save_json(pdf_name: str, path: Path | str, pymupdf4llm_result: dict, is_suffix: bool = True):
    if is_suffix:
        json_path = Path(path) / f"{pdf_name}_pymupdf4llm.json"
    else:
        json_path = Path(path) / f"{pdf_name}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(pymupdf4llm_result, f, indent=4, ensure_ascii=False)

def load_json(pdf_name: str, path: Path | str, is_suffix: bool = True) -> dict:
    if is_suffix:
        json_path = Path(path) / f"{pdf_name}_pymupdf4llm.json"
    else:
        json_path = Path(path) / f"{pdf_name}.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"JSON file not found: {json_path}")
