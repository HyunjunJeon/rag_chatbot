"""
공통 상수 정의 모듈

이 모듈은 설정 파일들에서 공통으로 사용하는 상수를 정의합니다.
주로 프로젝트 루트 경로를 계산하여 .env 파일 위치를 찾는 데 사용됩니다.
"""

from pathlib import Path


def _find_project_root() -> Path:
    """
    프로젝트 루트 디렉토리를 찾습니다.

    현재 파일에서 시작하여 상위 디렉토리로 올라가면서
    프로젝트 루트를 나타내는 마커 파일을 찾습니다.

    마커 파일 우선순위:
    1. pyproject.toml (Python 프로젝트 표준)
    2. .git (Git 저장소 루트)
    3. setup.py (레거시 Python 프로젝트)

    반환값:
        Path: 프로젝트 루트 디렉토리 경로

    예외:
        RuntimeError: 프로젝트 루트를 찾지 못한 경우
    """
    # 마커 파일들 (우선순위 순서)
    markers = ["pyproject.toml", ".git", "setup.py"]

    # 현재 파일의 디렉토리에서 시작
    current_path = Path(__file__).resolve().parent

    # 루트 디렉토리에 도달할 때까지 상위로 탐색
    for parent in [current_path, *current_path.parents]:
        for marker in markers:
            if (parent / marker).exists():
                return parent

    # 마커를 찾지 못한 경우 에러 (fallback으로 4단계 위 사용)
    # 이는 개발 중 마커 파일이 없을 때를 위한 안전장치
    fallback = Path(__file__).resolve().parent.parent.parent.parent.parent
    return fallback


# 프로젝트 루트 경로
PROJECT_ROOT = _find_project_root()

__all__ = ["PROJECT_ROOT"]
