"""
Embedding 모듈

이 모듈은 langchain_naver의 ClovaXEmbeddings를 사용하여 임베딩 인스턴스를 생성합니다.

사용 예:
    from naver_connect_chatbot.config.embedding import get_embeddings
    
    # 기본 설정으로 임베딩 생성
    embeddings = get_embeddings()
    
    # 단일 쿼리 임베딩
    vector = embeddings.embed_query("안녕하세요")
    
    # 다중 문서 임베딩
    vectors = embeddings.embed_documents(["문서1", "문서2"])
"""

from typing import TYPE_CHECKING

from langchain_naver import ClovaXEmbeddings

if TYPE_CHECKING:
    from naver_connect_chatbot.config.settings.main import Settings


def get_embeddings(settings_obj: "Settings | None" = None) -> ClovaXEmbeddings:
    """
    ClovaXEmbeddings 인스턴스를 생성합니다.
    
    이 팩토리 함수는 설정에서 읽은 값으로 ClovaXEmbeddings를 초기화합니다.
    langchain_naver의 ClovaXEmbeddings는 CLOVASTUDIO_API_KEY 환경변수를
    자동으로 로드하지만, 명시적으로 전달하여 설정 통합을 유지합니다.
    
    매개변수:
        settings_obj: Settings 인스턴스 (None이면 전역 settings 사용)
    
    반환값:
        ClovaXEmbeddings 인스턴스
    
    예외:
        ValueError: API 키가 설정되지 않은 경우
    
    예시:
        >>> from naver_connect_chatbot.config.embedding import get_embeddings
        >>> 
        >>> # 기본 설정으로 임베딩 생성
        >>> embeddings = get_embeddings()
        >>> 
        >>> # 쿼리 임베딩
        >>> vector = embeddings.embed_query("GPU 메모리 부족 해결 방법")
        >>> print(len(vector))  # 1024 (bge-m3)
    """
    # 순환 import 방지를 위해 여기서 import
    from naver_connect_chatbot.config.settings.main import settings
    
    if settings_obj is None:
        settings_obj = settings
    
    config = settings_obj.clova_embeddings
    
    # API 키 검증
    if not config.api_key:
        raise ValueError(
            "CLOVASTUDIO_API_KEY가 설정되지 않았습니다. "
            ".env 파일에 CLOVASTUDIO_API_KEY를 설정하세요."
        )
    
    return ClovaXEmbeddings(
        model=config.embeddings_model,
        api_key=config.api_key.get_secret_value(),
    )


__all__ = [
    "get_embeddings",
]
