"""
Clova X LLM, Embeddings, Reranker 통합 테스트

이 파일은 langchain_naver 패키지와 Clova Studio Reranker의 기본 사용법을 보여줍니다.

실행 전 준비사항:
1. .env 파일에 CLOVASTUDIO_API_KEY 설정
2. .env 파일에 CLOVASTUDIO_RERANKER_ENDPOINT 설정 (선택사항)

실행 방법:
    uv run python tests/test_clova_model.py
"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_naver import ChatClovaX, ClovaXEmbeddings

from naver_connect_chatbot.config.settings.clova import ClovaStudioRerankerSettings
from naver_connect_chatbot.rag.rerank import ClovaStudioReranker

# 환경변수 로드
load_dotenv()

print("=" * 80)
print("Clova X 통합 테스트 시작")
print("=" * 80)

# ============================================================================
# 1. ChatClovaX (LLM) 테스트
# ============================================================================
print("\n[1] ChatClovaX (LLM) 테스트")
print("-" * 80)

chat = ChatClovaX(
    model="HCX-007",
    thinking={
        "effort": "low"  # 'none', 'low', 'medium', 'high'
    },
)

# 간단한 질문 테스트 (주석 처리)
# response = chat.invoke("Hello, how are you?")
# print(f"응답: {response.content}")
print("✓ ChatClovaX 초기화 완료")
print(f"  - 타입: {type(chat).__name__}")

# ============================================================================
# 2. ClovaXEmbeddings 테스트
# ============================================================================
print("\n[2] ClovaXEmbeddings 테스트")
print("-" * 80)

embeddings = ClovaXEmbeddings(
    model="bge-m3",  # 기본값: bge-m3
)

query = "CLOVA Studio는 HyperCLOVA X 모델을 활용하여 AI 서비스를 손쉽게 만들 수 있는 개발 도구입니다."
single_vector = embeddings.embed_query(query)

print("✓ ClovaXEmbeddings 실행 완료")
print(f"  - 타입: {type(embeddings).__name__}")
print(f"  - 쿼리 길이: {len(query)} 글자")
print(f"  - 벡터 차원: {len(single_vector)} (예상: 1024)")

# ============================================================================
# 3. ClovaStudioReranker 테스트
# ============================================================================
print("\n[3] ClovaStudioReranker 테스트")
print("-" * 80)

# 테스트용 문서 생성
test_documents = [
    Document(
        page_content="Python은 1991년 귀도 반 로섬이 개발한 인터프리터 프로그래밍 언어입니다.",
        metadata={"source": "python_intro.txt", "page": 1}
    ),
    Document(
        page_content="JavaScript는 웹 브라우저에서 동작하는 스크립트 언어로, 1995년에 만들어졌습니다.",
        metadata={"source": "javascript_intro.txt", "page": 1}
    ),
    Document(
        page_content="Java는 객체지향 프로그래밍 언어로, 1995년 썬 마이크로시스템즈에서 개발했습니다.",
        metadata={"source": "java_intro.txt", "page": 1}
    ),
    Document(
        page_content="CLOVA Studio는 네이버의 초거대 AI 모델을 활용한 개발 플랫폼입니다.",
        metadata={"source": "clova_studio.txt", "page": 1}
    ),
    Document(
        page_content="머신러닝은 인공지능의 한 분야로, 데이터로부터 패턴을 학습합니다.",
        metadata={"source": "ml_intro.txt", "page": 1}
    ),
]

# Reranker Settings 직접 로드 (전체 Settings 로드 우회)
try:
    reranker_settings = ClovaStudioRerankerSettings()
    reranker = ClovaStudioReranker.from_settings(reranker_settings)
    print("✓ ClovaStudioReranker 초기화 완료")
    print(f"  - 엔드포인트: {reranker_settings.endpoint}")
    print(f"  - 타임아웃: {reranker_settings.request_timeout}초")
    print(f"  - Max Tokens: {reranker_settings.max_tokens}")
    
    # 재정렬 테스트 쿼리
    rerank_query = "CLOVA Studio에 대해 설명해주세요"
    print(f"\n  질문: {rerank_query}")
    print(f"  입력 문서 수: {len(test_documents)}")
    
    # Reranking 실행
    reranked_docs = reranker.rerank(
        query=rerank_query,
        documents=test_documents,
    )
    
    print(f"\n  재정렬 결과 (상위 {len(reranked_docs)}개):")
    for i, doc in enumerate(reranked_docs, 1):
        score = doc.metadata.get("rerank_score", 0.0)
        print(f"    {i}. [점수: {score:.4f}] {doc.page_content[:50]}...")


except Exception as e:
    print(f"⚠️  Reranker 초기화 실패: {e}")
    print("   .env 파일의 CLOVASTUDIO_RERANKER_ENDPOINT를 확인하세요.")

# ============================================================================
# 테스트 완료
# ============================================================================
print("\n" + "=" * 80)
print("✅ 모든 테스트 완료")
print("=" * 80)