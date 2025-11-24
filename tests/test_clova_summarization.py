"""
Clova Studio Summarization API 통합 테스트

이 파일은 Clova Studio의 요약 API 기본 사용법을 보여줍니다.

실행 전 준비사항:
1. .env 파일에 CLOVASTUDIO_API_KEY 설정
2. .env 파일에 CLOVASTUDIO_SUMMARIZATION_ENDPOINT 설정 (선택사항)

실행 방법:
    uv run python tests/test_clova_summarization.py
"""

from dotenv import load_dotenv

from naver_connect_chatbot.config.settings.clova import ClovaStudioSummarizationSettings
from naver_connect_chatbot.rag.summarization import ClovaStudioSummarizer

# 환경변수 로드
load_dotenv()

print("=" * 80)
print("Clova Studio Summarization API 테스트 시작")
print("=" * 80)

# ============================================================================
# 1. ClovaStudioSummarizer 초기화 테스트
# ============================================================================
print("\n[1] ClovaStudioSummarizer 초기화")
print("-" * 80)

try:
    summarization_settings = ClovaStudioSummarizationSettings()
    summarizer = ClovaStudioSummarizer.from_settings(summarization_settings)
    print("✓ ClovaStudioSummarizer 초기화 완료")
    print(f"  - 엔드포인트: {summarization_settings.endpoint}")
    print(f"  - 타임아웃: {summarization_settings.request_timeout}초")
    print(f"  - Auto Sentence Splitter: {summarization_settings.auto_sentence_splitter}")
    print(f"  - Seg Max Size: {summarization_settings.seg_max_size}")
    print(f"  - Include AI Filters: {summarization_settings.include_ai_filters}")
except Exception as e:
    print(f"⚠️  Summarizer 초기화 실패: {e}")
    print("   .env 파일의 CLOVASTUDIO_API_KEY를 확인하세요.")
    exit(1)

# ============================================================================
# 2. 단일 텍스트 요약 테스트
# ============================================================================
print("\n[2] 단일 텍스트 요약")
print("-" * 80)

# 테스트용 긴 텍스트
test_text = """CLOVA Studio가 제공하는 다양한 기능은 다음과 같습니다. - 문장 생성: 몇 가지 키워드만 입력하면 해당 키워드를 기반으로 시나리오 창작, 자기소개서 작성, 이메일 생성, 마케팅 문구 창작 등 다양한 분야의 문장 생성. - 요약: 줄거리, 보고서, 이메일과 같이 긴 글에서 주요 요소를 파악하여 설정한 요약문 길이에 따라 글 요약. - 분류: 문장의 유형, 문서 색인, 감정, 의도와 같은 특징을 분류하거나 문단에서 주요 키워드 추출 가능. - 대화: 예제 입력을 통해 고유의 페르소나를 가진 AI를 생성하여 지식백과형 챗봇, 커스텀 챗봇 등 대화 인터페이스 제작 가능. - 문장 변환: 입력한 예제와 유사한 스타일의 문장으로 문장 형태 변환 가능 AI Filter: 민감하거나 안전하지 않은 결과물이 생성되는 것을 감지하여 알리는 AI Filter 기능 제공"""

texts = [test_text]

print(f"입력 텍스트 길이: {len(test_text)} 글자")
print(f"입력 텍스트 미리보기: {test_text[:150]}...")

try:
    # Context manager 사용
    with summarizer:
        result = summarizer.summarize(texts)
    
    print("\n✓ 요약 완료")
    print(f"  - 입력 토큰 수: {result.input_tokens}")
    print(f"  - 원본 길이: {len(test_text)} 글자")
    print(f"  - 요약 길이: {len(result.text)} 글자")
    print(f"  - 압축률: {len(result.text) / len(test_text) * 100:.1f}%")
    
    print(f"\n  요약 결과:")
    print(f"  {result.text}")

except Exception as e:
    print(f"⚠️  요약 실패: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 3. 여러 문장 요약 테스트
# ============================================================================
print("\n[3] 여러 문장 요약")
print("-" * 80)

multiple_texts = [
    "Python은 1991년 귀도 반 로섬이 개발한 인터프리터 프로그래밍 언어입니다.",
    "Python은 간결하고 읽기 쉬운 문법으로 초보자도 쉽게 배울 수 있습니다.",
    "데이터 분석, 웹 개발, 인공지능 등 다양한 분야에서 널리 사용됩니다.",
    "풍부한 라이브러리와 프레임워크가 제공되어 개발 생산성이 높습니다.",
    "크로스 플랫폼을 지원하여 Windows, macOS, Linux 등에서 실행 가능합니다.",
]

total_length = sum(len(text) for text in multiple_texts)
print(f"입력 텍스트 수: {len(multiple_texts)}개")
print(f"총 텍스트 길이: {total_length} 글자")

for idx, text in enumerate(multiple_texts, 1):
    print(f"  {idx}. {text}")

try:
    with summarizer:
        result = summarizer.summarize(multiple_texts)
    
    print("\n✓ 여러 문장 요약 완료")
    print(f"  - 입력 토큰 수: {result.input_tokens}")
    print(f"  - 원본 총 길이: {total_length} 글자")
    print(f"  - 요약 길이: {len(result.text)} 글자")
    print(f"  - 압축률: {len(result.text) / total_length * 100:.1f}%")
    
    print(f"\n  요약 결과:")
    print(f"  {result.text}")

except Exception as e:
    print(f"⚠️  여러 문장 요약 실패: {e}")

# ============================================================================
# 4. 짧은 텍스트 요약 테스트
# ============================================================================
print("\n[4] 짧은 텍스트 요약")
print("-" * 80)

short_text = "CLOVA Studio는 네이버의 초거대 AI 모델을 활용한 개발 플랫폼입니다."
short_texts = [short_text]

print(f"입력 텍스트: {short_text}")

try:
    with summarizer:
        result = summarizer.summarize(short_texts)
    
    print("\n✓ 짧은 텍스트 요약 완료")
    print(f"  - 입력 토큰 수: {result.input_tokens}")
    print(f"  - 요약 결과: {result.text}")

except Exception as e:
    print(f"⚠️  짧은 텍스트 요약 실패: {e}")

# ============================================================================
# 5. 입력 검증 테스트
# ============================================================================
print("\n[5] 입력 검증 테스트")
print("-" * 80)

# 빈 리스트 테스트
try:
    with summarizer:
        result = summarizer.summarize([])
    print("⚠️  빈 리스트가 허용되었습니다 (예상치 못한 동작)")
except ValueError as e:
    print(f"✓ 빈 리스트 검증 성공: {e}")
except Exception as e:
    print(f"⚠️  예상치 못한 오류: {e}")

# 빈 문자열 리스트 테스트
try:
    with summarizer:
        result = summarizer.summarize([""])
    print("⚠️  빈 문자열이 허용되었습니다 (예상치 못한 동작)")
except (ValueError, Exception) as e:
    print(f"✓ 빈 문자열 검증 확인: {e}")

# ============================================================================
# 6. 긴 문서 요약 테스트
# ============================================================================
print("\n[6] 긴 문서 요약")
print("-" * 80)

long_text = """CLOVA Studio는 네이버가 개발한 초거대 AI인 HyperCLOVA X를 기반으로 한 AI 서비스 개발 플랫폼입니다. 
이 플랫폼은 개발자와 기업이 자체 AI 서비스를 쉽고 빠르게 구축할 수 있도록 다양한 도구와 API를 제공합니다.

CLOVA Studio의 주요 기능으로는 텍스트 생성, 요약, 번역, 감정 분석, 질의응답 등이 있습니다. 
특히 한국어 처리에 특화되어 있어 한국 시장에 최적화된 AI 서비스를 만들 수 있습니다.

플랫폼은 직관적인 UI를 통해 코딩 없이도 AI 모델을 테스트하고 튜닝할 수 있는 기능을 제공합니다. 
또한 REST API를 통해 다양한 프로그래밍 언어와 쉽게 통합할 수 있습니다.

CLOVA Studio는 금융, 의료, 교육, 커머스 등 다양한 산업 분야에서 활용되고 있으며, 
기업들이 고객 서비스 자동화, 콘텐츠 생성, 데이터 분석 등의 업무를 효율적으로 처리할 수 있도록 돕고 있습니다.

보안과 프라이버시 측면에서도 엄격한 기준을 준수하여 기업 데이터를 안전하게 보호합니다. 
또한 지속적인 모델 업데이트를 통해 최신 AI 기술을 사용자에게 제공하고 있습니다."""

long_texts = [long_text]

print(f"입력 텍스트 길이: {len(long_text)} 글자")
print(f"입력 텍스트 미리보기: {long_text[:200]}...")

try:
    with summarizer:
        result = summarizer.summarize(long_texts)
    
    print("\n✓ 긴 문서 요약 완료")
    print(f"  - 입력 토큰 수: {result.input_tokens}")
    print(f"  - 원본 길이: {len(long_text)} 글자")
    print(f"  - 요약 길이: {len(result.text)} 글자")
    print(f"  - 압축률: {len(result.text) / len(long_text) * 100:.1f}%")
    
    print(f"\n  요약 결과:")
    print(f"  {result.text}")

except Exception as e:
    print(f"⚠️  긴 문서 요약 실패: {e}")

# ============================================================================
# 테스트 완료
# ============================================================================
print("\n" + "=" * 80)
print("✅ 모든 Summarization 테스트 완료")
print("=" * 80)

