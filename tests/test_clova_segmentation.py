"""
Clova Studio Segmentation API 통합 테스트

이 파일은 Clova Studio의 문단 나누기 API 기본 사용법을 보여줍니다.

실행 전 준비사항:
1. .env 파일에 CLOVASTUDIO_API_KEY 설정
2. .env 파일에 CLOVASTUDIO_SEGMENTATION_ENDPOINT 설정 (선택사항)

실행 방법:
    uv run python tests/test_clova_segmentation.py
"""

from dotenv import load_dotenv

from naver_connect_chatbot.config.settings.clova import ClovaStudioSegmentationSettings
from naver_connect_chatbot.rag.segmentation import ClovaStudioSegmenter

# 환경변수 로드
load_dotenv()

print("=" * 80)
print("Clova Studio Segmentation API 테스트 시작")
print("=" * 80)

# ============================================================================
# 1. ClovaStudioSegmenter 초기화 테스트
# ============================================================================
print("\n[1] ClovaStudioSegmenter 초기화")
print("-" * 80)

try:
    segmentation_settings = ClovaStudioSegmentationSettings()
    segmenter = ClovaStudioSegmenter.from_settings(segmentation_settings)
    print("✓ ClovaStudioSegmenter 초기화 완료")
    print(f"  - 엔드포인트: {segmentation_settings.endpoint}")
    print(f"  - 타임아웃: {segmentation_settings.request_timeout}초")
    print(f"  - Alpha: {segmentation_settings.alpha}")
    print(f"  - Seg Count: {segmentation_settings.seg_count}")
    print(f"  - Post Process: {segmentation_settings.post_process}")
except Exception as e:
    print(f"⚠️  Segmenter 초기화 실패: {e}")
    print("   .env 파일의 CLOVASTUDIO_API_KEY를 확인하세요.")
    exit(1)

# ============================================================================
# 2. 문단 나누기 테스트
# ============================================================================
print("\n[2] 문단 나누기 실행")
print("-" * 80)

# 테스트용 긴 텍스트 (여러 주제가 섞인 문서)
test_text = """노트는 어떻게 생성할 수 있나요?
두 가지 방법이 있습니다.
클로바노트 앱에서 추가 버튼을 눌러 녹음을 시작하거나, 스마트폰에 저장해둔 녹음 파일을 불러오면 노트가 생성된답니다.
이렇게 만들어진 노트는 앱뿐만 아니라 PC의 클로바노트 웹사이트에서도 연동되어 확인하실 수 있는데요.
클로바노트 사이트에서는 저장된 녹음파일을 불러오면 노트를 만들 수 있답니다.
북마크는 어떻게 사용하는 건가요?
클로바노트 앱 화면에서 녹음 중간에 북마크 버튼을 누르면, 아래처럼 표시되어 녹음을 마치고 나서도 필요한 구간을 쉽게 찾을 수 있죠.
평소 녹음을 마치고 나면 분명히 다시 찾아보고 싶은 녹음 구간이 있었을 거예요.
그런 순간을 위해 북마크를 제공하고 있답니다.
그럼 녹음한 음성은 어떻게 들어볼 수 있나요?
생성된 노트에서 기록된 대화를 선택하면 녹음 음성을 다시 들어볼 수 있답니다.
만약 음성 기록이 잘못된 구간이 있다면 다시 한 번 음성을 들어보고 편집 버튼을 눌러 쉽게 바로잡을 수 있죠."""

print(f"입력 텍스트 길이: {len(test_text)} 글자")
print(f"입력 텍스트 미리보기: {test_text[:100]}...")

try:
    # Context manager 사용
    with segmenter:
        result = segmenter.segment(test_text)
    
    print("\n✓ 문단 나누기 완료")
    print(f"  - 입력 토큰 수: {result.input_tokens}")
    print(f"  - 분리된 주제 수: {len(result.topic_segments)}")
    
    print("\n  분리된 주제별 문단:")
    for idx, segment in enumerate(result.topic_segments, 1):
        print(f"\n  [주제 {idx}] (문장 수: {len(segment)})")
        for sentence in segment:
            print(f"    - {sentence}")
    
    print("\n  문단 인덱스 (span):")
    for idx, span in enumerate(result.span, 1):
        print(f"    주제 {idx}: {span}")

except Exception as e:
    print(f"⚠️  문단 나누기 실패: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 3. 짧은 텍스트 테스트
# ============================================================================
print("\n[3] 짧은 텍스트 테스트")
print("-" * 80)

short_text = """CLOVA Studio는 네이버의 초거대 AI 모델 HyperCLOVA X를 활용한 개발 플랫폼입니다.
다양한 AI 기능을 제공하여 개발자들이 쉽게 AI 서비스를 만들 수 있습니다.
텍스트 생성, 요약, 분류, 대화 등의 기능이 포함되어 있습니다."""

print(f"입력 텍스트 길이: {len(short_text)} 글자")
print(f"입력 텍스트: {short_text}")

try:
    with segmenter:
        result = segmenter.segment(short_text)
    
    print("\n✓ 짧은 텍스트 문단 나누기 완료")
    print(f"  - 입력 토큰 수: {result.input_tokens}")
    print(f"  - 분리된 주제 수: {len(result.topic_segments)}")
    
    print("\n  분리된 주제별 문단:")
    for idx, segment in enumerate(result.topic_segments, 1):
        print(f"    [주제 {idx}] {segment}")

except Exception as e:
    print(f"⚠️  짧은 텍스트 처리 실패: {e}")

# ============================================================================
# 4. 입력 검증 테스트
# ============================================================================
print("\n[4] 입력 검증 테스트")
print("-" * 80)

# 빈 문자열 테스트
try:
    with segmenter:
        result = segmenter.segment("")
    print("⚠️  빈 문자열이 허용되었습니다 (예상치 못한 동작)")
except ValueError as e:
    print(f"✓ 빈 문자열 검증 성공: {e}")
except Exception as e:
    print(f"⚠️  예상치 못한 오류: {e}")

# 공백만 있는 문자열 테스트
try:
    with segmenter:
        result = segmenter.segment("   \n\t   ")
    print("⚠️  공백 문자열이 허용되었습니다 (예상치 못한 동작)")
except ValueError as e:
    print(f"✓ 공백 문자열 검증 성공: {e}")
except Exception as e:
    print(f"⚠️  예상치 못한 오류: {e}")

# ============================================================================
# 테스트 완료
# ============================================================================
print("\n" + "=" * 80)
print("✅ 모든 Segmentation 테스트 완료")
print("=" * 80)

