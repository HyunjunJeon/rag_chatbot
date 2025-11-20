"""
OpenRouter Embedding API 래퍼.

SentenceTransformer와 동일한 인터페이스를 제공하여 기존 코드와의 호환성을 유지합니다.
"""

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from dotenv import load_dotenv

__all__ = ["OpenRouterEmbeddings"]

# .env 파일 로드
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class OpenRouterEmbeddings:
    """
    OpenRouter API를 사용한 임베딩 생성 클래스.

    SentenceTransformer와 동일한 인터페이스를 제공합니다.

    예시:
        ```python
        embeddings = OpenRouterEmbeddings(
            model="qwen/qwen-3-embedding-4b"
        )

        # 단일 텍스트
        vector = embeddings.encode("안녕하세요")

        # 여러 텍스트 (배치)
        vectors = embeddings.encode(["텍스트1", "텍스트2"])

        # 차원 확인
        dim = embeddings.get_sentence_embedding_dimension()
        ```
    """

    def __init__(
        self,
        model: str = "qwen/qwen3-embedding-4b",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 3,
        timeout: int = 60,
    ) -> None:
        """
        초기화.

        매개변수:
            model: 임베딩 모델 이름
            api_key: OpenRouter API 키 (None이면 환경변수에서 가져옴)
            base_url: OpenRouter API Base URL
            max_retries: 실패 시 재시도 횟수
            timeout: API 호출 타임아웃 (초)
        """
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout

        # API 키 설정
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API 키가 필요합니다. "
                "환경변수 OPENROUTER_API_KEY를 설정하거나 api_key 인자를 전달하세요."
            )

        # 차원 수 (첫 호출 시 확인)
        self._dimension: int | None = None

        print(f"✅ OpenRouterEmbeddings 초기화: {model}")

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """
        OpenRouter API를 호출하여 임베딩을 생성합니다.

        매개변수:
            texts: 텍스트 리스트

        반환값:
            임베딩 벡터 리스트

        예외:
            Exception: API 호출 실패 시
        """
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": texts}

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    url, headers=headers, json=payload, timeout=self.timeout
                )
                response.raise_for_status()

                data = response.json()

                # 임베딩 추출
                embeddings = [item["embedding"] for item in data["data"]]

                # 차원 수 저장 (첫 호출 시)
                if self._dimension is None:
                    self._dimension = len(embeddings[0])
                    print(f"   임베딩 차원: {self._dimension}")

                return embeddings

            except requests.exceptions.Timeout:
                last_error = f"타임아웃 ({self.timeout}초)"
                print(f"   ⚠️  시도 {attempt}/{self.max_retries} 실패: {last_error}")
                time.sleep(2**attempt)  # 지수 백오프

            except requests.exceptions.HTTPError as e:
                # 에러 응답 본문 출력
                error_msg = f"HTTP 오류 {e.response.status_code}"
                try:
                    error_data = e.response.json()
                    error_msg += f": {error_data}"
                except Exception:
                    error_msg += f": {e.response.text}"
                
                last_error = error_msg
                print(f"   ⚠️  시도 {attempt}/{self.max_retries} 실패: {last_error}")

                # Rate limit 오류면 더 오래 대기
                if e.response.status_code == 429:
                    time.sleep(5 * attempt)
                else:
                    time.sleep(2**attempt)

            except Exception as e:
                last_error = str(e)
                print(f"   ⚠️  시도 {attempt}/{self.max_retries} 실패: {last_error}")
                time.sleep(2**attempt)

        # 모든 재시도 실패
        raise Exception(f"API 호출 실패: {last_error}")

    def encode(
        self,
        sentences: str | list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        **kwargs: Any,
    ) -> np.ndarray | list[np.ndarray]:
        """
        텍스트를 임베딩 벡터로 변환합니다.

        SentenceTransformer의 encode() 메서드와 동일한 인터페이스입니다.

        매개변수:
            sentences: 임베딩할 텍스트 또는 텍스트 리스트
            batch_size: 배치 크기
            show_progress_bar: 진행 바 표시 (미구현)
            convert_to_numpy: numpy 배열로 변환 여부

        반환값:
            임베딩 벡터 (numpy array)
        """
        # 단일 텍스트면 리스트로 변환
        single_input = isinstance(sentences, str)
        if single_input:
            sentences = [sentences]

        all_embeddings: list[list[float]] = []

        # 배치 처리
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]

            try:
                batch_embeddings = self._call_api(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"\n❌ 배치 {i//batch_size + 1} 처리 실패: {e}")
                raise

        # numpy 배열로 변환
        if convert_to_numpy:
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            if single_input:
                return embeddings_array[0]
            return embeddings_array
        else:
            if single_input:
                return all_embeddings[0]
            return all_embeddings

    def get_sentence_embedding_dimension(self) -> int:
        """
        임베딩 벡터의 차원 수를 반환합니다.

        첫 encode() 호출 후에 사용 가능합니다.

        반환값:
            임베딩 차원 수

        예외:
            ValueError: 아직 임베딩을 생성하지 않은 경우
        """
        if self._dimension is None:
            # 테스트 임베딩 생성
            print("   차원 확인을 위한 테스트 임베딩 생성 중...")
            self.encode("test")

        if self._dimension is None:
            raise ValueError("임베딩 차원을 확인할 수 없습니다.")

        return self._dimension


def main() -> None:
    """테스트 함수."""
    print("=" * 80)
    print("🧪 OpenRouter Embeddings 테스트")
    print("=" * 80)

    try:
        # 임베딩 모델 초기화
        embeddings = OpenRouterEmbeddings(model="qwen/qwen3-embedding-4b")

        # 테스트 텍스트
        test_texts = [
            "GPU 메모리 부족 문제를 해결하는 방법은?",
            "데이터 증강 기법에는 어떤 것들이 있나요?",
            "optimizer 선택 기준",
        ]

        print(f"\n📝 테스트 텍스트: {len(test_texts)}개")
        for i, text in enumerate(test_texts, 1):
            print(f"   {i}. {text}")

        # 임베딩 생성
        print("\n🔄 임베딩 생성 중...")
        vectors = embeddings.encode(test_texts)

        print("\n✅ 임베딩 생성 완료!")
        print(f"   Shape: {vectors.shape}")
        print(f"   Dtype: {vectors.dtype}")
        print(f"   차원: {embeddings.get_sentence_embedding_dimension()}")

        # 벡터 통계
        print("\n📊 벡터 통계:")
        print(f"   최소값: {vectors.min():.6f}")
        print(f"   최대값: {vectors.max():.6f}")
        print(f"   평균: {vectors.mean():.6f}")
        print(f"   표준편차: {vectors.std():.6f}")

        # 유사도 계산
        print("\n🔍 유사도 계산:")
        from numpy.linalg import norm

        for i in range(len(test_texts)):
            for j in range(i + 1, len(test_texts)):
                similarity = np.dot(vectors[i], vectors[j]) / (
                    norm(vectors[i]) * norm(vectors[j])
                )
                print(f"   [{i+1}] <-> [{j+1}]: {similarity:.4f}")

        print("\n" + "=" * 80)
        print("✅ 테스트 완료!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

