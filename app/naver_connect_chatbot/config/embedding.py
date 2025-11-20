"""
Embedding 모듈

이 모듈은 다양한 임베딩 서비스를 LangChain Embeddings 인터페이스로 제공합니다.
- NaverCloudEmbeddings: Naver Cloud의 BGE-M3 임베딩 서비스
- OpenRouterEmbeddings: OpenRouter API를 통한 임베딩 서비스
"""

import os
import time
from typing import List

import httpx
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, SecretStr, ConfigDict, model_validator


class NaverCloudEmbeddings(Embeddings, BaseModel):
    """
    Naver Cloud 에서 제공하는 Embedding V2 API 를 활용합니다
    Embedding V2 는 BGE-M3 모델을 활용 중(https://api.ncloud-docs.com/docs/clovastudio-embeddingv2)
    BGE-M3 차원: 1024
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_url: str = Field(..., description="The URL of the BGE-M3 embedding service")
    api_key: SecretStr | None = Field(default=None, description="API Key for the service if required")
    timeout: float = Field(default=10.0, description="Timeout for HTTP requests")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """검색 문서를 임베딩합니다."""
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """질의 문장을 임베딩합니다."""
        return self._get_embeddings([text])[0]

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        동기 방식으로 임베딩을 생성합니다.

        Warning: 비동기 컨텍스트에서는 aembed_documents() 또는 aembed_query()를 사용하세요.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"

        payload = {"inputs": texts}

        try:
            response = httpx.post(self.api_url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            # API가 직접 임베딩 리스트를 반환한다고 가정합니다.
            # (예: TEI, Triton, 사용자 정의 FastAPI) 등 공통 서빙 패턴에 맞춰 조정 가능.
            # 여기서는 리스트 혹은 {"embeddings": ...} 구조를 우선 가정합니다.
            data = response.json()
            if isinstance(data, list):
                return data
            elif "embeddings" in data:
                return data["embeddings"]
            else:
                raise ValueError(f"Unexpected response format: {data}")
        except httpx.HTTPError as e:
            raise ValueError(f"Error calling embedding service: {e}")

    async def _aget_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        비동기 방식으로 임베딩을 생성합니다.

        이벤트 루프 블로킹을 방지하기 위해 httpx.AsyncClient를 사용합니다.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"

        payload = {"inputs": texts}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, list):
                    return data
                elif "embeddings" in data:
                    return data["embeddings"]
                else:
                    raise ValueError(f"Unexpected response format: {data}")
        except httpx.HTTPError as e:
            raise ValueError(f"Error calling embedding service: {e}")

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        비동기적으로 여러 문서를 임베딩합니다.

        매개변수:
            texts: 임베딩할 문서 리스트

        반환값:
            각 문서에 대한 임베딩 벡터 리스트
        """
        return await self._aget_embeddings(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """
        비동기적으로 단일 쿼리를 임베딩합니다.

        매개변수:
            text: 임베딩할 쿼리 텍스트

        반환값:
            쿼리의 임베딩 벡터
        """
        embeddings = await self._aget_embeddings([text])
        return embeddings[0]


class OpenRouterEmbeddings(Embeddings, BaseModel):
    """
    OpenRouter API를 사용한 LangChain 호환 Embeddings 클래스.
    
    OpenRouter를 통해 다양한 임베딩 모델을 사용할 수 있으며,
    qwen/qwen3-embedding-4b 모델을 기본으로 사용합니다.
    
    속성:
        model: 사용할 임베딩 모델 이름
        api_key: OpenRouter API 키 (환경변수 OPENROUTER_API_KEY에서 자동 로드)
        base_url: OpenRouter API 기본 URL
        timeout: API 호출 타임아웃 (초)
        max_retries: 실패 시 최대 재시도 횟수
        batch_size: 배치 처리 크기
    
    예시:
        >>> from pydantic import SecretStr
        >>> embeddings = OpenRouterEmbeddings(
        ...     model="qwen/qwen3-embedding-4b",
        ...     api_key=SecretStr("your-api-key")
        ... )
        >>> 
        >>> # 단일 쿼리 임베딩
        >>> query_vector = embeddings.embed_query("GPU 메모리 부족 해결 방법")
        >>> 
        >>> # 다중 문서 임베딩
        >>> doc_vectors = embeddings.embed_documents(["문서1", "문서2", "문서3"])
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = Field(
        default="qwen/qwen3-embedding-4b",
        description="OpenRouter 임베딩 모델 이름"
    )
    api_key: SecretStr = Field(
        default=None,
        description="OpenRouter API 키 (환경변수 OPENROUTER_API_KEY에서 자동 로드)"
    )
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API 기본 URL"
    )
    timeout: float = Field(
        default=60.0,
        description="API 호출 타임아웃 (초)"
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="실패 시 최대 재시도 횟수"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=100,
        description="배치 처리 크기"
    )

    @model_validator(mode="before")
    @classmethod
    def validate_api_key(cls, values: dict) -> dict:
        """API 키를 환경변수에서 로드합니다."""
        if values.get("api_key") is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenRouter API 키가 필요합니다. "
                    "환경변수 OPENROUTER_API_KEY를 설정하거나 api_key 인자를 전달하세요."
                )
            values["api_key"] = SecretStr(api_key)
        elif isinstance(values["api_key"], str):
            values["api_key"] = SecretStr(values["api_key"])
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        동기적으로 여러 문서를 임베딩합니다.

        매개변수:
            texts: 임베딩할 문서 리스트

        반환값:
            각 문서에 대한 임베딩 벡터 리스트
        
        예시:
            >>> embeddings = OpenRouterEmbeddings()
            >>> vectors = embeddings.embed_documents(["문서1", "문서2"])
            >>> len(vectors)
            2
        """
        all_embeddings: List[List[float]] = []

        # 배치 처리
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self._get_embeddings(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        동기적으로 단일 쿼리를 임베딩합니다.

        매개변수:
            text: 임베딩할 쿼리 텍스트

        반환값:
            쿼리의 임베딩 벡터
        
        예시:
            >>> embeddings = OpenRouterEmbeddings()
            >>> vector = embeddings.embed_query("검색 쿼리")
            >>> len(vector) > 0
            True
        """
        return self._get_embeddings([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        비동기적으로 여러 문서를 임베딩합니다.

        매개변수:
            texts: 임베딩할 문서 리스트

        반환값:
            각 문서에 대한 임베딩 벡터 리스트
        
        예시:
            >>> embeddings = OpenRouterEmbeddings()
            >>> vectors = await embeddings.aembed_documents(["문서1", "문서2"])
            >>> len(vectors)
            2
        """
        all_embeddings: List[List[float]] = []

        # 배치 처리
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = await self._aget_embeddings(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def aembed_query(self, text: str) -> List[float]:
        """
        비동기적으로 단일 쿼리를 임베딩합니다.

        매개변수:
            text: 임베딩할 쿼리 텍스트

        반환값:
            쿼리의 임베딩 벡터
        
        예시:
            >>> embeddings = OpenRouterEmbeddings()
            >>> vector = await embeddings.aembed_query("검색 쿼리")
            >>> len(vector) > 0
            True
        """
        embeddings = await self._aget_embeddings([text])
        return embeddings[0]

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        동기 방식으로 임베딩을 생성합니다.

        매개변수:
            texts: 임베딩할 텍스트 리스트

        반환값:
            임베딩 벡터 리스트

        예외:
            ValueError: API 호출 실패 시
        """
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": texts,
        }

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = httpx.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()

                data = response.json()
                
                # OpenRouter API 응답 형식: {"data": [{"embedding": [...]}, ...]}
                embeddings = [item["embedding"] for item in data["data"]]
                
                return embeddings

            except httpx.TimeoutException:
                last_error = f"타임아웃 ({self.timeout}초)"
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)

            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP 오류 {e.response.status_code}"
                try:
                    error_data = e.response.json()
                    error_msg += f": {error_data}"
                except Exception:
                    error_msg += f": {e.response.text}"
                
                last_error = error_msg
                
                # Rate limit 오류면 더 오래 대기
                if e.response.status_code == 429 and attempt < self.max_retries:
                    wait_time = 5 * attempt
                    time.sleep(wait_time)
                elif attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    break

            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)

        # 모든 재시도 실패
        raise ValueError(f"OpenRouter API 호출 실패: {last_error}")

    async def _aget_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        비동기 방식으로 임베딩을 생성합니다.

        매개변수:
            texts: 임베딩할 텍스트 리스트

        반환값:
            임베딩 벡터 리스트

        예외:
            ValueError: API 호출 실패 시
        """
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": texts,
        }

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()

                    data = response.json()
                    
                    # OpenRouter API 응답 형식: {"data": [{"embedding": [...]}, ...]}
                    embeddings = [item["embedding"] for item in data["data"]]
                    
                    return embeddings

            except httpx.TimeoutException:
                last_error = f"타임아웃 ({self.timeout}초)"
                if attempt < self.max_retries:
                    import asyncio
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)

            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP 오류 {e.response.status_code}"
                try:
                    error_data = e.response.json()
                    error_msg += f": {error_data}"
                except Exception:
                    error_msg += f": {e.response.text}"
                
                last_error = error_msg
                
                # Rate limit 오류면 더 오래 대기
                if e.response.status_code == 429 and attempt < self.max_retries:
                    import asyncio
                    wait_time = 5 * attempt
                    await asyncio.sleep(wait_time)
                elif attempt < self.max_retries:
                    import asyncio
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    break

            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    import asyncio
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)

        # 모든 재시도 실패
        raise ValueError(f"OpenRouter API 호출 실패: {last_error}")
