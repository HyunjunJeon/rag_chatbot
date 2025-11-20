from typing import List
import httpx
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, SecretStr, ConfigDict


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
