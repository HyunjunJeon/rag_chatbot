"""
Clova Studio RAG Reasoning API 테스트

이 파일은 Clova Studio의 RAG Reasoning API 기본 사용법을 보여줍니다.

실행 전 준비사항:
1. .env 파일에 CLOVASTUDIO_API_KEY 설정
2. .env 파일에 CLOVASTUDIO_RAG_REASONING_ENDPOINT 설정 (선택사항)

실행 방법:
    # 단위 테스트 (Mock 사용, 빠름)
    uv run pytest tests/test_clova_rag_reasoning.py -v
    
    # 통합 테스트 (실제 API 호출, 느림)
    uv run pytest tests/test_clova_rag_reasoning.py -m integration -v
"""

import pytest
from unittest.mock import Mock, patch

from naver_connect_chatbot.config.settings.clova import ClovaStudioRAGReasoningSettings
from naver_connect_chatbot.rag.rag_reasoning import (
    ClovaStudioRAGReasoning,
    convert_langchain_tool_to_rag_reasoning,
    convert_langchain_tools_to_rag_reasoning,
)


# ============================================================================
# 단위 테스트 (Mock)
# ============================================================================


class TestLangChainToolConversion:
    """LangChain Tool 변환 테스트"""
    
    def test_convert_simple_tool(self):
        """간단한 LangChain tool 변환 테스트"""
        try:
            from langchain_core.tools import tool
        except ImportError:
            pytest.skip("langchain_core가 설치되지 않았습니다")
        
        @tool
        def search_documents(query: str) -> str:
            """문서를 검색합니다.
            
            Args:
                query: 검색할 질의
            """
            return f"Search results for: {query}"
        
        rag_tool = convert_langchain_tool_to_rag_reasoning(search_documents)
        
        assert rag_tool["type"] == "function"
        assert "function" in rag_tool
        assert rag_tool["function"]["name"] == "search_documents"
        assert "문서를 검색합니다" in rag_tool["function"]["description"]
        assert "parameters" in rag_tool["function"]
        assert rag_tool["function"]["parameters"]["type"] == "object"
        assert "query" in rag_tool["function"]["parameters"]["properties"]
        assert "query" in rag_tool["function"]["parameters"]["required"]
    
    def test_convert_complex_tool(self):
        """복잡한 파라미터를 가진 tool 변환 테스트"""
        try:
            from langchain_core.tools import tool
            from pydantic import BaseModel, Field
        except ImportError:
            pytest.skip("langchain_core가 설치되지 않았습니다")
        
        class SearchInput(BaseModel):
            query: str = Field(description="검색할 질의")
            max_results: int = Field(default=10, description="최대 결과 수")
            include_metadata: bool = Field(default=False, description="메타데이터 포함 여부")
        
        @tool(args_schema=SearchInput)
        def advanced_search(query: str, max_results: int = 10, include_metadata: bool = False) -> str:
            """고급 문서 검색을 수행합니다."""
            return f"Results: {query}"
        
        rag_tool = convert_langchain_tool_to_rag_reasoning(advanced_search)
        
        assert rag_tool["type"] == "function"
        assert rag_tool["function"]["name"] == "advanced_search"
        
        params = rag_tool["function"]["parameters"]
        assert "query" in params["properties"]
        assert "max_results" in params["properties"]
        assert "include_metadata" in params["properties"]
        
        # query는 필수, 나머지는 선택
        assert "query" in params["required"]
        assert "max_results" not in params["required"]
        assert "include_metadata" not in params["required"]
    
    def test_convert_multiple_tools(self):
        """여러 도구 일괄 변환 테스트"""
        try:
            from langchain_core.tools import tool
        except ImportError:
            pytest.skip("langchain_core가 설치되지 않았습니다")
        
        @tool
        def search_docs(query: str) -> str:
            """문서 검색"""
            return f"Results: {query}"
        
        @tool
        def get_weather(city: str) -> str:
            """날씨 조회"""
            return f"Weather in {city}"
        
        @tool
        def calculate(expression: str) -> str:
            """계산 수행"""
            return f"Result: {expression}"
        
        rag_tools = convert_langchain_tools_to_rag_reasoning([search_docs, get_weather, calculate])
        
        assert len(rag_tools) == 3
        assert all(tool["type"] == "function" for tool in rag_tools)
        
        tool_names = [tool["function"]["name"] for tool in rag_tools]
        assert "search_docs" in tool_names
        assert "get_weather" in tool_names
        assert "calculate" in tool_names
    
    def test_convert_empty_tools_list(self):
        """빈 도구 리스트 변환 시 예외 발생"""
        with pytest.raises(ValueError, match="tools는 비어있을 수 없습니다"):
            convert_langchain_tools_to_rag_reasoning([])
    
    def test_convert_without_langchain(self):
        """langchain_core 없이 변환 시도 시 ImportError"""
        # langchain_core import를 mock하여 실패하도록 만듦
        with patch.dict("sys.modules", {"langchain_core.utils.function_calling": None}):
            with pytest.raises(ImportError, match="langchain_core를 import할 수 없습니다"):
                # 임의의 객체로 시도
                convert_langchain_tool_to_rag_reasoning(Mock())


class TestClovaStudioRAGReasoningUnit:
    """ClovaStudioRAGReasoning 단위 테스트 (Mock 사용)"""
    
    def test_initialization(self):
        """초기화 테스트"""
        rag_reasoning = ClovaStudioRAGReasoning(
            endpoint="https://test.example.com/rag-reasoning",
            api_key="test-api-key",
            max_tokens=1024,
            temperature=0.5,
        )
        
        assert rag_reasoning.endpoint == "https://test.example.com/rag-reasoning"
        assert rag_reasoning.api_key == "test-api-key"
        assert rag_reasoning.max_tokens == 1024
        assert rag_reasoning.temperature == 0.5
        assert rag_reasoning.top_p == 0.8  # 기본값
        assert rag_reasoning.top_k == 0  # 기본값
    
    def test_initialization_with_invalid_endpoint(self):
        """잘못된 엔드포인트로 초기화 시 예외 발생"""
        with pytest.raises(ValueError, match="endpoint는 빈 문자열일 수 없습니다"):
            ClovaStudioRAGReasoning(
                endpoint="",
                api_key="test-api-key",
            )
    
    def test_initialization_with_invalid_api_key(self):
        """잘못된 API 키로 초기화 시 예외 발생"""
        with pytest.raises(ValueError, match="api_key는 빈 문자열일 수 없습니다"):
            ClovaStudioRAGReasoning(
                endpoint="https://test.example.com/rag-reasoning",
                api_key="",
            )
    
    def test_from_settings(self):
        """Settings 객체로부터 인스턴스 생성 테스트"""
        settings = ClovaStudioRAGReasoningSettings()
        rag_reasoning = ClovaStudioRAGReasoning.from_settings(settings)
        
        assert rag_reasoning.endpoint == settings.endpoint
        assert rag_reasoning.max_tokens == settings.max_tokens
        assert rag_reasoning.temperature == settings.temperature
    
    def test_build_headers(self):
        """헤더 생성 테스트"""
        rag_reasoning = ClovaStudioRAGReasoning(
            endpoint="https://test.example.com/rag-reasoning",
            api_key="test-api-key",
        )
        
        headers = rag_reasoning._build_headers()
        
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test-api-key"
    
    def test_build_payload(self):
        """페이로드 생성 테스트"""
        rag_reasoning = ClovaStudioRAGReasoning(
            endpoint="https://test.example.com/rag-reasoning",
            api_key="test-api-key",
            top_p=0.8,
            top_k=0,
            max_tokens=1024,
            temperature=0.5,
            repetition_penalty=1.1,
            seed=0,
            include_ai_filters=True,
        )
        
        messages = [{"role": "user", "content": "VPC 삭제 방법은?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "description": "문서 검색",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "검색어"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        payload = rag_reasoning._build_payload(messages, tools, "auto")
        
        assert payload["messages"] == messages
        assert payload["tools"] == tools
        assert payload["toolChoice"] == "auto"
        assert payload["topP"] == 0.8
        assert payload["topK"] == 0
        assert payload["maxTokens"] == 1024
        assert payload["temperature"] == 0.5
        assert payload["repetitionPenalty"] == 1.1
        assert payload["stop"] == []
        assert payload["seed"] == 0
        assert payload["includeAiFilters"] is True
    
    def test_validate_messages_empty(self):
        """빈 messages 검증 테스트"""
        from naver_connect_chatbot.rag.rag_reasoning import _validate_messages
        
        with pytest.raises(ValueError, match="messages는 비어있을 수 없습니다"):
            _validate_messages([])
    
    def test_validate_messages_invalid_role(self):
        """잘못된 role 검증 테스트"""
        from naver_connect_chatbot.rag.rag_reasoning import _validate_messages
        
        with pytest.raises(ValueError, match="role은"):
            _validate_messages([{"role": "invalid", "content": "test"}])
    
    def test_validate_messages_missing_content(self):
        """content 누락 검증 테스트"""
        from naver_connect_chatbot.rag.rag_reasoning import _validate_messages
        
        with pytest.raises(ValueError, match="content 필드가 없습니다"):
            _validate_messages([{"role": "user"}])
    
    def test_validate_messages_tool_missing_toolCallId(self):
        """tool 역할에서 toolCallId 누락 검증 테스트"""
        from naver_connect_chatbot.rag.rag_reasoning import _validate_messages
        
        with pytest.raises(ValueError, match="toolCallId가 없습니다"):
            _validate_messages([{"role": "tool", "content": "결과"}])
    
    def test_validate_tools_empty(self):
        """빈 tools 검증 테스트"""
        from naver_connect_chatbot.rag.rag_reasoning import _validate_tools
        
        with pytest.raises(ValueError, match="tools는 비어있을 수 없습니다"):
            _validate_tools([])
    
    def test_validate_tools_invalid_type(self):
        """잘못된 tool type 검증 테스트"""
        from naver_connect_chatbot.rag.rag_reasoning import _validate_tools
        
        with pytest.raises(ValueError, match="type은 'function'이어야 합니다"):
            _validate_tools([{"type": "invalid"}])
    
    def test_validate_tools_missing_name(self):
        """tool function에서 name 누락 검증 테스트"""
        from naver_connect_chatbot.rag.rag_reasoning import _validate_tools
        
        with pytest.raises(ValueError, match="name 필드가 없습니다"):
            _validate_tools([
                {
                    "type": "function",
                    "function": {
                        "description": "테스트",
                        "parameters": {}
                    }
                }
            ])
    
    @patch("naver_connect_chatbot.rag.rag_reasoning.httpx.Client")
    def test_parse_response_success(self, mock_client):
        """응답 파싱 성공 테스트"""
        rag_reasoning = ClovaStudioRAGReasoning(
            endpoint="https://test.example.com/rag-reasoning",
            api_key="test-api-key",
        )
        
        response_data = {
            "status": {"code": "20000", "message": "OK"},
            "result": {
                "message": {
                    "role": "assistant",
                    "content": "VPC 삭제 방법은 다음과 같습니다.",
                    "thinkingContent": "사용자가 VPC 삭제 방법에 대해 문의했습니다.",
                    "toolCalls": []
                },
                "usage": {
                    "promptTokens": 100,
                    "completionTokens": 50,
                    "totalTokens": 150
                }
            }
        }
        
        result = rag_reasoning._parse_response(response_data)
        
        assert result["message"]["role"] == "assistant"
        assert result["message"]["content"] == "VPC 삭제 방법은 다음과 같습니다."
        assert result["usage"]["promptTokens"] == 100
        assert result["usage"]["completionTokens"] == 50
    
    @patch("naver_connect_chatbot.rag.rag_reasoning.httpx.Client")
    def test_parse_response_failure(self, mock_client):
        """응답 파싱 실패 테스트 (잘못된 status code)"""
        rag_reasoning = ClovaStudioRAGReasoning(
            endpoint="https://test.example.com/rag-reasoning",
            api_key="test-api-key",
        )
        
        response_data = {
            "status": {"code": "40000", "message": "Bad Request"},
            "result": {}
        }
        
        with pytest.raises(ValueError, match="API 호출 실패"):
            rag_reasoning._parse_response(response_data)
    
    def test_context_manager_sync(self):
        """동기 context manager 테스트"""
        rag_reasoning = ClovaStudioRAGReasoning(
            endpoint="https://test.example.com/rag-reasoning",
            api_key="test-api-key",
        )
        
        with rag_reasoning as rag:
            assert rag is rag_reasoning
        
        # close가 호출되었는지 확인
        assert rag_reasoning._client is None
    
    @pytest.mark.asyncio
    async def test_context_manager_async(self):
        """비동기 context manager 테스트"""
        rag_reasoning = ClovaStudioRAGReasoning(
            endpoint="https://test.example.com/rag-reasoning",
            api_key="test-api-key",
        )
        
        async with rag_reasoning as rag:
            assert rag is rag_reasoning
        
        # aclose가 호출되었는지 확인
        assert rag_reasoning._async_client is None


# ============================================================================
# 통합 테스트 (실제 API 호출)
# ============================================================================


@pytest.mark.integration
class TestClovaStudioRAGReasoningIntegration:
    """ClovaStudioRAGReasoning 통합 테스트 (실제 API 호출)"""
    
    @pytest.fixture
    def rag_reasoning(self):
        """RAG Reasoning 인스턴스 fixture"""
        settings = ClovaStudioRAGReasoningSettings()
        return ClovaStudioRAGReasoning.from_settings(settings)
    
    @pytest.fixture
    def sample_tools(self):
        """샘플 도구 정의 fixture"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "ncloud_cs_retrieval",
                    "description": "NCloud 관련 검색을 할 때 사용하는 도구입니다.\n나누어 질문해야 하는 경우 쿼리를 쪼개 나누어서 도구를 사용합니다.\n정보를 찾을 수 없었던 경우, 최종 답을 하지 않고 suggested_queries를 참고하여 도구를 다시 사용할 수 있습니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "사용자의 검색어를 정제해서 넣으세요."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    def test_single_turn_invoke(self, rag_reasoning, sample_tools):
        """단일 턴 호출 테스트 (toolCalls 반환 확인)"""
        messages = [
            {"role": "user", "content": "VPC 삭제 방법 알려줘"}
        ]
        
        with rag_reasoning:
            result = rag_reasoning.invoke(
                messages=messages,
                tools=sample_tools,
                tool_choice="auto"
            )
        
        assert "message" in result
        assert "usage" in result
        assert result["message"]["role"] == "assistant"
        
        # 첫 번째 호출은 toolCalls를 반환할 가능성이 높음
        if "toolCalls" in result["message"]:
            tool_calls = result["message"]["toolCalls"]
            assert isinstance(tool_calls, list)
            if len(tool_calls) > 0:
                assert tool_calls[0]["type"] == "function"
                assert "function" in tool_calls[0]
                assert "name" in tool_calls[0]["function"]
                assert "arguments" in tool_calls[0]["function"]
        
        # 토큰 사용량 확인
        assert result["usage"]["promptTokens"] > 0
        assert result["usage"]["totalTokens"] > 0
    
    def test_multi_turn_invoke(self, rag_reasoning, sample_tools):
        """Multi-turn 호출 테스트 (toolCalls → tool → 최종 답변)"""
        # Step 1: 사용자 질문
        messages = [
            {"role": "user", "content": "A100 GPU 빌리는 방법"}
        ]
        
        with rag_reasoning:
            # 첫 번째 호출: toolCalls 받기
            result1 = rag_reasoning.invoke(
                messages=messages,
                tools=sample_tools,
                tool_choice="auto"
            )
        
        assert "message" in result1
        message1 = result1["message"]
        
        # toolCalls가 있는 경우에만 Step 2 진행
        if "toolCalls" in message1 and len(message1["toolCalls"]) > 0:
            tool_call = message1["toolCalls"][0]
            
            # Step 2: 검색 수행 (Mock 데이터)
            search_result = [
                {
                    "id": "doc-248",
                    "doc": "네이버 클라우드 플랫폼 콘솔의 Services > Compute > Server 메뉴에서 GPU A100 서버를 생성할 수 있습니다."
                },
                {
                    "id": "doc-179",
                    "doc": "GPU A100은 KR-1에서만 생성 가능하며, A100 생성 시에는 KR-1의 Subnet을 선택해야 합니다."
                }
            ]
            
            # Step 3: 검색 결과를 tool 역할로 추가하여 재호출
            messages.append({
                "role": "assistant",
                "content": message1.get("content", ""),
                "toolCalls": message1["toolCalls"]
            })
            messages.append({
                "role": "tool",
                "content": str({"search_result": search_result}),
                "toolCallId": tool_call["id"]
            })
            
            with rag_reasoning:
                result2 = rag_reasoning.invoke(
                    messages=messages,
                    tools=sample_tools,
                    tool_choice="auto"
                )
            
            assert "message" in result2
            message2 = result2["message"]
            
            # 최종 답변 확인
            assert message2["role"] == "assistant"
            assert len(message2.get("content", "")) > 0
            
            # 인용 표기가 있는지 확인 (선택적)
            content = message2["content"]
            if "<doc-" in content:
                print(f"\n✓ 인용 표기 발견: {content}")
    
    def test_tool_choice_none(self, rag_reasoning, sample_tools):
        """toolChoice='none' 테스트 (함수 호출 없이 일반 답변)"""
        messages = [
            {"role": "user", "content": "안녕하세요"}
        ]
        
        with rag_reasoning:
            result = rag_reasoning.invoke(
                messages=messages,
                tools=sample_tools,
                tool_choice="none"
            )
        
        assert "message" in result
        message = result["message"]
        
        # toolChoice가 none이므로 toolCalls가 없어야 함
        assert message["role"] == "assistant"
        # toolCalls가 없거나 비어있어야 함
        tool_calls = message.get("toolCalls", [])
        assert len(tool_calls) == 0
    
    def test_force_tool_choice(self, rag_reasoning, sample_tools):
        """특정 함수 강제 호출 테스트"""
        messages = [
            {"role": "user", "content": "GPU 서버 정보"}
        ]
        
        tool_choice = {
            "type": "function",
            "function": {
                "name": "ncloud_cs_retrieval"
            }
        }
        
        with rag_reasoning:
            result = rag_reasoning.invoke(
                messages=messages,
                tools=sample_tools,
                tool_choice=tool_choice
            )
        
        assert "message" in result
        message = result["message"]
        
        # 강제 호출이므로 toolCalls가 있어야 함
        if "toolCalls" in message:
            tool_calls = message["toolCalls"]
            assert len(tool_calls) > 0
            assert tool_calls[0]["function"]["name"] == "ncloud_cs_retrieval"
    
    @pytest.mark.asyncio
    async def test_async_invoke(self, rag_reasoning, sample_tools):
        """비동기 호출 테스트"""
        messages = [
            {"role": "user", "content": "VPC 삭제 방법 알려줘"}
        ]
        
        async with rag_reasoning:
            result = await rag_reasoning.ainvoke(
                messages=messages,
                tools=sample_tools,
                tool_choice="auto"
            )
        
        assert "message" in result
        assert "usage" in result
        assert result["message"]["role"] == "assistant"
        assert result["usage"]["totalTokens"] > 0
    
    def test_invalid_messages_integration(self, rag_reasoning, sample_tools):
        """잘못된 messages로 호출 시 예외 발생"""
        messages = []  # 빈 메시지
        
        with pytest.raises(ValueError, match="messages는 비어있을 수 없습니다"):
            with rag_reasoning:
                rag_reasoning.invoke(
                    messages=messages,
                    tools=sample_tools,
                    tool_choice="auto"
                )
    
    def test_invalid_tools_integration(self, rag_reasoning):
        """잘못된 tools로 호출 시 예외 발생"""
        messages = [{"role": "user", "content": "테스트"}]
        tools = []  # 빈 도구
        
        with pytest.raises(ValueError, match="tools는 비어있을 수 없습니다"):
            with rag_reasoning:
                rag_reasoning.invoke(
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
    
    def test_with_langchain_tools(self, rag_reasoning):
        """LangChain tool을 변환하여 사용하는 통합 테스트"""
        try:
            from langchain_core.tools import tool
        except ImportError:
            pytest.skip("langchain_core가 설치되지 않았습니다")
        
        # LangChain tool 정의
        @tool
        def search_ncloud_docs(query: str) -> str:
            """NCloud 문서를 검색합니다.
            
            Args:
                query: 검색할 질의
            """
            return f"Search results for: {query}"
        
        @tool
        def get_server_info(server_id: str) -> str:
            """서버 정보를 조회합니다.
            
            Args:
                server_id: 서버 ID
            """
            return f"Server info: {server_id}"
        
        # LangChain tools를 RAG Reasoning 형식으로 변환
        rag_tools = convert_langchain_tools_to_rag_reasoning([
            search_ncloud_docs,
            get_server_info
        ])
        
        messages = [
            {"role": "user", "content": "VPC에 대해 알려줘"}
        ]
        
        with rag_reasoning:
            result = rag_reasoning.invoke(
                messages=messages,
                tools=rag_tools,
                tool_choice="auto"
            )
        
        assert "message" in result
        assert "usage" in result
        assert result["message"]["role"] == "assistant"
        
        # toolCalls가 있는지 확인 (첫 번째 호출은 검색 요청할 가능성 높음)
        if "toolCalls" in result["message"]:
            tool_calls = result["message"]["toolCalls"]
            if len(tool_calls) > 0:
                # 호출된 함수 이름이 우리가 정의한 함수 중 하나인지 확인
                called_function = tool_calls[0]["function"]["name"]
                assert called_function in ["search_ncloud_docs", "get_server_info"]


if __name__ == "__main__":
    """
    직접 실행 시 간단한 통합 테스트 수행
    
    실행 방법:
        uv run python tests/test_clova_rag_reasoning.py
    """
    from dotenv import load_dotenv
    
    # 환경변수 로드
    load_dotenv()
    
    print("=" * 80)
    print("Clova Studio RAG Reasoning API 간단 테스트")
    print("=" * 80)
    
    # 1. 초기화
    print("\n[1] ClovaStudioRAGReasoning 초기화")
    print("-" * 80)
    
    try:
        settings = ClovaStudioRAGReasoningSettings()
        rag_reasoning = ClovaStudioRAGReasoning.from_settings(settings)
        print("✓ ClovaStudioRAGReasoning 초기화 완료")
        print(f"  - 엔드포인트: {settings.endpoint}")
        print(f"  - 타임아웃: {settings.request_timeout}초")
        print(f"  - Max Tokens: {settings.max_tokens}")
        print(f"  - Temperature: {settings.temperature}")
    except Exception as e:
        print(f"⚠️  RAG Reasoning 초기화 실패: {e}")
        print("   .env 파일의 CLOVASTUDIO_API_KEY를 확인하세요.")
        exit(1)
    
    # 2. 단일 턴 호출
    print("\n[2] 단일 턴 호출 테스트")
    print("-" * 80)
    
    messages = [
        {"role": "user", "content": "VPC 삭제 방법 알려줘"}
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "ncloud_cs_retrieval",
                "description": "NCloud 관련 검색을 할 때 사용하는 도구입니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "사용자의 검색어를 정제해서 넣으세요."
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    try:
        with rag_reasoning:
            result = rag_reasoning.invoke(
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
        
        print("\n✓ API 호출 완료")
        print(f"  - 프롬프트 토큰: {result['usage']['promptTokens']}")
        print(f"  - 생성 토큰: {result['usage']['completionTokens']}")
        print(f"  - 전체 토큰: {result['usage']['totalTokens']}")
        
        message = result["message"]
        print(f"\n  응답:")
        print(f"    - Role: {message['role']}")
        
        if "thinkingContent" in message and message["thinkingContent"]:
            print(f"    - Thinking: {message['thinkingContent'][:100]}...")
        
        if "toolCalls" in message and len(message["toolCalls"]) > 0:
            print(f"    - Tool Calls: {len(message['toolCalls'])}개")
            for idx, tool_call in enumerate(message["toolCalls"], 1):
                print(f"      [{idx}] {tool_call['function']['name']}")
                print(f"          Arguments: {tool_call['function']['arguments']}")
        
        if message.get("content"):
            print(f"    - Content: {message['content'][:200]}...")
    
    except Exception as e:
        print(f"⚠️  API 호출 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✅ 간단 테스트 완료")
    print("=" * 80)

