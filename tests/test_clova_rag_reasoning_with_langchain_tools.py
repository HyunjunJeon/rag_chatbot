"""
LangChain Tool을 사용한 RAG Reasoning 예제

이 파일은 LangChain의 @tool 데코레이터로 정의한 함수를
Clova Studio RAG Reasoning API에서 사용하는 방법을 보여줍니다.

실행 방법:
    uv run python tests/test_clova_rag_reasoning_with_langchain_tools.py
"""

from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

print("=" * 80)
print("LangChain Tool을 사용한 RAG Reasoning 예제")
print("=" * 80)

# ============================================================================
# 1. LangChain Tool 정의
# ============================================================================
print("\n[1] LangChain Tool 정의")
print("-" * 80)

try:
    from langchain_core.tools import tool
    from pydantic import BaseModel, Field
except ImportError:
    print("⚠️  langchain_core가 설치되지 않았습니다.")
    print("   설치 명령: uv add langchain-core")
    exit(1)


# 간단한 tool (파라미터만 있는 경우)
@tool
def search_ncloud_docs(query: str) -> str:
    """NCloud 문서를 검색합니다.
    
    이 도구는 NCloud 관련 문서를 검색하여 관련 정보를 반환합니다.
    
    Args:
        query: 검색할 질의
    
    Returns:
        검색 결과
    """
    # 실제로는 여기서 검색을 수행
    return f"[검색 결과] {query}에 대한 문서를 찾았습니다."


# 복잡한 tool (Pydantic BaseModel 사용)
class ServerQueryInput(BaseModel):
    """서버 조회 입력 스키마"""
    server_type: str = Field(description="서버 타입 (예: GPU, CPU)")
    region: str = Field(default="KR-1", description="리전 (기본값: KR-1)")
    include_pricing: bool = Field(default=False, description="가격 정보 포함 여부")


@tool(args_schema=ServerQueryInput)
def get_server_info(server_type: str, region: str = "KR-1", include_pricing: bool = False) -> str:
    """NCloud 서버 정보를 조회합니다.
    
    지정된 서버 타입과 리전의 서버 정보를 조회합니다.
    
    Args:
        server_type: 서버 타입 (예: GPU, CPU)
        region: 리전 (기본값: KR-1)
        include_pricing: 가격 정보 포함 여부 (기본값: False)
    
    Returns:
        서버 정보
    """
    # 실제로는 여기서 API 호출
    result = f"[서버 정보] {server_type} 서버 ({region})"
    if include_pricing:
        result += " - 가격 정보 포함"
    return result


print("✓ LangChain Tool 정의 완료")
print(f"  - search_ncloud_docs: {search_ncloud_docs.name}")
print(f"  - get_server_info: {get_server_info.name}")

# ============================================================================
# 2. LangChain Tool을 RAG Reasoning 형식으로 변환
# ============================================================================
print("\n[2] LangChain Tool을 RAG Reasoning 형식으로 변환")
print("-" * 80)

from naver_connect_chatbot.rag import (
    convert_langchain_tool_to_rag_reasoning,
    convert_langchain_tools_to_rag_reasoning,
)

# 단일 tool 변환
search_tool_rag = convert_langchain_tool_to_rag_reasoning(search_ncloud_docs)
print("\n✓ 단일 Tool 변환 완료:")
print(f"  - Type: {search_tool_rag['type']}")
print(f"  - Name: {search_tool_rag['function']['name']}")
print(f"  - Description: {search_tool_rag['function']['description'][:50]}...")

# 여러 tools 일괄 변환
rag_tools = convert_langchain_tools_to_rag_reasoning([
    search_ncloud_docs,
    get_server_info
])

print(f"\n✓ 여러 Tools 일괄 변환 완료: {len(rag_tools)}개")
for idx, tool in enumerate(rag_tools, 1):
    print(f"  [{idx}] {tool['function']['name']}")
    print(f"      - Required params: {tool['function']['parameters'].get('required', [])}")
    print(f"      - All params: {list(tool['function']['parameters']['properties'].keys())}")

# ============================================================================
# 3. RAG Reasoning API 호출
# ============================================================================
print("\n[3] RAG Reasoning API 호출")
print("-" * 80)

from naver_connect_chatbot.config.settings.clova import ClovaStudioRAGReasoningSettings
from naver_connect_chatbot.rag import ClovaStudioRAGReasoning

try:
    settings = ClovaStudioRAGReasoningSettings()
    rag_reasoning = ClovaStudioRAGReasoning.from_settings(settings)
    
    messages = [
        {"role": "user", "content": "A100 GPU 서버를 KR-1 리전에서 사용하려면 어떻게 해야 하나요? 가격도 알려주세요."}
    ]
    
    print(f"질문: {messages[0]['content']}")
    print("\nAPI 호출 중...")
    
    with rag_reasoning:
        result = rag_reasoning.invoke(
            messages=messages,
            tools=rag_tools,
            tool_choice="auto"
        )
    
    print("\n✓ API 호출 완료")
    print(f"  - 프롬프트 토큰: {result['usage']['promptTokens']}")
    print(f"  - 생성 토큰: {result['usage']['completionTokens']}")
    print(f"  - 전체 토큰: {result['usage']['totalTokens']}")
    
    message = result["message"]
    
    if "thinkingContent" in message and message["thinkingContent"]:
        print(f"\n  [Thinking]")
        print(f"    {message['thinkingContent'][:150]}...")
    
    if "toolCalls" in message and len(message["toolCalls"]) > 0:
        print(f"\n  [Tool Calls] {len(message['toolCalls'])}개")
        for idx, tool_call in enumerate(message["toolCalls"], 1):
            print(f"    [{idx}] {tool_call['function']['name']}")
            print(f"        ID: {tool_call['id']}")
            print(f"        Arguments: {tool_call['function']['arguments']}")
        
        # Step 2로 진행 가능 (검색 수행 후 재호출)
        print("\n  💡 다음 단계: 실제 검색을 수행하고 결과를 'tool' 역할로 추가하여 재호출하면")
        print("     모델이 검색 결과를 바탕으로 최종 답변을 생성합니다.")
    
    if message.get("content"):
        print(f"\n  [Content]")
        print(f"    {message['content'][:200]}...")

except Exception as e:
    print(f"⚠️  API 호출 실패: {e}")
    print("   .env 파일의 CLOVASTUDIO_API_KEY를 확인하세요.")
    import traceback
    traceback.print_exc()

# ============================================================================
# 4. 변환된 Tool 형식 상세 출력
# ============================================================================
print("\n[4] 변환된 Tool 형식 상세")
print("-" * 80)

import json

print("\n✓ search_ncloud_docs 변환 결과:")
print(json.dumps(search_tool_rag, indent=2, ensure_ascii=False))

print("\n✓ get_server_info 변환 결과:")
print(json.dumps(rag_tools[1], indent=2, ensure_ascii=False))

# ============================================================================
# 완료
# ============================================================================
print("\n" + "=" * 80)
print("✅ 예제 실행 완료")
print("=" * 80)
print("\n📚 추가 정보:")
print("  - LangChain tool을 정의하면 자동으로 RAG Reasoning 형식으로 변환 가능")
print("  - 복잡한 파라미터도 Pydantic BaseModel로 쉽게 정의")
print("  - 기존 LangChain 생태계의 도구들을 그대로 활용 가능")
print()

