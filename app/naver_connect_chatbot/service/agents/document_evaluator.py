"""
Adaptive RAG용 문서 평가 에이전트 구현.

검색된 문서의 관련성과 충분성을 판별합니다.
"""

from typing import Annotated, Any
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain.agents import create_agent

from naver_connect_chatbot.prompts import get_prompt
from naver_connect_chatbot.config import logger


class DocumentEvaluation(BaseModel):
    """
    문서 관련성과 충분성 평가 결과를 담는 모델입니다.
    
    속성:
        relevant_count: 관련 문서 개수
        irrelevant_count: 비관련 문서 개수
        sufficient: 문서가 답변에 충분한지 여부
        confidence: 평가 신뢰도 (0.0 ~ 1.0)
        improvement_suggestions: 검색 개선 제안
    """
    relevant_count: int = Field(
        description="Number of relevant documents",
        ge=0
    )
    irrelevant_count: int = Field(
        description="Number of irrelevant documents",
        ge=0
    )
    sufficient: bool = Field(
        description="Whether documents are sufficient to answer the question"
    )
    confidence: float = Field(
        description="Confidence in the evaluation (0.0 ~ 1.0)",
        ge=0.0,
        le=1.0
    )
    improvement_suggestions: list[str] = Field(
        description="Suggestions for improving retrieval",
        default_factory=list
    )


@tool(args_schema=DocumentEvaluation)
def emit_document_evaluation(
    relevant_count: Annotated[int, Field(description="Number of relevant documents", ge=0)],
    irrelevant_count: Annotated[int, Field(description="Number of irrelevant documents", ge=0)],
    sufficient: Annotated[bool, Field(description="Whether documents are sufficient to answer the question")],
    confidence: Annotated[float, Field(description="Confidence in the evaluation (0.0 ~ 1.0)", ge=0.0, le=1.0)],
    improvement_suggestions: Annotated[list[str], Field(description="Suggestions for improving retrieval")],
) -> DocumentEvaluation:
    """
    Emit structured document evaluation results.
    
    Call this tool after evaluating retrieved documents to return assessment
    of relevance and sufficiency.
    
    Returns:
        DocumentEvaluation instance with all evaluation results
    """
    return DocumentEvaluation(
        relevant_count=relevant_count,
        irrelevant_count=irrelevant_count,
        sufficient=sufficient,
        confidence=confidence,
        improvement_suggestions=improvement_suggestions,
    )


def create_document_evaluator(llm: Runnable) -> Any:
    """
    문서 평가 에이전트를 생성합니다.
    
    다음 기준으로 문서를 평가합니다.
    - 사용자 질문과의 관련성
    - 충분한 답변 제공 여부
    - 정보의 품질과 명확성
    
    매개변수:
        llm: 평가에 사용할 언어 모델
    
    반환값:
        문서를 평가하는 에이전트
        
    예시:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4o-mini")
        >>> evaluator = create_document_evaluator(llm)
        >>> docs = [Document(page_content="PyTorch is a deep learning framework")]
        >>> result = evaluator.invoke({
        ...     "messages": [{
        ...         "role": "user",
        ...         "content": f"question: What is PyTorch?\ndocuments: {docs}"
        ...     }]
        ... })
    """
    try:        
        prompt_template = get_prompt("document_evaluation")
        system_prompt = prompt_template.messages[0].prompt.template if prompt_template.messages else ""
        
        # 도구 사용 가이드 추가
        enhanced_prompt = (
            f"{system_prompt}\n\n"
            "IMPORTANT: After evaluating the documents, you MUST call the emit_document_evaluation tool "
            "with all required parameters (relevant_count, irrelevant_count, sufficient, confidence, "
            "improvement_suggestions) to return your evaluation."
        )
        
        agent = create_agent(
            model=llm,
            tools=[emit_document_evaluation],
            system_prompt=enhanced_prompt,
            name="document_evaluator",
        )
        
        logger.debug("Document evaluator agent created successfully with emit_document_evaluation tool")
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create document evaluator: {e}")
        raise


def evaluate_documents(
    question: str,
    documents: list[Document],
    llm: Runnable
) -> DocumentEvaluation:
    """
    단일 질문에 대해 문서를 평가하는 편의 함수입니다.
    
    매개변수:
        question: 사용자 질문
        documents: 평가할 검색 문서
        llm: 사용할 언어 모델
    
    반환값:
        DocumentEvaluation 평가 결과
        
    예시:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_core.documents import Document
        >>> llm = ChatOpenAI(model="gpt-4o-mini")
        >>> docs = [Document(page_content="PyTorch is a framework")]
        >>> result = evaluate_documents("What is PyTorch?", docs, llm)
        >>> print(result.sufficient)
        True
    """
    evaluator = create_document_evaluator(llm)
    
    # 평가에 사용할 문서를 포맷합니다.
    doc_text = "\n\n".join([
        f"Document {i+1}:\n{doc.page_content[:500]}..."  # Limit length
        for i, doc in enumerate(documents)
    ])
    
    response = evaluator.invoke({
        "messages": [{
            "role": "user",
            "content": f"question: {question}\n\ndocuments:\n{doc_text}"
        }]
    })
    
    # 에이전트 응답에서 ToolMessage를 찾아 DocumentEvaluation을 추출합니다.
    try:
        if isinstance(response, dict) and "messages" in response:
            messages = response["messages"]
            for msg in reversed(messages):
                is_tool_msg = (
                    msg.__class__.__name__ == "ToolMessage" or
                    (hasattr(msg, "type") and msg.type == "tool")
                )
                
                if is_tool_msg:
                    tool_content = msg.content
                    if isinstance(tool_content, DocumentEvaluation):
                        logger.debug("Successfully extracted DocumentEvaluation from ToolMessage")
                        return tool_content
                    elif isinstance(tool_content, dict):
                        logger.debug("Converting dict content to DocumentEvaluation")
                        return DocumentEvaluation(**tool_content)
                    elif isinstance(tool_content, str):
                        try:
                            logger.debug("Attempting to parse string content as DocumentEvaluation")
                            import re
                            import ast
                            data = {}
                            
                            # 숫자 필드 추출
                            for field in ['relevant_count', 'irrelevant_count']:
                                match = re.search(rf'{field}=(\d+)', tool_content)
                                if match:
                                    data[field] = int(match.group(1))
                            
                            # confidence 추출
                            conf_match = re.search(r'confidence=([\d.]+)', tool_content)
                            if conf_match:
                                data['confidence'] = float(conf_match.group(1))
                            
                            # sufficient 추출
                            suff_match = re.search(r'sufficient=(True|False)', tool_content)
                            if suff_match:
                                data['sufficient'] = suff_match.group(1) == 'True'
                            
                            # improvement_suggestions 추출
                            sugg_match = re.search(r'improvement_suggestions=(\[.*?\](?=\s+\w+=|$))', tool_content, re.DOTALL)
                            if sugg_match:
                                try:
                                    data['improvement_suggestions'] = ast.literal_eval(sugg_match.group(1))
                                except Exception:
                                    data['improvement_suggestions'] = []
                            
                            if data:
                                logger.debug(f"Successfully parsed DocumentEvaluation from string: {data}")
                                return DocumentEvaluation(**data)
                        except Exception as e:
                            logger.warning(f"Failed to parse string content: {e}")
        
        if isinstance(response, DocumentEvaluation):
            return response
        
        logger.warning(f"Unable to extract DocumentEvaluation from response: {type(response)}")
        
    except Exception as e:
        logger.error(f"Error extracting DocumentEvaluation from response: {e}")
    
    # 최종 폴백
    return DocumentEvaluation(
        relevant_count=len(documents),
        irrelevant_count=0,
        sufficient=len(documents) > 0,
        confidence=0.5,
        improvement_suggestions=["Unable to evaluate documents properly"]
    )

