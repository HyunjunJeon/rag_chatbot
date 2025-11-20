"""
Adaptive RAG 워크플로에서 사용하는 노드 함수 집합.

각 노드는 RAG 프로세스의 개별 단계를 나타냅니다.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Type, TypeVar, List

from pydantic import BaseModel
from langchain_core.runnables import Runnable
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import AIMessage

from naver_connect_chatbot.service.graph.state import AdaptiveRAGState
from naver_connect_chatbot.service.agents import (
    create_intent_classifier,
    IntentClassification,
    create_query_analyzer,
    QueryAnalysis,
    create_document_evaluator,
    DocumentEvaluation,
    create_answer_generator,
    create_answer_validator,
    create_corrector,
    CorrectionStrategy,
    AnswerValidation,
)
from naver_connect_chatbot.service.agents.answer_generator import get_generation_strategy
from naver_connect_chatbot.service.tool import retrieve_documents_async
from naver_connect_chatbot.config import logger

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _coerce_model_response(model_type: Type[_ModelT], response: Any) -> _ModelT:
    """
    LangChain 에이전트 응답을 지정된 Pydantic 모델로 강제 변환합니다.

    에이전트가 dict, 문자열(JSON), BaseModel, AIMessage 등을 반환해도
    일관된 모델 객체를 돌려줍니다.
    """
    if isinstance(response, model_type):
        return response

    if isinstance(response, BaseModel):
        return model_type.model_validate(response.model_dump())

    if isinstance(response, dict):
        # 에이전트 응답에서 ToolMessage를 찾아 추출합니다.
        if "messages" in response:
            messages = response["messages"]
            for msg in reversed(messages):
                # ToolMessage인지 확인
                is_tool_msg = (
                    msg.__class__.__name__ == "ToolMessage" or
                    (hasattr(msg, "type") and msg.type == "tool")
                )
                
                if is_tool_msg:
                    tool_content = msg.content
                    if isinstance(tool_content, model_type):
                        return tool_content
                    elif isinstance(tool_content, BaseModel):
                        return model_type.model_validate(tool_content.model_dump())
                    elif isinstance(tool_content, dict):
                        return model_type.model_validate(tool_content)
                    elif isinstance(tool_content, str):
                        try:
                            # JSON 문자열 시도
                            data = json.loads(tool_content)
                            return model_type.model_validate(data)
                        except json.JSONDecodeError:
                            # Python 표현식 시도
                            try:
                                import re
                                import ast
                                # 모델의 모든 필드를 추출 시도
                                data = {}
                                for field_name, field_info in model_type.model_fields.items():
                                    # 리스트 필드는 특별 처리
                                    is_list_field = (
                                        hasattr(field_info.annotation, '__origin__') and
                                        field_info.annotation.__origin__ in (list, List)
                                    )
                                    
                                    if is_list_field:
                                        # 리스트 패턴: field_name=[...] (다음 필드 또는 끝까지)
                                        pattern = rf'{field_name}=(\[.*?\])(?=\s+\w+=|$)'
                                        match = re.search(pattern, tool_content, re.DOTALL)
                                        if match:
                                            value_str = match.group(1).strip()
                                            try:
                                                data[field_name] = ast.literal_eval(value_str)
                                            except (ValueError, SyntaxError):
                                                pass  # 파싱 실패시 건너뜀
                                    else:
                                        # 일반 필드: field_name=value (공백이나 줄바꿈 전까지)
                                        pattern = rf'{field_name}=([^\s]+)'
                                        match = re.search(pattern, tool_content)
                                        if match:
                                            value_str = match.group(1).strip()
                                            try:
                                                data[field_name] = ast.literal_eval(value_str)
                                            except (ValueError, SyntaxError):
                                                # 문자열로 처리
                                                if value_str.lower() == 'true':
                                                    data[field_name] = True
                                                elif value_str.lower() == 'false':
                                                    data[field_name] = False
                                                else:
                                                    data[field_name] = value_str
                                
                                if data:
                                    return model_type.model_validate(data)
                            except Exception as e:
                                logger.debug(f"Failed to parse tool content using regex: {e}")
                            
                            raise ValueError(
                                f"Unable to parse ToolMessage content into {model_type.__name__}: {tool_content}"
                            )
        
        if "output" in response:
            output_data = response["output"]
            if isinstance(output_data, BaseModel):
                return model_type.model_validate(output_data.model_dump())
            if isinstance(output_data, dict):
                return model_type.model_validate(output_data)
            if isinstance(output_data, str):
                try:
                    data = json.loads(output_data)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Unable to parse agent output into {model_type.__name__}: {output_data}"
                    ) from exc
                return model_type.model_validate(data)

        return model_type.model_validate(response)

    # LangChain 메시지 또는 기타 객체에서 content 추출
    content = getattr(response, "content", None)
    if isinstance(content, BaseModel):
        return model_type.model_validate(content.model_dump())
    if isinstance(content, dict):
        return model_type.model_validate(content)
    if isinstance(content, str):
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise ValueError(
                f"Unable to parse JSON content into {model_type.__name__}: {content}"
            ) from exc
        return model_type.model_validate(data)

    if isinstance(response, str):
        try:
            data = json.loads(response)
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise ValueError(
                f"Unable to parse JSON string into {model_type.__name__}: {response}"
            ) from exc
        return model_type.model_validate(data)

    raise ValueError(
        f"Agent response could not be parsed into {model_type.__name__}: {type(response)}"
    )


def _extract_text_response(response: Any) -> str:
    """
    LangChain 에이전트 응답에서 텍스트를 안전하게 추출합니다.
    """
    if isinstance(response, AIMessage):
        if isinstance(response.content, str):
            return response.content
        return str(response.content)

    if hasattr(response, "content"):
        content = response.content  # type: ignore[attr-defined]
        if isinstance(content, str):
            return content
        return str(content)

    if isinstance(response, dict):
        if isinstance(response.get("output"), str):
            return response["output"]  # type: ignore[index]
        if isinstance(response.get("content"), str):
            return response["content"]  # type: ignore[index]

    if isinstance(response, str):
        return response

    return str(response)


async def classify_intent_node(
    state: AdaptiveRAGState,
    llm: Runnable
) -> Dict[str, Any]:
    """
    사용자 의도를 분류합니다.
    
    매개변수:
        state: 현재 워크플로 상태
        llm: 분류에 사용할 언어 모델
    
    반환값:
        의도 분류 결과가 포함된 상태 업데이트
    """
    logger.info("---CLASSIFY INTENT---")
    question = state["question"]
    
    try:
        # 의도 분류 에이전트를 생성합니다.
        classifier = create_intent_classifier(llm)
        
        # 의도를 분류합니다.
        response_raw = await classifier.ainvoke({
            "messages": [{"role": "user", "content": question}]
        })

        response = _coerce_model_response(IntentClassification, response_raw)
        
        # 분류 결과를 추출합니다.
        return {
            "intent": response.intent,
            "intent_confidence": response.confidence,
            "intent_reasoning": response.reasoning,
        }
        
    except Exception as e:
        logger.error(f"Intent classification error: {e}")
        return {
            "intent": "SIMPLE_QA",
            "intent_confidence": 0.5,
            "intent_reasoning": f"Error during classification: {str(e)}"
        }


async def analyze_query_node(
    state: AdaptiveRAGState,
    llm: Runnable
) -> Dict[str, Any]:
    """
    질의 품질을 분석하고 개선안을 작성합니다.
    
    매개변수:
        state: 현재 워크플로 상태
        llm: 분석에 사용할 언어 모델
    
    반환값:
        질의 분석과 개선된 질의를 포함한 상태 업데이트
    """
    logger.info("---ANALYZE QUERY---")
    question = state["question"]
    intent = state.get("intent", "SIMPLE_QA")
    
    try:
        # 질의 분석 에이전트를 생성합니다.
        analyzer = create_query_analyzer(llm)
        
        # 질의를 분석합니다.
        response_raw = await analyzer.ainvoke({
            "messages": [{
                "role": "user",
                "content": f"question: {question}\nintent: {intent}"
            }]
        })

        response = _coerce_model_response(QueryAnalysis, response_raw)
        
        # 분석 결과를 추출합니다.
        return {
            "query_analysis": {
                "clarity_score": response.clarity_score,
                "specificity_score": response.specificity_score,
                "searchability_score": response.searchability_score,
            },
            "refined_queries": response.improved_queries if response.improved_queries else [question],
            "original_query": question,
        }
        
    except Exception as e:
        logger.error(f"Query analysis error: {e}")
        return {
            "query_analysis": {"error": str(e)},
            "refined_queries": [question],
            "original_query": question,
        }


async def retrieve_node(
    state: AdaptiveRAGState,
    retriever: BaseRetriever
) -> Dict[str, Any]:
    """
    문서를 검색합니다.
    
    매개변수:
        state: 현재 워크플로 상태
        retriever: 문서 검색기
    
    반환값:
        검색된 문서를 포함한 상태 업데이트
    """
    logger.info("---RETRIEVE---")
    
    # 가능하면 정제된 질의를 사용하여 검색합니다.
    queries = state.get("refined_queries", [state["question"]])
    primary_query = queries[0] if queries else state["question"]
    
    try:
        # 문서를 검색합니다.
        documents = await retrieve_documents_async(retriever, primary_query)
        
        logger.info(f"Retrieved {len(documents)} documents")
        
        return {
            "documents": documents,
            "context": documents,  # 하위 호환성 유지를 위한 필드
            "retrieval_strategy": "hybrid",
        }
        
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return {
            "documents": [],
            "context": [],
            "retrieval_strategy": "hybrid",
        }


async def evaluate_documents_node(
    state: AdaptiveRAGState,
    llm: Runnable
) -> Dict[str, Any]:
    """
    검색된 문서를 평가합니다.
    
    매개변수:
        state: 현재 워크플로 상태
        llm: 평가에 사용할 언어 모델
    
    반환값:
        문서 평가 결과를 포함한 상태 업데이트
    """
    logger.info("---EVALUATE DOCUMENTS---")
    question = state["question"]
    documents = state.get("documents", [])
    
    if not documents:
        logger.warning("No documents to evaluate")
        return {
            "document_evaluation": {"error": "No documents"},
            "sufficient_context": False,
            "relevant_doc_count": 0,
        }
    
    try:
        # 문서 평가 에이전트를 생성합니다.
        evaluator = create_document_evaluator(llm)
        
        # 평가에 사용할 문서를 포맷합니다.
        doc_text = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content[:500]}..."
            for i, doc in enumerate(documents[:5])  # 평가 시 상위 5개까지만 사용
        ])
        
        # 평가를 수행합니다.
        response_raw = await evaluator.ainvoke({
            "messages": [{
                "role": "user",
                "content": f"question: {question}\n\ndocuments:\n{doc_text}"
            }]
        })

        response = _coerce_model_response(DocumentEvaluation, response_raw)
        
        # 평가 결과를 추출합니다.
        return {
            "document_evaluation": {
                "relevant_count": response.relevant_count,
                "irrelevant_count": response.irrelevant_count,
                "confidence": response.confidence,
            },
            "sufficient_context": response.sufficient,
            "relevant_doc_count": response.relevant_count,
        }
        
    except Exception as e:
        logger.error(f"Document evaluation error: {e}")
        return {
            "document_evaluation": {"error": str(e)},
            "sufficient_context": len(documents) > 0,
            "relevant_doc_count": len(documents),
        }


async def generate_answer_node(
    state: AdaptiveRAGState,
    llm: Runnable
) -> Dict[str, Any]:
    """
    문맥을 기반으로 답변을 생성합니다.
    
    매개변수:
        state: 현재 워크플로 상태
        llm: 생성에 사용할 언어 모델
    
    반환값:
        생성된 답변을 포함한 상태 업데이트
    """
    logger.info("---GENERATE ANSWER---")
    question = state["question"]
    documents = state.get("documents", [])
    intent = state.get("intent", "SIMPLE_QA")
    
    try:
        # 사용할 생성 전략을 결정합니다.
        strategy = get_generation_strategy(intent)
        
        # 답변 생성 에이전트를 생성합니다.
        generator: Runnable = create_answer_generator(llm, strategy=strategy)
        
        # 생성에 사용할 문맥을 포맷합니다.
        context_text = "\n\n".join([
            f"[문서 {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(documents)
        ])
        
        if not context_text:
            context_text = "참고할 수 있는 문서가 없습니다."
        
        # 답변을 생성합니다.
        response_raw = await generator.ainvoke({
            "messages": [{
                "role": "user",
                "content": f"question: {question}\n\ncontext:\n{context_text}"
            }]
        })
        
        # 생성된 답변을 추출합니다.
        answer = _extract_text_response(response_raw)
        logger.info(f"Generated answer: {answer[:100]}...")
        
        return {
            "answer": answer,
            "generation_metadata": {
                "strategy": strategy,
                "context_length": len(context_text),
            },
            "generation_strategy": strategy,
        }
        
    except Exception as e:
        logger.error(f"Answer generation error: {e}")
        return {
            "answer": f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}",
            "generation_metadata": {"error": str(e)},
            "generation_strategy": "error",
        }


async def validate_answer_node(
    state: AdaptiveRAGState,
    llm: Runnable
) -> Dict[str, Any]:
    """
    생성된 답변을 검증합니다.
    
    매개변수:
        state: 현재 워크플로 상태
        llm: 검증에 사용할 언어 모델
    
    반환값:
        검증 결과를 포함한 상태 업데이트
    """
    logger.info("---VALIDATE ANSWER---")
    question = state["question"]
    answer = state.get("answer", "")
    documents = state.get("documents", [])
    
    try:
        # 답변 검증 에이전트를 생성합니다.
        validator = create_answer_validator(llm)
        
        # 검증에 사용할 문맥을 포맷합니다.
        context_text = "\n\n".join([
            f"[문서 {i+1}]\n{doc.page_content[:500]}..."
            for i, doc in enumerate(documents[:5])
        ])
        
        # 검증을 수행합니다.
        response_raw = await validator.ainvoke({
            "messages": [{
                "role": "user",
                "content": f"question: {question}\n\ncontext:\n{context_text}\n\nanswer:\n{answer}"
            }]
        })

        response = _coerce_model_response(AnswerValidation, response_raw)
        
        # 검증 결과를 추출합니다.
        return {
            "validation_result": {
                "has_hallucination": response.has_hallucination,
                "is_grounded": response.is_grounded,
                "is_complete": response.is_complete,
                "quality_score": response.quality_score,
                "issues": response.issues,
            },
            "has_hallucination": response.has_hallucination,
            "is_grounded": response.is_grounded,
            "is_complete": response.is_complete,
            "quality_score": response.quality_score,
            "validation_issues": response.issues,
        }
        
    except Exception as e:
        logger.error(f"Answer validation error: {e}")
        return {
            "validation_result": {"error": str(e)},
            "has_hallucination": False,
            "is_grounded": True,
            "is_complete": True,
            "quality_score": 0.7,
            "validation_issues": [str(e)],
        }


async def correct_node(
    state: AdaptiveRAGState,
    llm: Runnable
) -> Dict[str, Any]:
    """
    교정 전략을 결정합니다.
    
    매개변수:
        state: 현재 워크플로 상태
        llm: 교정 분석에 사용할 언어 모델
    
    반환값:
        교정 전략이 포함된 상태 업데이트
    """
    logger.info("---CORRECT---")
    validation_result = state.get("validation_result", {})
    answer = state.get("answer", "")
    correction_count = state.get("correction_count", 0)
    
    try:
        # 교정 에이전트를 생성합니다.
        corrector = create_corrector(llm)
        
        # 검증 결과를 문자열로 정리합니다.
        validation_text = "\n".join([
            f"{key}: {value}"
            for key, value in validation_result.items()
        ])
        
        # 교정 전략을 판단합니다.
        response_raw = await corrector.ainvoke({
            "messages": [{
                "role": "user",
                "content": f"validation_result:\n{validation_text}\n\nanswer:\n{answer}"
            }]
        })

        response = _coerce_model_response(CorrectionStrategy, response_raw)
        
        # 교정 결과를 추출합니다.
        return {
            "correction_action": response.action,
            "correction_feedback": state.get("correction_feedback", []) + [response.feedback],
            "correction_count": correction_count + 1,
        }
        
    except Exception as e:
        logger.error(f"Correction error: {e}")
        return {
            "correction_action": "REGENERATE",
            "correction_feedback": state.get("correction_feedback", []) + [f"Error: {str(e)}"],
            "correction_count": correction_count + 1,
        }


def finalize_node(state: AdaptiveRAGState) -> Dict[str, Any]:
    """
    워크플로를 마무리합니다.
    
    매개변수:
        state: 현재 워크플로 상태
    
    반환값:
        종료 상태 업데이트
    """
    logger.info("---FINALIZE---")
    
    # 워크플로를 완료 상태로 표시합니다.
    return {
        "workflow_stage": "completed",
    }
