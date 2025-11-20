"""
Adaptive RAG 워크플로에서 사용하는 노드 함수 집합.

각 노드는 RAG 프로세스의 개별 단계를 나타냅니다.
"""

from typing import Any, Dict
from langchain_core.runnables import Runnable
from langchain_core.retrievers import BaseRetriever

from naver_connect_chatbot.service.graph.state import AdaptiveRAGState
from naver_connect_chatbot.service.agents import (
    create_intent_classifier,
    create_query_analyzer,
    create_document_evaluator,
    create_answer_generator,
    create_answer_validator,
    create_corrector,
)
from naver_connect_chatbot.service.agents.answer_generator import get_generation_strategy
from naver_connect_chatbot.service.tool import retrieve_documents_async
from naver_connect_chatbot.config import logger


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
        response = await classifier.ainvoke({
            "messages": [{"role": "user", "content": question}]
        })
        
        # 분류 결과를 추출합니다.
        if hasattr(response, "messages") and response.messages:
            last_msg = response.messages[-1]
            if hasattr(last_msg, "content"):
                result = last_msg.content
                return {
                    "intent": result.intent if hasattr(result, "intent") else "SIMPLE_QA",
                    "intent_confidence": result.confidence if hasattr(result, "confidence") else 0.8,
                    "intent_reasoning": result.reasoning if hasattr(result, "reasoning") else "",
                }
        
        # 폴백
        logger.warning("Could not extract intent from response, using default")
        return {
            "intent": "SIMPLE_QA",
            "intent_confidence": 0.8,
            "intent_reasoning": "Default classification"
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
        response = await analyzer.ainvoke({
            "messages": [{
                "role": "user",
                "content": f"question: {question}\nintent: {intent}"
            }]
        })
        
        # 분석 결과를 추출합니다.
        if hasattr(response, "messages") and response.messages:
            last_msg = response.messages[-1]
            if hasattr(last_msg, "content"):
                result = last_msg.content
                improved = result.improved_queries if hasattr(result, "improved_queries") else [question]
                return {
                    "query_analysis": {
                        "clarity_score": getattr(result, "clarity_score", 0.8),
                        "specificity_score": getattr(result, "specificity_score", 0.8),
                        "searchability_score": getattr(result, "searchability_score", 0.8),
                    },
                    "refined_queries": improved if improved else [question],
                    "original_query": question,
                }
        
        # 폴백
        return {
            "query_analysis": {"clarity_score": 0.8, "specificity_score": 0.8, "searchability_score": 0.8},
            "refined_queries": [question],
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
        response = await evaluator.ainvoke({
            "messages": [{
                "role": "user",
                "content": f"question: {question}\n\ndocuments:\n{doc_text}"
            }]
        })
        
        # 평가 결과를 추출합니다.
        if hasattr(response, "messages") and response.messages:
            last_msg = response.messages[-1]
            if hasattr(last_msg, "content"):
                result = last_msg.content
                return {
                    "document_evaluation": {
                        "relevant_count": getattr(result, "relevant_count", len(documents)),
                        "irrelevant_count": getattr(result, "irrelevant_count", 0),
                        "confidence": getattr(result, "confidence", 0.8),
                    },
                    "sufficient_context": getattr(result, "sufficient", True),
                    "relevant_doc_count": getattr(result, "relevant_count", len(documents)),
                }
        
        # 폴백: 문서가 있으면 충분하다고 가정합니다.
        return {
            "document_evaluation": {"relevant_count": len(documents), "irrelevant_count": 0, "confidence": 0.7},
            "sufficient_context": len(documents) > 0,
            "relevant_doc_count": len(documents),
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
        generator = create_answer_generator(llm, strategy=strategy)
        
        # 생성에 사용할 문맥을 포맷합니다.
        context_text = "\n\n".join([
            f"[문서 {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(documents)
        ])
        
        if not context_text:
            context_text = "참고할 수 있는 문서가 없습니다."
        
        # 답변을 생성합니다.
        response = await generator.ainvoke({
            "messages": [{
                "role": "user",
                "content": f"question: {question}\n\ncontext:\n{context_text}"
            }]
        })
        
        # 생성된 답변을 추출합니다.
        answer = ""
        if hasattr(response, "messages") and response.messages:
            last_msg = response.messages[-1]
            if hasattr(last_msg, "content"):
                answer = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
        
        if not answer:
            answer = "죄송합니다. 답변을 생성할 수 없습니다."
        
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
        response = await validator.ainvoke({
            "messages": [{
                "role": "user",
                "content": f"question: {question}\n\ncontext:\n{context_text}\n\nanswer:\n{answer}"
            }]
        })
        
        # 검증 결과를 추출합니다.
        if hasattr(response, "messages") and response.messages:
            last_msg = response.messages[-1]
            if hasattr(last_msg, "content"):
                result = last_msg.content
                return {
                    "validation_result": {
                        "has_hallucination": getattr(result, "has_hallucination", False),
                        "is_grounded": getattr(result, "is_grounded", True),
                        "is_complete": getattr(result, "is_complete", True),
                        "quality_score": getattr(result, "quality_score", 0.8),
                        "issues": getattr(result, "issues", []),
                    },
                    "has_hallucination": getattr(result, "has_hallucination", False),
                    "is_grounded": getattr(result, "is_grounded", True),
                    "is_complete": getattr(result, "is_complete", True),
                    "quality_score": getattr(result, "quality_score", 0.8),
                    "validation_issues": getattr(result, "issues", []),
                }
        
        # 폴백: 정상으로 간주합니다.
        return {
            "validation_result": {"has_hallucination": False, "is_grounded": True, "is_complete": True, "quality_score": 0.8},
            "has_hallucination": False,
            "is_grounded": True,
            "is_complete": True,
            "quality_score": 0.8,
            "validation_issues": [],
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
        response = await corrector.ainvoke({
            "messages": [{
                "role": "user",
                "content": f"validation_result:\n{validation_text}\n\nanswer:\n{answer}"
            }]
        })
        
        # 교정 결과를 추출합니다.
        if hasattr(response, "messages") and response.messages:
            last_msg = response.messages[-1]
            if hasattr(last_msg, "content"):
                result = last_msg.content
                action = getattr(result, "action", "REGENERATE")
                feedback = getattr(result, "feedback", "Please improve the answer")
                
                return {
                    "correction_action": action,
                    "correction_feedback": state.get("correction_feedback", []) + [feedback],
                    "correction_count": correction_count + 1,
                }
        
        # 폴백
        return {
            "correction_action": "REGENERATE",
            "correction_feedback": state.get("correction_feedback", []) + ["No specific feedback available"],
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
