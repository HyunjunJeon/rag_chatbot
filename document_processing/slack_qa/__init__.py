"""
Slack Q&A 처리 모듈

Slack 채널의 Q&A 데이터를 로드, 처리, 필터링, 품질 평가하는 기능을 제공합니다.
"""

from .slack_qa_loader import SlackQALoader, QAPair, Message
from .filter_qa_data import filter_qa_pairs, should_remove_question, should_remove_answer
from .quality_schemas import (
    DimensionScore,
    QualityEvaluation,
    EvaluationInput,
    extract_for_evaluation,
)
from .quality_evaluator import QualityEvaluator
from .batch_processor import BatchConfig, ProcessingState, BatchProcessor

__all__ = [
    # Loader
    "SlackQALoader",
    "QAPair",
    "Message",
    # Filter
    "filter_qa_pairs",
    "should_remove_question",
    "should_remove_answer",
    # Quality Evaluation
    "DimensionScore",
    "QualityEvaluation",
    "EvaluationInput",
    "extract_for_evaluation",
    "QualityEvaluator",
    "BatchConfig",
    "ProcessingState",
    "BatchProcessor",
]
