"""
Reflection and strategic thinking tools for Adaptive RAG.

These tools enable agents to pause and reflect on their progress,
creating deliberate decision-making points in the workflow.
"""

from typing import Annotated
from pydantic import Field
from langchain_core.tools import tool


@tool(description="Strategic reflection tool for research and decision-making")
def think_tool(
    reflection: Annotated[str, Field(description="Your detailed reflection on progress, findings, gaps, and next steps")],
) -> str:
    """
    Tool for strategic reflection on research progress and decision-making.
    
    Use this tool after each major step to analyze results and plan next actions systematically.
    This creates a deliberate pause in the workflow for quality decision-making.
    
    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding: Can I provide a complete answer now?
    
    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?
    
    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps
    
    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"


@tool(description="Document analysis and assessment tool")
def analyze_documents(
    analysis: Annotated[str, Field(description="Your analysis of document quality, relevance, and gaps")],
) -> str:
    """
    Analyze retrieved documents for quality, relevance, and information gaps.
    
    Use this tool to systematically evaluate search results before proceeding.
    
    Analysis should include:
    1. Relevance assessment - How well do documents match the query?
    2. Coverage analysis - What aspects of the question are covered?
    3. Gap identification - What important information is missing?
    4. Quality evaluation - Are sources reliable and informative?
    
    Args:
        analysis: Your detailed analysis of the retrieved documents
    
    Returns:
        Confirmation that analysis was recorded
    """
    return f"Document analysis recorded: {analysis}"


@tool(description="Answer quality self-evaluation tool")
def evaluate_answer_quality(
    evaluation: Annotated[str, Field(description="Your evaluation of answer completeness, accuracy, and quality")],
) -> str:
    """
    Evaluate the quality of a generated answer before finalizing.
    
    Use this tool for self-assessment of answer quality.
    
    Evaluation should cover:
    1. Completeness - Does it fully answer the question?
    2. Accuracy - Is all information grounded in context?
    3. Clarity - Is the explanation clear and well-structured?
    4. Evidence - Are claims properly supported?
    
    Args:
        evaluation: Your detailed quality evaluation of the answer
    
    Returns:
        Confirmation that evaluation was recorded
    """
    return f"Answer evaluation recorded: {evaluation}"


__all__ = [
    "think_tool",
    "analyze_documents",
    "evaluate_answer_quality",
]

