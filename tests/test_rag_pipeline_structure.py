"""
새로운 RAG 파이프라인 구조 검증 테스트 (Import 없이)

CLOVA Studio 기반으로 재구성된 RAG 파이프라인의 구조를 검증합니다.
"""

import os
import sys
from pathlib import Path


def test_removed_files():
    """
    제거되어야 할 파일들이 제거되었는지 확인합니다.
    """
    base_path = Path(__file__).parent.parent
    
    removed_files = [
        "app/naver_connect_chatbot/service/tool/reflection.py",
        "app/naver_connect_chatbot/service/agents/answer_validator.py",
        "app/naver_connect_chatbot/service/agents/corrector.py",
        "app/naver_connect_chatbot/prompts/templates/answer_validation.yaml",
        "app/naver_connect_chatbot/prompts/templates/correction.yaml",
    ]
    
    for file_path in removed_files:
        full_path = base_path / file_path
        assert not full_path.exists(), \
            f"File {file_path} should be removed but still exists"
    
    print("✓ All deprecated files removed")


def test_new_files_exist():
    """
    새로 추가되어야 할 기능들이 있는지 확인합니다.
    """
    base_path = Path(__file__).parent.parent
    
    # Segmentation과 Rerank는 이미 존재하므로 확인
    required_files = [
        "app/naver_connect_chatbot/rag/segmentation.py",
        "app/naver_connect_chatbot/rag/rerank.py",
        "app/naver_connect_chatbot/rag/summarization.py",
    ]
    
    for file_path in required_files:
        full_path = base_path / file_path
        assert full_path.exists(), \
            f"Required file {file_path} does not exist"
    
    print("✓ All required files exist")


def test_nodes_file_structure():
    """
    nodes.py 파일의 구조를 확인합니다.
    """
    base_path = Path(__file__).parent.parent
    nodes_file = base_path / "app/naver_connect_chatbot/service/graph/nodes.py"
    
    assert nodes_file.exists(), "nodes.py file should exist"
    
    with open(nodes_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 새로운 노드가 있는지 확인
    assert "rerank_node" in content, "rerank_node should be defined"
    assert "segment_documents_node" in content, "segment_documents_node should be defined"
    
    # 제거된 노드가 없는지 확인
    assert "validate_answer_node" not in content or "validate_answer_node" in content and "removed" in content.lower(), \
        "validate_answer_node should be removed or deprecated"
    assert "correct_node" not in content or "correct_node" in content and "removed" in content.lower(), \
        "correct_node should be removed or deprecated"
    
    # Reasoning 관련 주석이 있는지 확인
    assert "Reasoning" in content or "reasoning" in content, \
        "nodes.py should mention Reasoning capabilities"
    
    print("✓ nodes.py structure is correct")


def test_workflow_file_structure():
    """
    workflow.py 파일의 구조를 확인합니다.
    """
    base_path = Path(__file__).parent.parent
    workflow_file = base_path / "app/naver_connect_chatbot/service/graph/workflow.py"
    
    assert workflow_file.exists(), "workflow.py file should exist"
    
    with open(workflow_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 새로운 노드 import 확인
    assert "rerank_node" in content, "rerank_node should be imported"
    assert "segment_documents_node" in content, "segment_documents_node should be imported"
    
    # 제거된 import 확인
    assert "validate_answer_node" not in content, "validate_answer_node should not be imported"
    assert "correct_node" not in content, "correct_node should not be imported"
    assert "check_answer_quality" not in content, "check_answer_quality should not be imported"
    assert "route_after_correction" not in content, "route_after_correction should not be imported"
    
    # 새로운 워크플로 구조 확인
    assert '"rerank"' in content and 'add_node' in content, "rerank node should be added to workflow"
    assert '"segment_documents"' in content and 'add_node' in content, "segment_documents node should be added"
    
    # 단순화된 엣지 확인
    assert 'workflow.add_edge("retrieve", "rerank")' in content, "retrieve -> rerank edge should exist"
    assert 'workflow.add_edge("rerank", "segment_documents")' in content, "rerank -> segment_documents edge should exist"
    assert 'workflow.add_edge("generate_answer", "finalize")' in content, \
        "generate_answer -> finalize direct edge should exist (no validation/correction)"
    
    print("✓ workflow.py structure is correct")


def test_settings_file_updates():
    """
    rag_settings.py 파일의 업데이트를 확인합니다.
    """
    base_path = Path(__file__).parent.parent
    settings_file = base_path / "app/naver_connect_chatbot/config/settings/rag_settings.py"
    
    assert settings_file.exists(), "rag_settings.py file should exist"
    
    with open(settings_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 새로운 설정 필드 확인
    assert "use_segmentation" in content, "use_segmentation field should exist"
    assert "segmentation_threshold" in content, "segmentation_threshold field should exist"
    
    # 비활성화된 설정 확인
    assert "enable_answer_validation" in content, "enable_answer_validation field should exist"
    assert "enable_correction" in content, "enable_correction field should exist"
    
    # default=False로 설정되어 있는지 확인
    # (enable_answer_validation과 enable_correction)
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "enable_answer_validation" in line and "Field" in line:
            # 다음 몇 줄에서 default=False 찾기
            for j in range(i, min(i+5, len(lines))):
                if "default=False" in lines[j]:
                    break
            else:
                raise AssertionError("enable_answer_validation should default to False")
        
        if "enable_correction" in line and "Field" in line:
            # 다음 몇 줄에서 default=False 찾기
            for j in range(i, min(i+5, len(lines))):
                if "default=False" in lines[j]:
                    break
            else:
                raise AssertionError("enable_correction should default to False")
    
    print("✓ rag_settings.py updates are correct")


def test_prompts_updated():
    """
    프롬프트 템플릿이 업데이트되었는지 확인합니다.
    """
    base_path = Path(__file__).parent.parent
    prompts_dir = base_path / "app/naver_connect_chatbot/prompts/templates"
    
    # Answer generation 프롬프트들 확인
    answer_gen_files = [
        "answer_generation_simple.yaml",
        "answer_generation_complex.yaml",
        "answer_generation_exploratory.yaml",
    ]
    
    for filename in answer_gen_files:
        file_path = prompts_dir / filename
        assert file_path.exists(), f"{filename} should exist"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Reasoning 관련 내용이 있는지 확인
        assert "Reasoning" in content or "reasoning" in content or "step-by-step" in content, \
            f"{filename} should include Reasoning guidance"
    
    # Query analysis 프롬프트 확인
    query_analysis_file = prompts_dir / "query_analysis.yaml"
    assert query_analysis_file.exists(), "query_analysis.yaml should exist"
    
    with open(query_analysis_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Multi-Query 통합 확인
    assert "multi" in content.lower() or "diverse" in content.lower(), \
        "query_analysis.yaml should mention multi-query or diverse queries"
    
    print("✓ Prompt templates updated correctly")


def test_types_file_updates():
    """
    types.py 파일의 업데이트를 확인합니다.
    """
    base_path = Path(__file__).parent.parent
    types_file = base_path / "app/naver_connect_chatbot/service/graph/types.py"
    
    assert types_file.exists(), "types.py file should exist"
    
    with open(types_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 제거된 타입 확인
    assert "class ValidationUpdate" not in content, "ValidationUpdate class should be removed"
    assert "class CorrectionUpdate" not in content, "CorrectionUpdate class should be removed"
    
    # 유지된 타입 확인
    assert "class IntentUpdate" in content, "IntentUpdate should exist"
    assert "class QueryAnalysisUpdate" in content, "QueryAnalysisUpdate should exist"
    assert "class RetrievalUpdate" in content, "RetrievalUpdate should exist"
    assert "class DocumentEvaluationUpdate" in content, "DocumentEvaluationUpdate should exist"
    assert "class AnswerUpdate" in content, "AnswerUpdate should exist"
    
    print("✓ types.py updates are correct")


def test_agents_init_file():
    """
    agents/__init__.py 파일이 올바르게 업데이트되었는지 확인합니다.
    """
    base_path = Path(__file__).parent.parent
    agents_init = base_path / "app/naver_connect_chatbot/service/agents/__init__.py"
    
    assert agents_init.exists(), "agents/__init__.py should exist"
    
    with open(agents_init, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 제거된 import 확인
    assert "answer_validator" not in content, "answer_validator should not be imported"
    assert "corrector" not in content, "corrector should not be imported"
    assert "AnswerValidation" not in content, "AnswerValidation should not be imported"
    assert "CorrectionStrategy" not in content, "CorrectionStrategy should not be imported"
    
    # 유지된 import 확인
    assert "intent_classifier" in content, "intent_classifier should be imported"
    assert "query_analyzer" in content, "query_analyzer should be imported"
    assert "document_evaluator" in content, "document_evaluator should be imported"
    assert "answer_generator" in content, "answer_generator should be imported"
    
    print("✓ agents/__init__.py updates are correct")


def test_retriever_factory_updates():
    """
    retriever_factory.py 파일의 업데이트를 확인합니다.
    """
    base_path = Path(__file__).parent.parent
    factory_file = base_path / "app/naver_connect_chatbot/rag/retriever_factory.py"
    
    assert factory_file.exists(), "retriever_factory.py should exist"
    
    with open(factory_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Multi-Query 기본 활성화 확인
    # enable_multi_query: bool = True 패턴 찾기
    assert "enable_multi_query: bool = True" in content, \
        "enable_multi_query should default to True"
    
    # Docstring 업데이트 확인
    assert "기본값: True" in content or "기본" in content, \
        "Documentation should mention Multi-Query is enabled by default"
    
    print("✓ retriever_factory.py updates are correct")


if __name__ == "__main__":
    print("="*60)
    print("RAG Pipeline Structure Validation Tests")
    print("="*60)
    
    tests = [
        ("Removed files check", test_removed_files),
        ("New files check", test_new_files_exist),
        ("nodes.py structure", test_nodes_file_structure),
        ("workflow.py structure", test_workflow_file_structure),
        ("rag_settings.py updates", test_settings_file_updates),
        ("Prompt templates", test_prompts_updated),
        ("types.py updates", test_types_file_updates),
        ("agents/__init__.py", test_agents_init_file),
        ("retriever_factory.py", test_retriever_factory_updates),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"\n{name}...")
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✓ All RAG Pipeline Structure Tests Passed!")
        sys.exit(0)
    else:
        print(f"\n✗ {failed} test(s) failed")
        sys.exit(1)

