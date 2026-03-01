"""
C-3: Slack handler 초기화 테스트.

- C-3-1: get_agent_app()가 그래프를 정상 구성하는지 확인
- C-3-2: 분류용/응답용 LLM 인스턴스가 분리되어 생성되는지 확인
"""

from __future__ import annotations

from unittest.mock import MagicMock


def test_get_agent_app_compiles_graph(monkeypatch, tmp_path, mock_retriever):
    """C-3-1: get_agent_app() + MockRetriever로 그래프 빌드가 호출된다."""
    from naver_connect_chatbot.slack import handler

    fake_graph = object()
    fake_checkpointer = object()

    monkeypatch.setattr(handler, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(handler, "get_embeddings", lambda: object())
    monkeypatch.setattr(handler, "get_chat_model", MagicMock(side_effect=[MagicMock(), MagicMock()]))
    monkeypatch.setattr(handler, "build_dense_sparse_hybrid_from_saved", lambda **kwargs: mock_retriever)
    monkeypatch.setattr(handler, "get_hybrid_retriever", lambda **kwargs: mock_retriever)

    build_graph_mock = MagicMock(return_value=fake_graph)
    monkeypatch.setattr(handler, "build_adaptive_rag_graph", build_graph_mock)

    result = handler.get_agent_app(checkpointer=fake_checkpointer)

    assert result is fake_graph
    build_graph_mock.assert_called_once()
    called_kwargs = build_graph_mock.call_args.kwargs
    assert called_kwargs["retriever"] is mock_retriever
    assert called_kwargs["check_pointers"] is fake_checkpointer


def test_get_agent_app_uses_separate_llm_instances(monkeypatch, tmp_path, mock_retriever):
    """C-3-2: classification/answer용 LLM이 분리 생성되어 그래프에 전달된다."""
    from naver_connect_chatbot.slack import handler

    classification_llm = MagicMock(name="classification_llm")
    answer_llm = MagicMock(name="answer_llm")

    get_chat_model_mock = MagicMock(side_effect=[classification_llm, answer_llm])
    build_graph_mock = MagicMock(return_value=object())

    monkeypatch.setattr(handler, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(handler, "get_embeddings", lambda: object())
    monkeypatch.setattr(handler, "get_chat_model", get_chat_model_mock)
    monkeypatch.setattr(handler, "build_dense_sparse_hybrid_from_saved", lambda **kwargs: mock_retriever)
    monkeypatch.setattr(handler, "get_hybrid_retriever", lambda **kwargs: mock_retriever)
    monkeypatch.setattr(handler, "build_adaptive_rag_graph", build_graph_mock)

    handler.get_agent_app()

    assert get_chat_model_mock.call_count == 2

    first_call_kwargs = get_chat_model_mock.call_args_list[0].kwargs
    second_call_kwargs = get_chat_model_mock.call_args_list[1].kwargs

    assert first_call_kwargs.get("thinking_level") == "low"
    assert "thinking_level" not in second_call_kwargs

    called_kwargs = build_graph_mock.call_args.kwargs
    assert called_kwargs["llm"] is classification_llm
    assert called_kwargs["reasoning_llm"] is answer_llm
