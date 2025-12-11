"""
AsyncSqliteSaver Checkpointer 테스트

LangGraph checkpointer가 대화 상태를 올바르게 저장/복원하는지 검증합니다.
"""

import asyncio
import tempfile
from pathlib import Path

import aiosqlite
import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict


class SimpleState(TypedDict, total=False):
    """간단한 테스트용 상태"""
    messages: list[str]
    count: int


def add_message_node(state: SimpleState) -> SimpleState:
    """메시지를 추가하는 노드"""
    messages = state.get("messages", [])
    count = state.get("count", 0)
    messages.append(f"Message {count + 1}")
    return {"messages": messages, "count": count + 1}


def build_simple_graph(checkpointer):
    """테스트용 간단한 그래프 생성"""
    workflow = StateGraph(state_schema=SimpleState)
    workflow.add_node("add_message", add_message_node)
    workflow.set_entry_point("add_message")
    workflow.add_edge("add_message", END)
    return workflow.compile(checkpointer=checkpointer)


@pytest.mark.asyncio
async def test_checkpointer_saves_state():
    """Checkpointer가 상태를 저장하는지 테스트"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_checkpoints.db"

        async with aiosqlite.connect(str(db_path)) as conn:
            checkpointer = AsyncSqliteSaver(conn)
            await checkpointer.setup()

            graph = build_simple_graph(checkpointer)

            # 첫 번째 호출
            thread_id = "test-thread-1"
            config = {"configurable": {"thread_id": thread_id}}

            result1 = await graph.ainvoke({"messages": [], "count": 0}, config=config)

            assert result1["count"] == 1
            assert result1["messages"] == ["Message 1"]

            print(f"✓ 첫 번째 호출 성공: {result1}")


@pytest.mark.asyncio
async def test_checkpointer_restores_state():
    """Checkpointer가 상태를 복원하는지 테스트 (멀티턴 대화)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_checkpoints.db"

        async with aiosqlite.connect(str(db_path)) as conn:
            checkpointer = AsyncSqliteSaver(conn)
            await checkpointer.setup()

            graph = build_simple_graph(checkpointer)

            thread_id = "test-thread-multi"
            config = {"configurable": {"thread_id": thread_id}}

            # 첫 번째 호출
            result1 = await graph.ainvoke({"messages": [], "count": 0}, config=config)
            assert result1["count"] == 1
            print(f"✓ 첫 번째 호출: count={result1['count']}, messages={result1['messages']}")

            # 두 번째 호출 - 이전 상태가 복원되어야 함
            result2 = await graph.ainvoke({}, config=config)
            assert result2["count"] == 2
            assert len(result2["messages"]) == 2
            print(f"✓ 두 번째 호출: count={result2['count']}, messages={result2['messages']}")

            # 세 번째 호출
            result3 = await graph.ainvoke({}, config=config)
            assert result3["count"] == 3
            assert len(result3["messages"]) == 3
            print(f"✓ 세 번째 호출: count={result3['count']}, messages={result3['messages']}")


@pytest.mark.asyncio
async def test_different_threads_isolated():
    """다른 thread_id는 독립적인 상태를 가지는지 테스트"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_checkpoints.db"

        async with aiosqlite.connect(str(db_path)) as conn:
            checkpointer = AsyncSqliteSaver(conn)
            await checkpointer.setup()

            graph = build_simple_graph(checkpointer)

            # Thread A
            config_a = {"configurable": {"thread_id": "thread-A"}}
            result_a1 = await graph.ainvoke({"messages": [], "count": 0}, config=config_a)
            result_a2 = await graph.ainvoke({}, config=config_a)

            # Thread B (새로운 스레드)
            config_b = {"configurable": {"thread_id": "thread-B"}}
            result_b1 = await graph.ainvoke({"messages": [], "count": 0}, config=config_b)

            # Thread A는 count=2, Thread B는 count=1이어야 함
            assert result_a2["count"] == 2
            assert result_b1["count"] == 1

            print(f"✓ Thread A: count={result_a2['count']}")
            print(f"✓ Thread B: count={result_b1['count']}")
            print("✓ 스레드 격리 확인 완료")


@pytest.mark.asyncio
async def test_checkpointer_persists_across_connections():
    """연결을 닫았다가 다시 열어도 상태가 유지되는지 테스트"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_checkpoints.db"
        thread_id = "persist-test-thread"
        config = {"configurable": {"thread_id": thread_id}}

        # 첫 번째 연결: 상태 저장
        async with aiosqlite.connect(str(db_path)) as conn:
            checkpointer = AsyncSqliteSaver(conn)
            await checkpointer.setup()
            graph = build_simple_graph(checkpointer)

            result1 = await graph.ainvoke({"messages": [], "count": 0}, config=config)
            result2 = await graph.ainvoke({}, config=config)

            assert result2["count"] == 2
            print(f"✓ 첫 번째 연결에서 count={result2['count']}")

        # 연결 닫힘 (async with 종료)
        print("✓ 연결 종료")

        # 두 번째 연결: 상태 복원 확인
        async with aiosqlite.connect(str(db_path)) as conn:
            checkpointer = AsyncSqliteSaver(conn)
            await checkpointer.setup()
            graph = build_simple_graph(checkpointer)

            # 이전 상태가 복원되어야 함
            result3 = await graph.ainvoke({}, config=config)

            assert result3["count"] == 3
            print(f"✓ 두 번째 연결에서 count={result3['count']} (이전 상태 복원 확인)")


@pytest.mark.asyncio
async def test_slack_thread_ts_as_thread_id():
    """Slack thread_ts 형식이 thread_id로 잘 동작하는지 테스트"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_checkpoints.db"

        async with aiosqlite.connect(str(db_path)) as conn:
            checkpointer = AsyncSqliteSaver(conn)
            await checkpointer.setup()

            graph = build_simple_graph(checkpointer)

            # Slack thread_ts 형식 (실제 Slack에서 사용하는 형식)
            slack_thread_ts = "1734012345.123456"
            config = {"configurable": {"thread_id": slack_thread_ts}}

            result1 = await graph.ainvoke({"messages": [], "count": 0}, config=config)
            result2 = await graph.ainvoke({}, config=config)

            assert result1["count"] == 1
            assert result2["count"] == 2

            print(f"✓ Slack thread_ts '{slack_thread_ts}' 형식 지원 확인")
            print(f"✓ 멀티턴: count {result1['count']} → {result2['count']}")


if __name__ == "__main__":
    # 직접 실행 시 모든 테스트 수행
    async def run_all_tests():
        print("=" * 60)
        print("AsyncSqliteSaver Checkpointer 테스트")
        print("=" * 60)

        print("\n[1] 상태 저장 테스트")
        await test_checkpointer_saves_state()

        print("\n[2] 상태 복원 테스트 (멀티턴)")
        await test_checkpointer_restores_state()

        print("\n[3] 스레드 격리 테스트")
        await test_different_threads_isolated()

        print("\n[4] 연결 재시작 후 상태 유지 테스트")
        await test_checkpointer_persists_across_connections()

        print("\n[5] Slack thread_ts 형식 테스트")
        await test_slack_thread_ts_as_thread_id()

        print("\n" + "=" * 60)
        print("✅ 모든 테스트 통과!")
        print("=" * 60)

    asyncio.run(run_all_tests())
