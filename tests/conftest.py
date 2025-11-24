"""테스트 전역 설정."""

# pytest-asyncio 플러그인을 명시적으로 로드하여 asyncio 테스트를 지원합니다.
pytest_plugins = ("pytest_asyncio",)

import asyncio
import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """async def 테스트를 asyncio.run으로 강제 실행합니다."""
    testfunction = pyfuncitem.obj
    import inspect
    if inspect.iscoroutinefunction(testfunction):
        sig = inspect.signature(testfunction)
        kwargs = {name: pyfuncitem.funcargs[name] for name in sig.parameters}
        asyncio.run(testfunction(**kwargs))
        return True
    return None
