"""Keep agent tests hermetic: CI job variables must not leak into budget/key behaviour."""
import pytest

_LEAKY = ("MAX_USD", "PRIOR_SPEND_USD", "GEMINI_API_KEY", "DASHSCOPE_API_KEY",
          "DASHSCOPE_BASE_URL", "GEMINI_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL")


@pytest.fixture(autouse=True)
def _clean_job_env(monkeypatch):
    for name in _LEAKY:
        monkeypatch.delenv(name, raising=False)
