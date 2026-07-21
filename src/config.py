"""Model tier -> concrete model name mapping, so call sites never hardcode model ids.

Phase 10 patch 22 follow-up 4: multi-provider support. `LLM_PROVIDER` (env,
default "openai") picks which backend every LLM call in the app uses —
openai, anthropic, ollama, or google. Each provider gets its own
tier -> model-id mapping (also env-overridable, since model ids never
transfer across providers). Every call site goes through get_chat_model()
now instead of importing a provider's LangChain class directly, so
switching providers is a one-line env var change, not a code change."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

_PROVIDER_MODELS = {
    "openai": {
        "simple": os.environ.get("OPENAI_SIMPLE_MODEL", "gpt-5.4-mini"),
        "reasoning": os.environ.get("OPENAI_REASONING_MODEL", "gpt-5.6-terra"),
    },
    "anthropic": {
        "simple": os.environ.get("ANTHROPIC_SIMPLE_MODEL", "claude-haiku-4-5"),
        "reasoning": os.environ.get("ANTHROPIC_REASONING_MODEL", "claude-sonnet-5"),
    },
    "ollama": {
        "simple": os.environ.get("OLLAMA_SIMPLE_MODEL", "llama3.1"),
        "reasoning": os.environ.get("OLLAMA_REASONING_MODEL", "llama3.1"),
    },
    "google": {
        "simple": os.environ.get("GOOGLE_SIMPLE_MODEL", "gemini-2.5-flash"),
        "reasoning": os.environ.get("GOOGLE_REASONING_MODEL", "gemini-2.5-pro"),
    },
}


def _current_provider() -> str:
    provider = os.environ.get("LLM_PROVIDER", "openai").strip().lower()
    if provider not in _PROVIDER_MODELS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r} (expected one of {sorted(_PROVIDER_MODELS)})"
        )
    return provider


def get_model(tier: str) -> str:
    """Model id for `tier` under the currently configured provider."""
    models = _PROVIDER_MODELS[_current_provider()]
    try:
        return models[tier]
    except KeyError:
        raise ValueError(f"Unknown model tier: {tier!r}")


def parallel_rag_checks_enabled() -> bool:
    """Whether rag_check.run_rag_checks may fire its independent LLM calls
    (check_rule_and_notes, check_status_consistency) concurrently instead
    of sequentially. Defaults to true (cloud providers have no trouble
    serving two concurrent requests, and it roughly halves wall-clock
    latency for events where both checks actually fire) — but a local
    Ollama setup running as large a model as VRAM allows may only be able
    to run one generation at a time at all, where firing two at once could
    contend for the same GPU memory instead of actually running in
    parallel. Set PARALLEL_RAG_CHECKS=false in .env to force sequential."""
    return os.environ.get("PARALLEL_RAG_CHECKS", "true").strip().lower() not in ("false", "0", "no")


def get_chat_model(tier: str, temperature: float = 0):
    """Instantiate the LangChain chat model for `tier` under the currently
    configured provider (LLM_PROVIDER env var, default "openai"). Every LLM
    call site in the app goes through this instead of importing a
    provider's class directly, so adding/switching providers never touches
    the call sites themselves.

    Each provider reads its own API key from the environment the way it
    always has (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY) — no
    extra plumbing needed here. Ollama needs no key (local server); its
    endpoint is configurable via OLLAMA_BASE_URL (defaults to Ollama's own
    default, http://localhost:11434, when unset)."""
    provider = _current_provider()
    model = get_model(tier)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model, temperature=temperature, base_url=os.environ.get("OLLAMA_BASE_URL")
        )
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")  # pragma: no cover — _current_provider already validates
