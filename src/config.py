"""Model tier -> concrete model name mapping, so call sites never hardcode model ids."""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

_MODELS = {
    "simple": "gpt-5.4-mini",
    "reasoning": "gpt-5.6-terra",
}


def get_model(tier: str) -> str:
    try:
        return _MODELS[tier]
    except KeyError:
        raise ValueError(f"Unknown model tier: {tier!r}")
