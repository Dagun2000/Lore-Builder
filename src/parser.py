"""Rule-based input parsing: tag/year extraction via regex only, no LLM."""

import re
from dataclasses import dataclass

from . import settings


@dataclass
class ParsedInput:
    years: list
    tags: list
    raw_text: str


_TAG_PATTERN = re.compile(r"\[([^\[\]]+)\]")
_YEAR_PATTERN_KO = re.compile(r"(\d+)\s*년")
_YEAR_BRACKET_CONTENT = re.compile(r"^\d+$")


def parse_input(text: str) -> ParsedInput:
    bracket_contents = _TAG_PATTERN.findall(text)

    if settings.get_world_language() == "en":
        # English has no year-marking postfix the way Korean's "년" is —
        # scanning bare prose for standalone digit runs is genuinely
        # ambiguous ("2000 soldiers" is a count, not a year, and digit-count
        # alone can't tell them apart). Brackets already unambiguously mark
        # entity tags in every mode, so English mode reuses that exact same
        # marker for years too instead of inventing a new one: "[2100]" is a
        # year, "[the Inn]" is a tag — split purely-numeric bracket contents
        # from the rest rather than scanning outside the brackets at all.
        years = sorted({int(t) for t in bracket_contents if _YEAR_BRACKET_CONTENT.match(t)})
        tags = [t for t in bracket_contents if not _YEAR_BRACKET_CONTENT.match(t)]
    else:
        tags = bracket_contents
        # Year extraction ignores bracket contents entirely — a tag like
        # "[100년 전쟁]" contains something that looks like a year but isn't
        # one (it's a proper noun). Blank out every bracketed span before
        # scanning for years, so only years mentioned in the actual prose
        # count.
        text_outside_brackets = _TAG_PATTERN.sub(" ", text)
        years = sorted({int(m) for m in _YEAR_PATTERN_KO.findall(text_outside_brackets)})

    # A year is no longer required at parse time (Phase 10 patch 3, E): a
    # pure entity-introduction sentence with no event ("[아마조네스 용병단]은
    # 여성만이 가입 가능한 특수한 용병단이다") is valid input on its own —
    # years only matter once there's an actual event to anchor, which
    # pipeline_session._pipeline_generator already handles by returning
    # status "entity_only" when no year survives entity resolution.
    return ParsedInput(years=years, tags=tags, raw_text=text)
