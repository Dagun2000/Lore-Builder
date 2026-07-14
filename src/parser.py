"""Rule-based input parsing: tag/year extraction via regex only, no LLM."""

import re
from dataclasses import dataclass


@dataclass
class ParsedInput:
    years: list
    tags: list
    raw_text: str


_TAG_PATTERN = re.compile(r"\[([^\[\]]+)\]")
_YEAR_PATTERN = re.compile(r"(\d+)\s*년")


def parse_input(text: str) -> ParsedInput:
    tags = _TAG_PATTERN.findall(text)

    # Year extraction ignores bracket contents entirely — a tag like
    # "[100년 전쟁]" contains something that looks like a year but isn't one
    # (it's a proper noun). Blank out every bracketed span before scanning
    # for years, so only years mentioned in the actual prose count.
    text_outside_brackets = _TAG_PATTERN.sub(" ", text)
    years = sorted({int(m) for m in _YEAR_PATTERN.findall(text_outside_brackets)})

    if not years:
        raise ValueError("연도를 명시해주세요 (예: '2100년').")

    return ParsedInput(years=years, tags=tags, raw_text=text)
