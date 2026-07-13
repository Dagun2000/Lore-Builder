"""Rule-based input parsing: tag/year extraction via regex only, no LLM."""

import re
from dataclasses import dataclass


@dataclass
class ParsedInput:
    year: int
    tags: list
    raw_text: str


_TAG_PATTERN = re.compile(r"\[([^\[\]]+)\]")
_YEAR_PATTERN = re.compile(r"(\d+)\s*년")


def parse_input(text: str) -> ParsedInput:
    tags = _TAG_PATTERN.findall(text)

    year_match = _YEAR_PATTERN.search(text)
    if year_match is None:
        raise ValueError("연도를 명시해주세요 (예: '2100년').")

    return ParsedInput(year=int(year_match.group(1)), tags=tags, raw_text=text)
