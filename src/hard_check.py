"""Deterministic hard checks (Phase 1) — no LLM calls."""

from dataclasses import dataclass

from . import schema, storage


@dataclass
class Conflict:
    check_type: str  # "terminal" | "lifespan"
    severity: str  # "blocking" | "warning"
    entity_id: str
    reason: str


def check_terminal_violation(
    category: str, entity_id: str, extra_years: list | None = None
) -> Conflict | None:
    """lifecycle_start <= min(event years) and lifecycle_end >= max(event years).

    `extra_years` lets a caller include a candidate event's year that hasn't
    been persisted yet — e.g. Phase 5's pipeline runs this check *before*
    archivist writes the new timeline record, so storage.get_event_years()
    alone can't see the event currently being submitted."""
    start_fields = schema.get_fields_with_role(category, "lifecycle_start")
    end_fields = schema.get_fields_with_role(category, "lifecycle_end")
    if not start_fields and not end_fields:
        return None

    entity = storage.get_entity(category, entity_id)
    if entity is None:
        return None

    start_name = start_fields[0]["name"] if start_fields else None
    end_name = end_fields[0]["name"] if end_fields else None
    start_value = entity.get(start_name) if start_name else None
    end_value = entity.get(end_name) if end_name else None

    if start_value is None and end_value is None:
        return None

    years = storage.get_event_years(entity_id)
    if extra_years:
        years = years + list(extra_years)
    if not years:
        return None

    min_year, max_year = min(years), max(years)

    if start_value is not None and start_value > min_year:
        return Conflict(
            check_type="terminal",
            severity="blocking",
            entity_id=entity_id,
            reason=(
                f"{entity_id}의 {start_name}({start_value})이(가) "
                f"관련 사건 연도({min_year})보다 늦습니다."
            ),
        )

    if end_value is not None and end_value < max_year:
        return Conflict(
            check_type="terminal",
            severity="blocking",
            entity_id=entity_id,
            reason=(
                f"{entity_id}의 {end_name}({end_value})이(가) "
                f"관련 사건 연도({max_year})보다 이릅니다."
            ),
        )

    return None


def check_lifespan_violation(
    character_id: str, extra_years: list | None = None
) -> Conflict | None:
    """Warn when a character's inferred lifespan exceeds their race's lifespan.

    See check_terminal_violation for why `extra_years` exists."""
    entity = storage.get_entity("character", character_id)
    if entity is None:
        return None

    if entity.get("lifespan_check_ack"):
        return None

    race_id = entity.get("race")
    if not race_id:
        return None

    race_entity = storage.get_entity("race", race_id)
    if race_entity is None:
        return None

    lifespan = race_entity.get("lifespan")
    if lifespan is None:
        return None

    years = storage.get_event_years(character_id)
    if extra_years:
        years = years + list(extra_years)

    birth_year = entity.get("birth_year")
    lower = birth_year if birth_year is not None else (min(years) if years else None)

    death_year = entity.get("death_year")
    upper = death_year if death_year is not None else (max(years) if years else None)

    if lower is None or upper is None:
        return None

    age = upper - lower
    if age > lifespan:
        exceeded = age - lifespan
        reason = (
            f"{character_id}(종족: {race_id})의 생존 기간이 {lower}~{upper}년으로 "
            f"총 {age}년입니다. 종족 수명 {lifespan}년을 {exceeded}년 초과했습니다."
        )
        return Conflict(
            check_type="lifespan",
            severity="warning",
            entity_id=character_id,
            reason=reason,
        )

    return None


def run_hard_checks(
    category: str, entity_id: str, extra_years: list | None = None
) -> list[Conflict]:
    """Run every applicable hard check for an entity and collect the conflicts."""
    conflicts = []

    terminal = check_terminal_violation(category, entity_id, extra_years=extra_years)
    if terminal is not None:
        conflicts.append(terminal)

    if category == "character":
        lifespan = check_lifespan_violation(entity_id, extra_years=extra_years)
        if lifespan is not None:
            conflicts.append(lifespan)

    return conflicts
