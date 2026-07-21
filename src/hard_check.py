"""Deterministic hard checks (Phase 1) — no LLM calls."""

from dataclasses import dataclass

from . import schema, storage


@dataclass
class Conflict:
    check_type: str  # "terminal" | "lifespan" | "duration_closure"
    severity: str  # "blocking" | "warning"
    entity_id: str
    reason: str


def check_terminal_violation(
    category: str,
    entity_id: str,
    extra_years: list | None = None,
    candidate_start: int | None = None,
    candidate_end: int | None = None,
) -> Conflict | None:
    """lifecycle_start <= min(event years) and lifecycle_end >= max(event years).

    `extra_years` lets a caller include a candidate event's year that hasn't
    been persisted yet — e.g. Phase 5's pipeline runs this check *before*
    archivist writes the new timeline record, so storage.get_event_years()
    alone can't see the event currently being submitted.

    `candidate_start`/`candidate_end` let a caller check a lifecycle field
    value that hasn't been written to the entity yet either — e.g. whether
    setting death_year=2060 would conflict with an already-recorded 2080
    event, checked *before* death_year is actually saved."""
    start_fields = schema.get_fields_with_role(category, "lifecycle_start")
    end_fields = schema.get_fields_with_role(category, "lifecycle_end")
    if not start_fields and not end_fields:
        return None

    entity = storage.get_entity(category, entity_id)
    if entity is None:
        return None

    start_name = start_fields[0]["name"] if start_fields else None
    end_name = end_fields[0]["name"] if end_fields else None
    start_value = candidate_start if candidate_start is not None else (
        entity.get(start_name) if start_name else None
    )
    end_value = candidate_end if candidate_end is not None else (
        entity.get(end_name) if end_name else None
    )

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


def check_duration_closure_conflict(duration_effect: dict | None) -> Conflict | None:
    """A `clear`/`set_closed` duration_effect proposes ending an (entity,
    predicate, target) status/relationship at some end_year — if the
    record that's actually still open is closed for real (its own
    end_year already set) and the proposed end_year disagrees, that's a
    genuine conflict.

    This exists because an LLM contradiction check was tried first and
    didn't hold: rag_check's context now spells out a record's real
    "(실제 기간: 2085년~2090년, 이미 종료됨)" explicitly, right next to the
    record's own notes, and a fictional new event closing the same status
    at a *different* year (e.g. 2086) still sailed through — the model
    just didn't connect the two. A plain field comparison in code doesn't
    have that failure mode; it either matches or it doesn't.

    The same (entity, predicate, target) triple can legitimately recur —
    imprisoned once, released, imprisoned again later — leaving multiple
    matching records, only one of which (if any) is still open. A first
    version of this check compared the proposed end_year against *every*
    matching record regardless of whether it was already closed, so
    closing a genuinely new, currently-open second imprisonment got
    rejected because it disagreed with the first, unrelated, already-
    closed one from years earlier (caught via direct repro, not
    anticipated). Only the currently-open record (if one exists) is what
    this closure could possibly be about — an already-closed record from
    a separate past episode is irrelevant to it. Comparison against
    already-closed records only happens when nothing is currently open at
    all, which is exactly the original bug this check was built to catch.

    That "compare against the one already-closed record" step only makes
    sense for `clear`, which by spec carries no start_year of its own — it
    is inherently a reference to *some specific existing* record it means
    to close, so if nothing is open, the only candidate left is whichever
    already-closed record shares the triple, and a disagreeing end_year on
    that is a genuine self-contradiction. `set_closed` is the opposite: it
    always supplies its own start_year, so it is a wholly new, self-
    contained episode by construction, never a reference to an older one —
    "traveled_world_with 2080-2084, already closed" existing on record must
    never block a brand-new "traveled_world_with 2105-2110" set_closed for
    the same pair; recurrence is legitimate (this function's own docstring
    says so), and set_closed has no ambiguity to resolve in the first place
    since it never needs another record's start_year to be complete. Caught
    via direct user report: a second, later 2105-2110 relationship of the
    exact same predicate was rejected as contradicting the first, unrelated,
    already-closed 2080-2084 episode — the same conflating-separate-
    episodes bug this docstring already warns about, just reached through
    set_closed instead of the currently-open branch above."""
    if not duration_effect:
        return None
    action = duration_effect.get("action")
    if action not in ("clear", "set_closed"):
        return None
    entity_id = duration_effect.get("entity")
    predicate = duration_effect.get("predicate")
    proposed_end = duration_effect.get("end_year")
    if not entity_id or not predicate or proposed_end is None:
        return None
    target = duration_effect.get("target")

    matching = [
        event for event in storage.get_events_for_entity(entity_id)
        if event.get("entity") == entity_id
        and event.get("predicate") == predicate
        and event.get("target") == target
    ]
    if any(event.get("end_year") is None for event in matching):
        # A currently-open record exists — that's the one this closure is
        # about, and closing it is always valid regardless of what any
        # separate, already-closed past episode says.
        return None
    if action == "set_closed":
        # A brand-new, self-contained episode never needs to match an
        # older closed one — see the docstring section above.
        return None

    for event in matching:
        existing_end = event.get("end_year")
        if existing_end is not None and existing_end != proposed_end:
            return Conflict(
                check_type="duration_closure",
                severity="blocking",
                entity_id=entity_id,
                reason=(
                    f"{entity_id}의 '{predicate}' 상태/관계는 이미 {existing_end}년에 종료된 "
                    f"것으로 기록되어 있습니다 ({event['id']}). {proposed_end}년 종료로 다시 "
                    f"닫을 수 없습니다."
                ),
            )
    return None


def get_existence_range(category: str, entity_id: str) -> tuple:
    """(earliest, latest) year entity_id can be assumed to exist — used by
    Creator (Phase 10 patch 22) to scope a generated story to years where
    every involved entity is actually around. A category with no lifecycle
    fields at all is never constrained (returns (None, None), unbounded).

    The lower bound falls back to the earliest year the entity is on
    record for (storage.get_event_years) when no lifecycle_start field is
    set — the same signal check_terminal_violation already trusts as "at
    least existed by this year". The upper bound never gets a symmetric
    fallback: an entity with no death/destroyed year on record is presumed
    to still exist indefinitely, not bounded by whatever their most recent
    recorded event happens to be — bounding it that way would forbid
    placing any new story after the last thing we happened to record,
    which is exactly what Creator exists to add."""
    start_fields = schema.get_fields_with_role(category, "lifecycle_start")
    end_fields = schema.get_fields_with_role(category, "lifecycle_end")
    if not start_fields and not end_fields:
        return None, None

    entity = storage.get_entity(category, entity_id)
    if entity is None:
        return None, None

    start_name = start_fields[0]["name"] if start_fields else None
    end_name = end_fields[0]["name"] if end_fields else None
    lower = entity.get(start_name) if start_name else None
    upper = entity.get(end_name) if end_name else None

    if lower is None:
        years = storage.get_event_years(entity_id)
        if years:
            lower = min(years)

    return lower, upper


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
