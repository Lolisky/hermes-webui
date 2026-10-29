"""Single-session CLI metadata lookups must not re-project the whole catalog.

``_lookup_cli_session_metadata()`` used to call ``get_cli_sessions()`` — a
projection over the whole agent ``state.db`` — and then scan the result for one
id. That projection costs seconds once the database grows, and its cache key
embeds a live ``state.db`` fingerprint, so an agent write makes the snapshot
unreachable and the next lookup pays again. A single page load issued several
of these lookups and paid the projection each time.

``lookup_cli_session_row()`` is a **probe of the producer's own cache**: it
answers only when the snapshot it finds is provably what a current enumeration
would return, and returns ``{}`` otherwise so the caller falls back to
``get_cli_sessions()``. It never enumerates itself and never answers from a
snapshot it cannot vouch for.

These tests pin both halves:

1. it answers without re-projecting (the perf win), and
2. it never answers for a snapshot it cannot vouch for — different profile,
   different revision, disabled cache, a streaming freeze past its freshness
   window, or a confirmation that resolves a different owner.
"""
import pytest

from api import models, routes


ROWS = [
    {
        "session_id": "cli-fastpath-a",
        "profile": "default",
        "source_tag": "cli",
        "raw_source": "cli",
        "session_source": "cli",
        "source_label": "CLI",
        "parent_session_id": None,
        "workspace": None,
        "title": "Fastpath A",
    },
    {
        "session_id": "cli-fastpath-b",
        "profile": "default",
        "source_tag": "cron",
        "raw_source": "cron",
        "session_source": "cron",
        "source_label": "Cron",
        "parent_session_id": None,
        "workspace": None,
        "title": "Fastpath B",
    },
]


@pytest.fixture
def stubbed_catalog(monkeypatch):
    """Serve a fixed CLI catalog so no real ``state.db`` is required."""

    def _load(*args, **kwargs):
        return [dict(row) for row in ROWS]

    monkeypatch.setattr(models, "_load_cli_sessions_uncached", _load)
    models.clear_cli_sessions_cache()
    try:
        yield ROWS
    finally:
        models.clear_cli_sessions_cache()


def _authoritative_reader(rows):
    """A bounded-reader stand-in mirroring get_cli_session_by_id's contract."""

    def _read(sid, source, **kwargs):
        for row in rows:
            if row.get("session_id") == sid:
                return dict(row)
        return {}

    return _read


def _prime():
    """Populate the producer cache through the real producer."""
    return models.get_cli_sessions()


def _expire_primary():
    with models._CLI_SESSIONS_CACHE_LOCK:
        for key in list(models._CLI_SESSIONS_CACHE):
            entry = models._CLI_SESSIONS_CACHE[key]
            models._CLI_SESSIONS_CACHE[key] = (0.0, entry[1], entry[2])


def _count_enumerations(monkeypatch, rows=None):
    """Make every enumeration path record a call and answer ``rows``."""
    calls = []

    def _enum(*args, **kwargs):
        calls.append(1)
        return [dict(r) for r in (rows or [])]

    monkeypatch.setattr(models, "get_cli_sessions", _enum)
    monkeypatch.setattr(routes, "get_cli_sessions", _enum)
    return calls


def _block_fallback_reads(monkeypatch):
    """Silence the route's pre-existing ``oneshot`` fallback read.

    ``_lookup_cli_session_metadata`` retries ``get_cli_session_by_id`` for a
    few sources after the enumeration misses. ``routes`` holds its own
    reference to that function, so a ``models``-level patch does not reach it.
    """
    monkeypatch.setattr(routes, "get_cli_session_by_id", lambda *a, **kw: {})


# ── The fast path answers, and answers correctly ──────────────────────────


def test_lookup_matches_full_enumeration(stubbed_catalog):
    """The fast path returns exactly the row the enumeration returns."""
    expected = {row["session_id"]: row for row in _prime()}
    assert expected, "fixture should produce rows"

    for session_id, row in expected.items():
        assert models.lookup_cli_session_row(session_id) == row

    # Unknown and empty ids keep the enumeration's "not found" answer ({}).
    assert models.lookup_cli_session_row("no-such-session-id") == {}
    assert models.lookup_cli_session_row("") == {}


def test_metadata_lookup_does_not_enumerate_when_snapshot_is_cached(
    stubbed_catalog, monkeypatch
):
    """A cached catalog answers the lookup without re-running the projection.

    This is the behavioural assertion: on the pre-fix code the second call
    below re-entered ``get_cli_sessions()`` (the spy raises).
    """
    target = ROWS[0]["session_id"]
    assert routes._lookup_cli_session_metadata(target)["session_id"] == target

    enumerations = []

    def _boom(*args, **kwargs):
        enumerations.append(1)
        raise AssertionError(
            "_lookup_cli_session_metadata re-ran the CLI catalog projection; "
            "a cached snapshot must answer a single-session lookup"
        )

    monkeypatch.setattr(routes, "get_cli_sessions", _boom)
    monkeypatch.setattr(models, "get_cli_session_by_id", _authoritative_reader(ROWS))

    meta = routes._lookup_cli_session_metadata(target)
    assert meta["session_id"] == target
    assert enumerations == []


def test_expired_snapshot_still_answers(stubbed_catalog, monkeypatch):
    """TTL expiry alone must not force a re-projection.

    Expiry changes freshness, not membership: the revision is unchanged, so
    the snapshot still describes the current catalog and the bounded read
    confirms the content.
    """
    target = ROWS[0]["session_id"]
    assert _prime()
    _expire_primary()

    monkeypatch.setattr(models, "get_cli_session_by_id", _authoritative_reader(ROWS))
    enumerations = _count_enumerations(monkeypatch)
    assert routes._lookup_cli_session_metadata(target)["session_id"] == target
    assert enumerations == [], "an unchanged revision must not re-project"


def test_metadata_lookup_honours_a_replaced_enumeration(stubbed_catalog, monkeypatch):
    """A caller that replaces ``get_cli_sessions()`` still drives the lookup.

    The fast path is an optimisation, not a second source of truth: with
    nothing cached, the module-level enumeration decides.
    """
    rows = [{"session_id": "replaced-enumeration-1", "profile": "default"}]

    models.clear_cli_sessions_cache()
    monkeypatch.setattr(models, "_load_cli_sessions_uncached", lambda *a, **kw: [])
    monkeypatch.setattr(routes, "get_cli_sessions", lambda **kw: [dict(r) for r in rows])

    assert routes._lookup_cli_session_metadata("replaced-enumeration-1") == rows[0]
    assert routes._lookup_cli_session_metadata("not-in-the-enumeration") == {}


def test_returned_row_is_a_defensive_copy(stubbed_catalog, monkeypatch):
    """Mutating the returned row must not poison the cached snapshot.

    The reader is made to hand back the cached row object itself — the exact
    aliasing the first cut of this PR had. Without a defensive copy, mutating
    the answer poisons the snapshot for every later request.
    """
    target = ROWS[0]["session_id"]
    assert _prime()

    with models._CLI_SESSIONS_CACHE_LOCK:
        cached_rows = [r for entry in models._CLI_SESSIONS_CACHE.values() for r in entry[2]]
    assert cached_rows, "the positive cache entry should still be there"
    aliased = next(r for r in cached_rows if r["session_id"] == target)
    monkeypatch.setattr(
        models, "get_cli_session_by_id", lambda sid, source, **kw: aliased
    )

    row = models.lookup_cli_session_row(target)
    assert row["title"] == "Fastpath A"
    assert row is not aliased, "the answer aliased the cached snapshot"
    row["title"] = "mutated by the caller"
    assert aliased["title"] == "Fastpath A", (
        "mutating the answer poisoned the cached snapshot"
    )


# ── The snapshot must never outlive what it can vouch for ────────────────


def test_lookup_does_not_answer_across_profiles(stubbed_catalog, monkeypatch):
    """A snapshot primed under one identity must not answer another."""
    target = ROWS[0]["session_id"]
    assert _prime()
    assert models.lookup_cli_session_row(target)["session_id"] == target

    monkeypatch.setattr(
        models, "_cli_sessions_current_cache_key", lambda **kw: ("switched-profile",)
    )
    assert models.lookup_cli_session_row(target) == {}


def test_disabled_cache_never_answers_from_a_snapshot(stubbed_catalog, monkeypatch):
    """With the CLI cache disabled the producer re-projects, so must the lookup.

    ``get_cli_sessions`` gates its whole cache path on ``ttl > 0``. Serving an
    expired snapshot here would answer what the producer would have rebuilt.
    """
    assert _prime()
    monkeypatch.setattr(models, "_CLI_SESSIONS_CACHE_TTL_SECONDS", 0)

    monkeypatch.setattr(models, "get_cli_session_by_id", _authoritative_reader(ROWS))
    assert models.lookup_cli_session_row(ROWS[0]["session_id"]) == {}


@pytest.mark.parametrize("all_profiles", [False, True])
def test_expired_frozen_snapshot_defers_to_the_enumeration(
    stubbed_catalog, monkeypatch, all_profiles
):
    """A frozen key hides writes, so past its freshness window it proves nothing.

    Under the streaming freeze (#4842) the producer's key stops tracking
    state.db for the length of a turn. Two snapshots can share that key
    without describing the same catalog, so once the entry is expired the
    producer would have re-projected — and so must the lookup.
    """
    monkeypatch.setattr(
        models, "_cli_sessions_streaming_freeze_marker", lambda: ("streaming", ("t1",))
    )
    models.clear_cli_sessions_cache()
    assert models.get_cli_sessions(all_profiles=all_profiles)
    _expire_primary()

    monkeypatch.setattr(models, "get_cli_session_by_id", _authoritative_reader(ROWS))
    _block_fallback_reads(monkeypatch)
    enumerations = _count_enumerations(monkeypatch)
    assert routes._lookup_cli_session_metadata(
        ROWS[0]["session_id"], all_profiles=all_profiles
    ) == {}
    assert enumerations == [1]


@pytest.mark.parametrize("all_profiles", [False, True])
def test_fresh_frozen_snapshot_still_answers(stubbed_catalog, monkeypatch, all_profiles):
    """Within its freshness window a frozen snapshot is what the producer serves."""
    monkeypatch.setattr(
        models, "_cli_sessions_streaming_freeze_marker", lambda: ("streaming", ("t1",))
    )
    models.clear_cli_sessions_cache()
    assert models.get_cli_sessions(all_profiles=all_profiles)

    monkeypatch.setattr(models, "get_cli_session_by_id", _authoritative_reader(ROWS))
    enumerations = _count_enumerations(monkeypatch)
    got = models.lookup_cli_session_row(
        ROWS[0]["session_id"], all_profiles=all_profiles
    )
    assert got["session_id"] == ROWS[0]["session_id"]
    assert enumerations == []


# ── The confirmation must actually confirm ───────────────────────────────


def test_hint_without_authoritative_row_falls_through(stubbed_catalog, monkeypatch):
    """An external DELETE: the bounded read answers {} → the enumeration decides."""
    target = ROWS[0]["session_id"]
    assert _prime()
    assert models.lookup_cli_session_row(target)["session_id"] == target

    monkeypatch.setattr(models, "get_cli_session_by_id", lambda sid, source, **kw: {})
    _block_fallback_reads(monkeypatch)
    enumerations = _count_enumerations(monkeypatch)
    assert routes._lookup_cli_session_metadata(target) == {}
    assert enumerations == [1]


def test_hint_confirmed_returns_fresh_authoritative_row(stubbed_catalog, monkeypatch):
    """An external title UPDATE: the confirmed row is the AUTHORITATIVE one.

    The hint selects the candidate; the producer supplies the content.
    """
    target = ROWS[0]["session_id"]
    assert _prime()
    assert models.lookup_cli_session_row(target)["title"] == "Fastpath A"

    fresh = dict(ROWS[0], title="Renamed externally")
    monkeypatch.setattr(
        models, "get_cli_session_by_id", lambda sid, source, **kw: dict(fresh)
    )
    assert models.lookup_cli_session_row(target)["title"] == "Renamed externally"


def test_confirmation_must_reproduce_the_hinted_owner(stubbed_catalog, monkeypatch):
    """A confirmation that returns another profile's row is not a confirmation.

    In a multi-profile install the bounded reader resolves a SID per profile in
    aggregate order, so it can hand back an EARLIER profile's row for the same
    SID even when that profile's own window no longer contains it.
    """
    target = ROWS[0]["session_id"]
    assert _prime()
    assert models.lookup_cli_session_row(target)["profile"] == "default"

    monkeypatch.setattr(
        models,
        "get_cli_session_by_id",
        lambda sid, source, **kw: dict(ROWS[0], profile="alpha"),
    )
    enumerations = _count_enumerations(monkeypatch, [dict(ROWS[0])])
    assert routes._lookup_cli_session_metadata(target)["profile"] == "default"
    assert enumerations == [1], "a foreign-owner confirmation must defer"


def test_unconfirmable_hint_never_answers(stubbed_catalog, monkeypatch):
    """A hint with no resolvable source cannot be confirmed → enumeration wins."""
    rows = [
        {
            "session_id": "cli-fastpath-nosrc",
            "profile": "default",
            "title": "No source",
        }
    ]
    monkeypatch.setattr(
        models, "_load_cli_sessions_uncached", lambda *a, **kw: [dict(r) for r in rows]
    )
    models.clear_cli_sessions_cache()
    assert _prime()

    enumerations = _count_enumerations(
        monkeypatch, [dict(rows[0], title="from enumeration")]
    )
    assert routes._lookup_cli_session_metadata("cli-fastpath-nosrc")["title"] == (
        "from enumeration"
    )
    assert enumerations == [1]


def test_claude_code_hint_does_not_pay_an_unanswerable_read(stubbed_catalog, monkeypatch):
    """A Claude Code hint must not spend a bounded read that can only say ``{}``.

    Those rows are global, and the bounded reader loads with
    ``include_claude_code=False``, so it can neither confirm nor refute them.
    """
    rows = [
        {
            "session_id": "cc-1",
            "profile": "default",
            "raw_source": "claude_code",
            "source_tag": "claude_code",
            "title": "Claude Code",
        }
    ]
    monkeypatch.setattr(
        models, "_load_cli_sessions_uncached", lambda *a, **kw: [dict(r) for r in rows]
    )
    models.clear_cli_sessions_cache()
    assert _prime()

    reads = []
    monkeypatch.setattr(
        models, "get_cli_session_by_id", lambda *a, **kw: reads.append(1) or {}
    )
    enumerations = _count_enumerations(monkeypatch, [dict(rows[0])])
    assert routes._lookup_cli_session_metadata("cc-1")["session_id"] == "cc-1"
    assert reads == [], "the bounded reader cannot answer a Claude Code row"
    assert enumerations == [1]


# ── Enumeration-equivalence edges ────────────────────────────────────────


def test_lookup_keeps_the_enumerators_exact_id_comparison(stubbed_catalog):
    """Ids compare exactly as the enumeration compares them.

    ``_lookup_cli_session_metadata`` compares ``row["session_id"] == session_id``
    with no normalisation, so a padded string or a non-string id is a miss
    there. Normalising here would answer ids the enumeration never matches.
    """
    assert _prime()
    for sid in (" cli-fastpath-a ", "CLI-FASTPATH-A", 123, "", None):
        assert models.lookup_cli_session_row(sid) == {}, sid


def test_aggregate_duplicate_sid_first_match_wins(stubbed_catalog, monkeypatch):
    """Duplicate SIDs answer with the FIRST row, as the enumeration scans."""
    dup_first = dict(ROWS[0], title="first")
    dup_last = dict(ROWS[0], title="last")
    rows = [dup_first, dup_last, ROWS[1]]
    monkeypatch.setattr(
        models, "_load_cli_sessions_uncached", lambda *a, **kw: [dict(r) for r in rows]
    )
    models.clear_cli_sessions_cache()
    enumerated = _prime()
    expected = next(r for r in enumerated if r["session_id"] == ROWS[0]["session_id"])
    assert expected["title"] == "first"

    seen = []

    def _echo(sid, source, **kw):
        seen.append((sid, source))
        return next(r for r in rows if r["title"] == "first")

    monkeypatch.setattr(models, "get_cli_session_by_id", _echo)
    got = models.lookup_cli_session_row(ROWS[0]["session_id"])
    assert got["title"] == "first"
    assert seen == [(ROWS[0]["session_id"], "cli")]
