"""Single-session CLI metadata lookups must not re-project the whole catalog.

perf(webui/session-load-latency): ``_lookup_cli_session_metadata()`` answered by
calling ``get_cli_sessions()`` — a projection over the whole agent ``state.db``
— and then scanning the result. That projection costs seconds once the database
grows, and its cache key embeds a live ``state.db`` fingerprint, so an agent
write makes the cached snapshot unreachable again. The gateway SSE re-announces
the active session on every tick, so a single page load issued three of these
lookups and paid the projection each time.

The fix is a hint-plus-confirm contract (per the #8412 review round 2): a cached
snapshot is a HINT, never a second source of truth. A hint hit is confirmed by
a bounded authoritative single-SID read scoped to the hinted row's own source;
anything the producer cannot confirm falls through to the original
enumeration. These tests pin both halves of that contract:

1. the fast path answers without re-projecting (the perf win), and
2. the fast path cannot outlive the producer's scope or freshness — cross
   profile, external delete/update, expired snapshots, duplicate aggregation
   order, and defensive copies all answer exactly like the enumeration.
"""
import threading

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
    """A bounded-reader stand-in that mirrors get_cli_session_by_id's contract."""

    def _read(sid, source, **kwargs):
        for row in rows:
            if row.get("session_id") == sid:
                return dict(row)
        return {}

    return _read


def test_lookup_matches_full_enumeration(stubbed_catalog):
    """The fast path returns exactly the row the enumeration returns."""
    expected = {row["session_id"]: row for row in models.get_cli_sessions()}
    assert expected, "fixture should produce rows"

    for session_id, row in expected.items():
        assert models.lookup_cli_session_row(session_id) == row

    # Unknown and empty ids keep the enumeration's "not found" answer ({}), which
    # every call site already treats as "no CLI metadata".
    assert models.lookup_cli_session_row("no-such-session-id") == {}
    assert models.lookup_cli_session_row("") == {}


def test_metadata_lookup_does_not_enumerate_when_snapshot_is_cached(
    stubbed_catalog, monkeypatch
):
    """A cached catalog answers the lookup without re-running the projection.

    This is the behavioural assertion: on the pre-fix code the second call below
    re-entered ``get_cli_sessions()`` (the spy raises), because the lookup had no
    cache path at all. It deliberately primes through the public route helper so
    it fails for the right reason on old code instead of tripping over a missing
    name.
    """
    target = ROWS[0]["session_id"]

    # Prime the catalog through the same helper the request path uses.
    assert routes._lookup_cli_session_metadata(target)["session_id"] == target

    enumerations = []

    def _boom(*args, **kwargs):
        enumerations.append(1)
        raise AssertionError(
            "_lookup_cli_session_metadata re-ran the CLI catalog projection; "
            "a cached snapshot must answer a single-session lookup"
        )

    # The pre-fix implementation resolved this name on every call.
    monkeypatch.setattr(routes, "get_cli_sessions", _boom)
    # The bounded confirmation read is NOT the projection; keep it alive.
    monkeypatch.setattr(models, "get_cli_session_by_id", _authoritative_reader(ROWS))

    meta = routes._lookup_cli_session_metadata(target)
    assert meta["session_id"] == target
    assert enumerations == []


def test_metadata_lookup_honours_a_replaced_enumeration(stubbed_catalog, monkeypatch):
    """A caller that replaces ``get_cli_sessions()`` must still drive the lookup.

    The fast path is an optimisation, not a second source of truth: when nothing
    is cached, the module-level enumeration decides. Upstream callers rely on
    this — ``test_git_info_untrusted_state_db_workspace_is_not_inspected`` stubs
    ``routes.get_cli_sessions`` and expects ``/api/git-info`` to see the row.
    Answering ``{}`` here turned that into a 404 "Session not found".
    """
    rows = [{"session_id": "replaced-enumeration-1", "profile": "default"}]

    # Cache nothing, so the fast path cannot answer and the fallback must run.
    models.clear_cli_sessions_cache()
    monkeypatch.setattr(models, "_load_cli_sessions_uncached", lambda *a, **kw: [])
    monkeypatch.setattr(routes, "get_cli_sessions", lambda **kw: [dict(r) for r in rows])

    assert routes._lookup_cli_session_metadata("replaced-enumeration-1") == rows[0]

    # And a genuinely absent id still reports "no CLI metadata".
    assert routes._lookup_cli_session_metadata("not-in-the-enumeration") == {}


def test_lookup_survives_primary_cache_eviction(stubbed_catalog, monkeypatch):
    """A primary-cache eviction must not force a re-projection.

    The primary map is bounded; the last-known-good map keeps the same rows
    under the stable identity. While the catalog revision is unchanged that
    snapshot still describes the current catalog, so it may answer.
    """
    target = ROWS[1]["session_id"]
    assert models.lookup_cli_session_row(target)["session_id"] == target

    # Drop every primary-cache entry while keeping the last-known-good map.
    with models._CLI_SESSIONS_CACHE_LOCK:
        models._CLI_SESSIONS_CACHE.clear()

    # The LKG hint is confirmed through the bounded reader, not the projection.
    monkeypatch.setattr(models, "get_cli_session_by_id", _authoritative_reader(ROWS))

    enumerations = []
    monkeypatch.setattr(
        models, "get_cli_sessions", lambda *a, **kw: enumerations.append(1) or []
    )

    row = models.lookup_cli_session_row(target)
    assert row["session_id"] == target
    assert enumerations == [], (
        "the stable-identity snapshot should have answered this lookup"
    )


def test_last_known_good_does_not_answer_a_changed_revision(
    stubbed_catalog, monkeypatch
):
    """A snapshot from a different revision is a fallback, not a catalog.

    The last-known-good key deliberately drops the state.db revision, so the
    entry can still be found after a write. Its rows, however, may have left
    the producer's window since — membership is only proven when the revision
    it was taken at still matches, otherwise the enumeration must decide.
    """
    target = ROWS[0]["session_id"]
    assert models.lookup_cli_session_row(target)["session_id"] == target

    # Drop the primary cache and age the LKG entry to a different revision.
    with models._CLI_SESSIONS_CACHE_LOCK:
        models._CLI_SESSIONS_CACHE.clear()
        for key, entry in list(models._CLI_SESSIONS_LAST_KNOWN_GOOD.items()):
            models._CLI_SESSIONS_LAST_KNOWN_GOOD[key] = (
                entry[0], entry[1], ("stale-revision",),
            )

    # Even a willing confirmation reader must not rescue the stale snapshot.
    monkeypatch.setattr(models, "get_cli_session_by_id", _authoritative_reader(ROWS))
    enumerations = []
    monkeypatch.setattr(
        models, "get_cli_sessions", lambda *a, **kw: enumerations.append(1) or []
    )

    assert models.lookup_cli_session_row(target) == {}
    assert enumerations == [1], "a stale-revision snapshot must defer to enumeration"


def test_returned_row_is_a_defensive_copy(stubbed_catalog, monkeypatch):
    """Mutating the returned row must not poison the cached snapshot.

    The lookup is served from a *primed positive cache entry*, and the live
    reader is made to hand back the cached row object itself — the exact
    aliasing the first cut of this PR had. Without a defensive copy, mutating
    the answer poisons the snapshot for every later request.
    """
    target = ROWS[0]["session_id"]
    models.get_cli_sessions()  # prime a positive cache entry

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


# ── Review round 2: the hint must never outlive the producer's scope ───────


def test_lookup_does_not_answer_across_profiles(stubbed_catalog, monkeypatch):
    """A snapshot primed under identity A must not answer a lookup under B.

    The producer keys its cache by the resolved profile identity; the fast path
    may only read hints stored under the CALLER's exact identity. Simulate the
    A→B switch by priming, then swapping the identity resolver's answer: the
    exact-key primary entry and the stable-key LKG entry both become
    unreachable and the lookup must fall through to the enumeration rather
    than serve A's row.
    """
    target = ROWS[0]["session_id"]
    assert models.lookup_cli_session_row(target)["session_id"] == target

    def _other_identity(*, all_profiles):
        return ("switched-profile",), lambda: []

    monkeypatch.setattr(models, "_cli_sessions_lookup_identity", _other_identity)

    # A stale-identity LKG entry must not be reachable from the new identity.
    assert models._CLI_SESSIONS_LAST_KNOWN_GOOD, "fixture should have primed LKG"

    enumerations = []
    monkeypatch.setattr(
        models,
        "get_cli_sessions",
        lambda *a, **kw: enumerations.append(1) or [],
    )
    assert models.lookup_cli_session_row(target) == {}
    assert enumerations == [1], "must fall through to the enumeration, not the hint"


def test_hint_without_authoritative_row_falls_through(stubbed_catalog, monkeypatch):
    """An external DELETE: the bounded authoritative read answers {} → miss.

    The hint still carries the row; the producer, read live from state.db,
    does not. The enumeration — the source of truth — gets the last word, so
    the caller sees the enumeration's answer rather than the stale hint.
    """
    target = ROWS[0]["session_id"]
    assert models.lookup_cli_session_row(target)["session_id"] == target

    monkeypatch.setattr(models, "get_cli_session_by_id", lambda sid, source, **kw: {})
    # A fresh enumeration no longer contains the deleted session.
    monkeypatch.setattr(models, "get_cli_sessions", lambda *a, **kw: [])

    # The fast path must not serve the stale hint; it defers to enumeration.
    # (``routes`` additionally retries its own fallback sources, unchanged from
    # master, so this asserts the unit's contract rather than the route's.)
    assert models.lookup_cli_session_row(target) == {}


def test_hint_confirmed_returns_fresh_authoritative_row(stubbed_catalog, monkeypatch):
    """An external title UPDATE: the confirmed row is the AUTHORITATIVE one.

    Even when the cached snapshot still carries the old title, a positive
    confirmation returns the live row from the bounded read — the hint selects
    the candidate, the producer supplies the content.
    """
    target = ROWS[0]["session_id"]
    assert models.lookup_cli_session_row(target)["title"] == "Fastpath A"

    fresh = dict(ROWS[0], title="Renamed externally")
    monkeypatch.setattr(
        models, "get_cli_session_by_id", lambda sid, source, **kw: dict(fresh)
    )
    assert models.lookup_cli_session_row(target)["title"] == "Renamed externally"


def test_expired_snapshot_hint_still_confirmed(stubbed_catalog, monkeypatch):
    """An expired-but-stamp-valid snapshot is a hint, not an answer.

    Expiry only demotes the snapshot from "answer" to "hint": the lookup still
    works (perf) but every hit passes the authoritative gate, so an external
    change across the TTL boundary is caught on the next request instead of
    being served stale forever.
    """
    target = ROWS[0]["session_id"]
    assert models.lookup_cli_session_row(target)["session_id"] == target

    # Force every primary entry to be long-expired.
    with models._CLI_SESSIONS_CACHE_LOCK:
        for key in list(models._CLI_SESSIONS_CACHE):
            entry = models._CLI_SESSIONS_CACHE[key]
            models._CLI_SESSIONS_CACHE[key] = (0.0, entry[1], entry[2])

    confirmed_calls = []

    def _authoritative(sid, source, **kw):
        confirmed_calls.append((sid, source))
        return {}  # the live producer says: gone

    monkeypatch.setattr(models, "get_cli_session_by_id", _authoritative)
    monkeypatch.setattr(models, "_load_cli_sessions_uncached", lambda *a, **kw: [])
    monkeypatch.setattr(routes, "get_cli_sessions", lambda **kw: [])

    assert routes._lookup_cli_session_metadata(target) == {}
    assert (target, "cli") in confirmed_calls


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
    assert models.get_cli_sessions()  # prime

    # The row exists in the snapshot but carries no source: not confirmable, so
    # the fast path must not answer it. The enumeration decides — stubbed here
    # to a distinguishable title so the test can tell which path answered.
    monkeypatch.setattr(
        models,
        "get_cli_sessions",
        lambda *a, **kw: [dict(rows[0], title="from enumeration")],
    )
    assert models.lookup_cli_session_row("cli-fastpath-nosrc")["title"] == "from enumeration"


def test_aggregate_duplicate_sid_first_match_wins(stubbed_catalog, monkeypatch):
    """Duplicate SIDs in an aggregate answer with the FIRST row, as enumeration does."""
    dup_first = dict(ROWS[0], title="first")
    dup_last = dict(ROWS[0], title="last")
    rows = [dup_first, dup_last, ROWS[1]]
    monkeypatch.setattr(
        models, "_load_cli_sessions_uncached", lambda *a, **kw: [dict(r) for r in rows]
    )
    models.clear_cli_sessions_cache()
    enumerated = models.get_cli_sessions()
    expected = next(r for r in enumerated if r["session_id"] == ROWS[0]["session_id"])
    assert expected["title"] == "first"

    # The fast path confirms the FIRST hinted row. The bounded reader proves
    # which candidate was handed to it by the hint scan.
    seen = []

    def _echo(sid, source, **kw):
        seen.append((sid, source))
        return next(r for r in rows if r["title"] == "first")

    monkeypatch.setattr(models, "get_cli_session_by_id", _echo)
    got = models.lookup_cli_session_row(ROWS[0]["session_id"])
    assert got["title"] == "first"
    assert seen == [(ROWS[0]["session_id"], "cli")]


def test_startup_warm_primes_the_default_identity(monkeypatch):
    """The async warm writes the identity the default lookup will read.

    A warm that primes a different identity than lookups resolve is pure waste:
    the first page still pays the cold projection. This pins warm→lookup
    agreement without touching a real state.db.
    """
    models.clear_cli_sessions_cache()
    rows = [
        {"session_id": "warm-prime-1", "profile": "default", "raw_source": "cli"}
    ]
    monkeypatch.setattr(
        models, "_load_cli_sessions_uncached", lambda *a, **kw: [dict(r) for r in rows]
    )

    done = threading.Event()
    real_load = models._load_and_cache_cli_sessions

    def _watchful_load(*args, **kwargs):
        result = real_load(*args, **kwargs)
        done.set()
        return result

    monkeypatch.setattr(models, "_load_and_cache_cli_sessions", _watchful_load)
    assert models.warm_cli_sessions_cache_async() is True
    assert done.wait(10), "background warm did not run"

    assert (
        models.lookup_cli_session_row("warm-prime-1")["session_id"] == "warm-prime-1"
    )


def test_warm_coalesces_concurrent_calls(stubbed_catalog, monkeypatch):
    """Two warm calls while one is in flight schedule exactly one rebuild."""
    started = threading.Event()
    release = threading.Event()

    def _slow_load(*args, **kwargs):
        started.set()
        assert release.wait(10)
        return []

    monkeypatch.setattr(models, "_load_and_cache_cli_sessions", _slow_load)
    assert models.warm_cli_sessions_cache_async() is True
    assert started.wait(10)
    assert models.warm_cli_sessions_cache_async() is False
    release.set()


def test_hint_identity_follows_the_active_profile(stubbed_catalog, monkeypatch):
    """The hint identity is resolved LIVE per request (no stale-profile pinning).

    Prime under profile alpha, then make the identity resolver answer bravo
    without clearing any cache: the bravo lookup must miss the hint rather than
    answer from alpha's snapshot, because both the exact primary key and the
    stable LKG key are recomputed per request.
    """
    target = ROWS[0]["session_id"]
    assert models.lookup_cli_session_row(target)["session_id"] == target

    real_identity = models._cli_sessions_lookup_identity

    def _identity(*, all_profiles):
        key, load = real_identity(all_profiles=all_profiles)
        return key + ("bravo",), load

    monkeypatch.setattr(models, "_cli_sessions_lookup_identity", _identity)

    # The bounded reader would happily confirm — yet nothing may be confirmed,
    # because no hint is reachable under the new identity.
    monkeypatch.setattr(models, "get_cli_session_by_id", _authoritative_reader(ROWS))
    enumerated = []
    monkeypatch.setattr(
        models, "get_cli_sessions", lambda *a, **kw: enumerated.append(1) or []
    )
    assert models.lookup_cli_session_row(target) == {}
    assert enumerated == [1]


# ── Review round 2 (co-review): enumeration-compatible edges ──────────────


def test_lookup_keeps_the_enumerators_exact_id_comparison(stubbed_catalog):
    """Ids must compare exactly as the enumeration compares them.

    ``_lookup_cli_session_metadata`` compares ``row["session_id"] == session_id``
    with no normalisation, so a padded string or a non-string id is a miss
    there. Normalising here would make the fast path answer ids the enumeration
    never matches.
    """
    models.get_cli_sessions()  # prime
    for sid in (" cli-fastpath-a ", "CLI-FASTPATH-A", 123, "", None):
        assert models.lookup_cli_session_row(sid) == {}, sid


def test_warm_joins_the_producers_single_flight(stubbed_catalog, monkeypatch):
    """A warm that starts during a request's projection must not stack a second one.

    The warm exists to move the cold projection off the request path; if it
    bypassed ``_cli_sessions_cache_claim_rebuild`` it would instead run a
    *second* projection alongside the one already in flight.
    """
    entered = threading.Event()
    release = threading.Event()
    real_load = models._load_cli_sessions_uncached
    calls = []

    def _blocked(*args, **kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(10)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(models, "_load_cli_sessions_uncached", _blocked)
    try:
        assert models.warm_cli_sessions_cache_async() is True
        assert entered.wait(10)
        # The warm is inside the producer's single-flight claim.
        assert models._CLI_SESSIONS_CACHE_INFLIGHT, "warm bypassed single-flight"
        # A second warm coalesces on the warm token instead of queueing another.
        assert models.warm_cli_sessions_cache_async() is False
    finally:
        release.set()
        for _ in range(100):
            if not models._CLI_SESSIONS_WARM_INFLIGHT:
                break
            threading.Event().wait(0.05)
    assert len(calls) == 1


# ── Review round 3 (co-review): a frozen key does not prove membership ─────


def _freeze(models, marker):
    models._cli_sessions_streaming_freeze_marker = lambda: marker


def test_expired_frozen_snapshot_defers_to_the_enumeration(stubbed_catalog, monkeypatch):
    """A frozen key hides writes, so past its freshness window it proves nothing.

    Under the streaming freeze (#4842) the producer's key stops tracking
    state.db for the length of a turn. Two snapshots can therefore share that
    key without describing the same catalog, so once the entry is expired the
    producer would have re-projected — and so must the lookup.
    """
    monkeypatch.setattr(
        models, "_cli_sessions_streaming_freeze_marker", lambda: ("streaming", ("t1",))
    )
    models.clear_cli_sessions_cache()
    assert models.get_cli_sessions()  # prime under the frozen key

    with models._CLI_SESSIONS_CACHE_LOCK:
        for key in list(models._CLI_SESSIONS_CACHE):
            entry = models._CLI_SESSIONS_CACHE[key]
            models._CLI_SESSIONS_CACHE[key] = (0.0, entry[1], entry[2])

    monkeypatch.setattr(models, "get_cli_session_by_id", _authoritative_reader(ROWS))
    enumerations = []
    monkeypatch.setattr(
        models, "get_cli_sessions", lambda *a, **kw: enumerations.append(1) or []
    )
    assert models.lookup_cli_session_row(ROWS[0]["session_id"]) == {}
    assert enumerations == [1]


def test_fresh_frozen_snapshot_still_answers(stubbed_catalog, monkeypatch):
    """Within its freshness window a frozen snapshot is what the producer serves."""
    monkeypatch.setattr(
        models, "_cli_sessions_streaming_freeze_marker", lambda: ("streaming", ("t1",))
    )
    models.clear_cli_sessions_cache()
    assert models.get_cli_sessions()  # prime under the frozen key

    monkeypatch.setattr(models, "get_cli_session_by_id", _authoritative_reader(ROWS))
    enumerations = []
    monkeypatch.setattr(
        models, "get_cli_sessions", lambda *a, **kw: enumerations.append(1) or []
    )
    assert models.lookup_cli_session_row(ROWS[0]["session_id"])["session_id"] == (
        ROWS[0]["session_id"]
    )
    assert enumerations == []


def test_frozen_last_known_good_never_proves_membership(stubbed_catalog, monkeypatch):
    """The TTL-less last-known-good map may not be used under a frozen key.

    A frozen entry there can be arbitrarily old — even from an earlier
    streaming episode with the same active-stream set.
    """
    monkeypatch.setattr(
        models, "_cli_sessions_streaming_freeze_marker", lambda: ("streaming", ("t1",))
    )
    models.clear_cli_sessions_cache()
    assert models.get_cli_sessions()  # prime: primary + LKG, both frozen

    # Drop the primary so only the LKG could answer.
    with models._CLI_SESSIONS_CACHE_LOCK:
        models._CLI_SESSIONS_CACHE.clear()

    monkeypatch.setattr(models, "get_cli_session_by_id", _authoritative_reader(ROWS))
    enumerations = []
    monkeypatch.setattr(
        models, "get_cli_sessions", lambda *a, **kw: enumerations.append(1) or []
    )
    assert models.lookup_cli_session_row(ROWS[0]["session_id"]) == {}
    assert enumerations == [1]


def test_claude_code_hint_does_not_pay_an_unanswerable_read(stubbed_catalog, monkeypatch):
    """A Claude Code hint must not spend a bounded read that can only say ``{}``.

    Those rows are global rather than profile-owned, and the bounded reader
    loads with ``include_claude_code=False``, so it can neither confirm nor
    refute them. Reading anyway would add a pointless round trip to every
    Claude Code lookup while the enumeration decides regardless.
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
    assert models.get_cli_sessions()  # prime

    reads = []
    monkeypatch.setattr(
        models, "get_cli_session_by_id", lambda *a, **kw: reads.append(1) or {}
    )
    enumerations = []
    monkeypatch.setattr(
        models,
        "get_cli_sessions",
        lambda *a, **kw: enumerations.append(1) or [dict(rows[0])],
    )

    assert models.lookup_cli_session_row("cc-1")["session_id"] == "cc-1"
    assert reads == [], "the bounded reader cannot answer a Claude Code row"
    assert enumerations == [1], "the enumeration must decide"


def test_disabled_cache_never_answers_from_a_snapshot(stubbed_catalog, monkeypatch):
    """With the CLI cache disabled the producer re-projects, so must the lookup.

    ``get_cli_sessions`` gates its whole cache path on ``ttl > 0``. Serving an
    expired snapshot here would answer what the producer would have rebuilt.
    """
    models.get_cli_sessions()  # prime a real entry
    monkeypatch.setattr(models, "_CLI_SESSIONS_CACHE_TTL_SECONDS", 0)

    monkeypatch.setattr(models, "get_cli_session_by_id", _authoritative_reader(ROWS))
    enumerations = []
    monkeypatch.setattr(
        models, "get_cli_sessions", lambda *a, **kw: enumerations.append(1) or []
    )
    assert models.lookup_cli_session_row(ROWS[0]["session_id"]) == {}
    assert enumerations == [1]


def test_confirmation_must_reproduce_the_hinted_owner(stubbed_catalog, monkeypatch):
    """A confirmation that returns another profile's row is not a confirmation.

    In a multi-profile install the bounded reader resolves a SID per profile in
    aggregate order, so it can hand back an EARLIER profile's row for the same
    SID even when that profile's own window no longer contains it. The hint's
    owner is the aggregate enumeration's answer, so a mismatch must fall
    through rather than substitute the other owner.
    """
    target = ROWS[0]["session_id"]
    models.get_cli_sessions()  # prime
    assert models.lookup_cli_session_row(target)["profile"] == "default"

    monkeypatch.setattr(
        models,
        "get_cli_session_by_id",
        lambda sid, source, **kw: dict(ROWS[0], profile="alpha"),
    )
    enumerations = []
    monkeypatch.setattr(
        models,
        "get_cli_sessions",
        lambda *a, **kw: enumerations.append(1) or [dict(ROWS[0])],
    )
    assert models.lookup_cli_session_row(target)["profile"] == "default"
    assert enumerations == [1], "a foreign-owner confirmation must defer"
