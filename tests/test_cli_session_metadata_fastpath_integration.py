"""#8412 review: real-producer/consumer regressions for the metadata fast path.

The unit tests in ``test_cli_session_metadata_fastpath.py`` stub the catalog
loader, which the reviewer correctly noted can miss the transitions that matter
("the submitted four tests pass but mock the loader and miss these
transitions"). These tests instead use:

* **real SQLite profile databases** (two of them, so cross-profile scope is
  exercised against actual files, not a boolean),
* the **real catalog/cache producer** ``get_cli_sessions()`` with its real
  cache-key construction and last-known-good map,
* the **real request-profile binding** (``api.profiles.get_active_hermes_home``
  / ``get_active_profile_name``, which ``_resolve_cli_sessions_context``
  imports at call time), and
* the **real route consumer** ``_lookup_cli_session_metadata``.

No loader is mocked. Every assertion compares the fast path against what the
enumeration itself returns for the same scope.
"""
import sqlite3
from pathlib import Path

import pytest

from api import models, routes


def _make_state_db(path: Path, rows) -> None:
    """Create a real agent-shaped state.db holding ``rows``.

    ``rows`` is a sequence of ``(session_id, source, title)``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_version (version INTEGER);
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                source TEXT,
                user_id TEXT,
                model TEXT,
                model_config TEXT,
                system_prompt TEXT,
                parent_session_id TEXT,
                started_at REAL,
                ended_at REAL,
                end_reason TEXT,
                message_count INTEGER DEFAULT 0,
                tool_call_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                billing_provider TEXT,
                billing_base_url TEXT,
                billing_mode TEXT,
                estimated_cost_usd REAL,
                actual_cost_usd REAL,
                cost_status TEXT,
                cost_source TEXT,
                pricing_version TEXT,
                title TEXT,
                api_call_count INTEGER DEFAULT 0,
                handoff_state TEXT,
                handoff_platform TEXT,
                handoff_error TEXT,
                cwd TEXT,
                rewind_count INTEGER DEFAULT 0,
                archived INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp REAL,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_call_count INTEGER DEFAULT 0
            );
            """
        )
        for offset, (sid, source, title) in enumerate(rows):
            started = 1781024055.0 + offset
            conn.execute(
                "INSERT OR REPLACE INTO sessions "
                "(id, source, model, message_count, started_at, title, cwd) "
                "VALUES (?, ?, ?, 2, ?, ?, '/tmp')",
                (sid, source, "test-model", started, title),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) "
                "VALUES (?, 'user', 'hi', ?)",
                (sid, started),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) "
                "VALUES (?, 'assistant', 'hello', ?)",
                (sid, started + 0.5),
            )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    """Two real profiles on disk, with the active one switchable mid-test."""
    homes = {
        "alpha": tmp_path / "profiles" / "alpha",
        "bravo": tmp_path / "profiles" / "bravo",
    }
    for home in homes.values():
        home.mkdir(parents=True, exist_ok=True)
    index_path = tmp_path / "sessions" / "_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("[]", encoding="utf-8")

    state = {"active": "alpha"}

    monkeypatch.setattr(
        "api.profiles.get_active_hermes_home", lambda: homes[state["active"]]
    )
    monkeypatch.setattr(
        "api.profiles.get_active_profile_name", lambda: state["active"]
    )
    # ``all_profiles`` resolves contexts through a different seam: the profile
    # root plus a per-name home resolver. Point both at the temp tree so the
    # aggregate really scans alpha and bravo, never the host's real profiles.
    monkeypatch.setattr("api.profiles._profiles_root", lambda: tmp_path / "profiles")
    monkeypatch.setattr(
        "api.profiles.get_hermes_home_for_profile",
        lambda name: homes[str(name or "default")]
        if str(name or "default") in homes
        else (tmp_path / "profiles" / str(name)),
    )
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", index_path)
    # The project catalog is an input to the projection (project_id resolution
    # and the assigned-recovery pass), so the fast path's identity has to see
    # it too. Point it at a temp file and start it empty.
    projects_file = tmp_path / "projects.json"
    projects_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(models, "PROJECTS_FILE", projects_file)
    monkeypatch.setattr(models, "_projects_migrated", True)
    # No Claude Code scan: it is global rather than profile-owned and would add
    # rows from the host's real ~/.claude to every assertion here.
    monkeypatch.setattr(models, "_default_claude_code_projects_dir", lambda: None)
    models.clear_cli_sessions_cache()
    try:
        yield state, homes
    finally:
        models.clear_cli_sessions_cache()


def _switch(state, name):
    state["active"] = name


# ── 1. A→B: one profile's snapshot must not answer another profile ─────────


def test_primed_profile_does_not_answer_a_switch(profiles):
    """Prime alpha's catalog, switch to bravo: alpha-only SID must miss."""
    state, homes = profiles
    _make_state_db(homes["alpha"] / "state.db", [("alpha-only", "cli", "Alpha only")])
    _make_state_db(homes["bravo"] / "state.db", [("bravo-only", "cli", "Bravo only")])

    # Prime through the real producer under alpha.
    assert [r["session_id"] for r in models.get_cli_sessions()] == ["alpha-only"]
    assert models.lookup_cli_session_row("alpha-only")["session_id"] == "alpha-only"

    _switch(state, "bravo")

    # The route consumer must agree with the enumeration for bravo's scope.
    enumerated = {r["session_id"] for r in models.get_cli_sessions()}
    assert enumerated == {"bravo-only"}
    assert models.lookup_cli_session_row("alpha-only") == {}
    assert routes._lookup_cli_session_metadata("alpha-only") == {}
    assert models.lookup_cli_session_row("bravo-only")["session_id"] == "bravo-only"


def test_same_sid_in_both_profiles_resolves_to_the_active_one(profiles):
    """A SID present in both profiles must resolve to the ACTIVE profile's row."""
    state, homes = profiles
    _make_state_db(homes["alpha"] / "state.db", [("shared-sid", "cli", "Alpha title")])
    _make_state_db(homes["bravo"] / "state.db", [("shared-sid", "cli", "Bravo title")])

    assert models.get_cli_sessions()[0]["title"] == "Alpha title"
    assert models.lookup_cli_session_row("shared-sid")["title"] == "Alpha title"

    _switch(state, "bravo")

    # Master returns bravo's row here; the fast path must not keep serving
    # alpha's title, and must not report a profile mismatch for its own scope.
    assert models.get_cli_sessions()[0]["title"] == "Bravo title"
    assert models.lookup_cli_session_row("shared-sid")["title"] == "Bravo title"
    meta = routes._lookup_cli_session_metadata("shared-sid")
    assert meta["title"] == "Bravo title"
    assert meta["profile"] == "bravo"


# ── 2. Primary eviction → last-known-good must still be scoped ─────────────


def test_primary_eviction_keeps_the_last_known_good_hint_scoped(profiles):
    """Evicting the primary cache must not let LKG answer a foreign scope."""
    state, homes = profiles
    _make_state_db(homes["alpha"] / "state.db", [("alpha-only", "cli", "Alpha only")])
    _make_state_db(homes["bravo"] / "state.db", [("bravo-only", "cli", "Bravo only")])

    models.get_cli_sessions()  # prime alpha: primary + LKG
    with models._CLI_SESSIONS_CACHE_LOCK:
        models._CLI_SESSIONS_CACHE.clear()  # what a fingerprint churn looks like
    # The LKG entry for alpha survives; it must still answer alpha.
    assert models.lookup_cli_session_row("alpha-only")["session_id"] == "alpha-only"

    _switch(state, "bravo")
    # bravo has its own LKG only after it projects; alpha's must not be reachable.
    assert models.lookup_cli_session_row("alpha-only") == {}


# ── 3. External writes: delete and title update ────────────────────────────


def test_external_delete_is_not_served_from_the_snapshot(profiles):
    """A row deleted by another process must not keep answering 200."""
    state, homes = profiles
    db = homes["alpha"] / "state.db"
    _make_state_db(db, [("victim", "cli", "Victim"), ("keeper", "cli", "Keeper")])

    models.get_cli_sessions()  # prime
    assert models.lookup_cli_session_row("victim")["session_id"] == "victim"

    # An external CLI deletes the session (no in-process cache clear).
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("DELETE FROM messages WHERE session_id = 'victim'")
        conn.execute("DELETE FROM sessions WHERE id = 'victim'")
        conn.commit()
    finally:
        conn.close()

    # The bounded confirm read sees the truth; the enumeration agrees.
    assert models.lookup_cli_session_row("victim") == {}
    assert "victim" not in {r["session_id"] for r in models.get_cli_sessions()}


def test_external_title_update_reaches_the_caller(profiles):
    """An externally renamed session must not keep serving the stale title."""
    state, homes = profiles
    db = homes["alpha"] / "state.db"
    _make_state_db(db, [("rename-me", "cli", "Old title")])

    models.get_cli_sessions()  # prime
    assert models.lookup_cli_session_row("rename-me")["title"] == "Old title"

    conn = sqlite3.connect(str(db))
    try:
        conn.execute("UPDATE sessions SET title = 'New title' WHERE id = 'rename-me'")
        conn.commit()
    finally:
        conn.close()

    assert models.lookup_cli_session_row("rename-me")["title"] == "New title"


# ── 4. Negative → new insert, and explicit invalidation ───────────────────


def test_new_insert_after_a_negative_lookup_is_found(profiles):
    """A session created after a miss must be found (negative→insert)."""
    state, homes = profiles
    db = homes["alpha"] / "state.db"
    _make_state_db(db, [("existing", "cli", "Existing")])

    assert models.lookup_cli_session_row("late-arrival") == {}

    # The agent inserts a new session; its write churns the state.db fingerprint
    # (the fingerprint is derived from message rows, as in a real turn).
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "INSERT INTO sessions (id, source, model, message_count, started_at, title, cwd) "
            "VALUES ('late-arrival', 'cli', 'test-model', 1, 1781030000.0, 'Late', '/tmp')"
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) "
            "VALUES ('late-arrival', 'user', 'hi', 1781030000.0)"
        )
        conn.commit()
    finally:
        conn.close()

    assert models.lookup_cli_session_row("late-arrival")["session_id"] == "late-arrival"


def test_explicit_invalidation_drops_every_hint(profiles):
    """``clear_cli_sessions_cache()`` must leave no hint reachable."""
    state, homes = profiles
    _make_state_db(homes["alpha"] / "state.db", [("keep-me", "cli", "Keep")])

    models.get_cli_sessions()
    assert models.lookup_cli_session_row("keep-me")["session_id"] == "keep-me"

    models.clear_cli_sessions_cache()
    with models._CLI_SESSIONS_CACHE_LOCK:
        assert not models._CLI_SESSIONS_CACHE
        assert not models._CLI_SESSIONS_LAST_KNOWN_GOOD

    # Still correct — answered by a fresh enumeration, not a stale hint.
    assert models.lookup_cli_session_row("keep-me")["session_id"] == "keep-me"


# ── 5. Aggregate ordering with a duplicate SID ────────────────────────────


def test_all_profiles_duplicate_sid_uses_enumeration_order(profiles):
    """With ``all_profiles``, the fast path must answer enumeration's first row."""
    state, homes = profiles
    _make_state_db(homes["alpha"] / "state.db", [("dup-sid", "cli", "Alpha copy")])
    _make_state_db(homes["bravo"] / "state.db", [("dup-sid", "cli", "Bravo copy")])

    enumerated = models.get_cli_sessions(all_profiles=True)
    first = next(r for r in enumerated if r["session_id"] == "dup-sid")
    assert first["title"] == "Alpha copy"  # active profile first in the aggregate

    got = models.lookup_cli_session_row("dup-sid", all_profiles=True)
    assert got["title"] == first["title"]


# ── 6. Startup warm → named-profile lookup ────────────────────────────────


def test_startup_warm_then_named_profile_lookup(profiles):
    """The startup warm primes the ACTIVE identity; a later switch must not
    inherit it, and must not be blocked by it."""
    import threading

    state, homes = profiles
    _make_state_db(homes["alpha"] / "state.db", [("alpha-only", "cli", "Alpha only")])
    _make_state_db(homes["bravo"] / "state.db", [("bravo-only", "cli", "Bravo only")])

    done = threading.Event()
    real_load = models._load_and_cache_cli_sessions

    def _watchful(*args, **kwargs):
        try:
            return real_load(*args, **kwargs)
        finally:
            done.set()

    original = models._load_and_cache_cli_sessions
    models._load_and_cache_cli_sessions = _watchful
    try:
        assert models.warm_cli_sessions_cache_async() is True
        assert done.wait(15), "startup warm did not run"
    finally:
        models._load_and_cache_cli_sessions = original

    # The warm landed on the identity the first lookup resolves.
    assert models.lookup_cli_session_row("alpha-only")["session_id"] == "alpha-only"

    _switch(state, "bravo")
    assert models.lookup_cli_session_row("alpha-only") == {}
    assert models.lookup_cli_session_row("bravo-only")["session_id"] == "bravo-only"


# ── 7. An incomplete aggregate must never be published ────────────────────


def test_warm_does_not_publish_an_incomplete_aggregate(profiles, monkeypatch):
    """If one profile is unavailable, the warm must serve but not cache it.

    ``_load_and_cache_cli_sessions`` treats a plain list as complete, so the
    warm's aggregate has to carry the producer's completeness accounting —
    otherwise a profile that was merely unreadable is published as a profile
    that owns no sessions, and every hint built from it is a lie.
    """
    state, homes = profiles
    _make_state_db(homes["alpha"] / "state.db", [("alpha-only", "cli", "Alpha only")])
    _make_state_db(homes["bravo"] / "state.db", [("bravo-only", "cli", "Bravo only")])

    real_loader = models._load_cli_sessions_uncached

    def _flaky(home, db_path, profile, **kwargs):
        if str(profile) == "bravo":
            raise sqlite3.OperationalError("database is locked")
        return real_loader(home, db_path, profile, **kwargs)

    monkeypatch.setattr(models, "_load_cli_sessions_uncached", _flaky)

    key, load = models._cli_sessions_lookup_identity(all_profiles=True)
    result = load()
    assert isinstance(result, models._CliSessionsLoadResult)
    assert result.complete is False, "one profile was unreadable"

    models.clear_cli_sessions_cache()
    models._load_and_cache_cli_sessions(
        cache_key=key,
        ttl=models._cli_sessions_cache_ttl_seconds(),
        invalidation_stamp=models._cli_sessions_cache_invalidation_stamp(),
        load_sessions=load,
        stale_sessions=None,
        stale_stamp=None,
        all_profiles=True,
        db_path="all profiles",
    )
    with models._CLI_SESSIONS_CACHE_LOCK:
        assert key not in models._CLI_SESSIONS_CACHE, (
            "an incomplete aggregate was published as authoritative"
        )


# ── 8. Enumeration-equivalence edges, pinned ─────────────────────────────


def test_recency_eviction_is_not_undone_by_the_confirmation_read(profiles):
    """A session evicted from the catalog window must stay absent.

    ``get_cli_session_by_id`` deliberately resolves explicit opens *outside*
    any recency window (#8265), so a source-scoped confirmation is not a
    catalog-membership check. The fast path must therefore prove membership
    itself: when the snapshot no longer describes the current catalog it has
    to defer to the enumeration, which is the only thing that knows the
    window.

    The lookup happens BEFORE any re-enumeration on purpose: enumerating first
    would replace the snapshot and hide the stale-hint schedule entirely.
    """
    state, homes = profiles
    db = homes["alpha"] / "state.db"
    _make_state_db(db, [("old-one", "cli", "Old one")])

    models.get_cli_sessions()  # prime while it is still in the window
    # Positive control: while it is in the window the fast path answers.
    assert models.lookup_cli_session_row("old-one")["session_id"] == "old-one"

    # Many newer sessions push it out of the bounded window; the row itself
    # still exists in state.db.
    conn = sqlite3.connect(str(db))
    try:
        for i in range(40):
            started = 1781024055.0 + 100 + i
            conn.execute(
                "INSERT OR REPLACE INTO sessions "
                "(id, source, model, message_count, started_at, title, cwd) "
                "VALUES (?, 'cli', 'test-model', 2, ?, ?, '/tmp')",
                (f"new-{i:02d}", started, f"New {i}"),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) "
                "VALUES (?, 'user', 'hi', ?)",
                (f"new-{i:02d}", started),
            )
        conn.commit()
    finally:
        conn.close()

    # Look up through the consumer BEFORE anything re-enumerates.
    assert routes._lookup_cli_session_metadata("old-one") == {}

    # And the enumeration it deferred to agrees.
    assert "old-one" not in {r["session_id"] for r in models.get_cli_sessions()}


def test_streaming_transition_re_enumerates_instead_of_reusing_the_idle_hint(
    profiles, monkeypatch
):
    """A marker transition must not reuse the idle snapshot.

    The marker is part of the volatile identity, so the snapshot taken while
    idle does not describe the frozen catalog — and a frozen snapshot does not
    describe the idle one. Either way the lookup must re-enumerate rather than
    answer from the previous mode's rows. (Inside a single frozen episode the
    fresh primary may answer; that is pinned separately.)
    """
    state, homes = profiles
    _make_state_db(homes["alpha"] / "state.db", [("mid-turn", "cli", "Mid turn")])

    models.get_cli_sessions()  # prime while idle
    monkeypatch.setattr(
        models, "_cli_sessions_streaming_freeze_marker", lambda: ("streaming", ("turn-1",))
    )

    loads = []
    real_load = models._load_cli_sessions_uncached

    def _counting(*args, **kwargs):
        loads.append(kwargs.get("session_ids"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(models, "_load_cli_sessions_uncached", _counting)

    assert models.lookup_cli_session_row("mid-turn")["session_id"] == "mid-turn"
    assert None in loads, f"the idle snapshot was reused across the transition: {loads}"

    # Content still comes from the producer, not from any snapshot.
    conn = sqlite3.connect(str(homes["alpha"] / "state.db"))
    try:
        conn.execute("UPDATE sessions SET title = 'Renamed mid-turn' WHERE id = 'mid-turn'")
        conn.commit()
    finally:
        conn.close()
    assert models.lookup_cli_session_row("mid-turn")["title"] == "Renamed mid-turn"


# ── 9. The optimisation itself, proven on a real store ───────────────────


def test_current_snapshot_answers_without_reprojecting(profiles, monkeypatch):
    """A lookup against an unchanged catalog must not re-run the projection.

    This is the whole point of the change, asserted against a real state.db:
    the snapshot describes the current revision, so membership is proven and
    the answer comes from the snapshot plus one bounded read.
    """
    state, homes = profiles
    _make_state_db(homes["alpha"] / "state.db", [("live-one", "cli", "Live one")])

    models.get_cli_sessions()  # prime
    loads = []
    real_load = models._load_cli_sessions_uncached

    def _counting(*args, **kwargs):
        loads.append(kwargs.get("session_ids"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(models, "_load_cli_sessions_uncached", _counting)

    row = models.lookup_cli_session_row("live-one")
    assert row["session_id"] == "live-one"
    # Exactly one bounded single-SID read; no full projection.
    assert loads == [("live-one",)], loads


def test_expired_but_unchanged_snapshot_still_answers(profiles, monkeypatch):
    """TTL expiry alone must not force a re-projection.

    Expiry changes freshness, not membership: the revision is unchanged, so the
    snapshot still describes the current catalog and the bounded read confirms
    the content.
    """
    state, homes = profiles
    _make_state_db(homes["alpha"] / "state.db", [("ttl-one", "cli", "TTL one")])

    models.get_cli_sessions()  # prime
    # Force every primary entry to be long-expired without touching its stamp.
    with models._CLI_SESSIONS_CACHE_LOCK:
        for key in list(models._CLI_SESSIONS_CACHE):
            entry = models._CLI_SESSIONS_CACHE[key]
            models._CLI_SESSIONS_CACHE[key] = (0.0, entry[1], entry[2])

    loads = []
    real_load = models._load_cli_sessions_uncached

    def _counting(*args, **kwargs):
        loads.append(kwargs.get("session_ids"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(models, "_load_cli_sessions_uncached", _counting)

    assert models.lookup_cli_session_row("ttl-one")["session_id"] == "ttl-one"
    assert loads == [("ttl-one",)], loads


def test_changed_revision_falls_back_to_the_enumeration(profiles, monkeypatch):
    """Once the catalog revision moves, the snapshot may not answer.

    Membership cannot be proven against a different revision — the row may have
    left the window or the store — so the lookup must defer to the enumeration
    even though the hint still carries the row.
    """
    state, homes = profiles
    db = homes["alpha"] / "state.db"
    _make_state_db(db, [("moved-one", "cli", "Moved one")])

    models.get_cli_sessions()  # prime at revision R1

    # An unrelated write advances the revision without touching moved-one.
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) "
            "VALUES ('moved-one', 'user', 'another', 1781099999.0)"
        )
        conn.commit()
    finally:
        conn.close()

    loads = []
    real_load = models._load_cli_sessions_uncached

    def _counting(*args, **kwargs):
        loads.append(kwargs.get("session_ids"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(models, "_load_cli_sessions_uncached", _counting)

    # The row is still there and still current — but membership is now proved
    # by the enumeration, not by the stale snapshot.
    assert models.lookup_cli_session_row("moved-one")["session_id"] == "moved-one"
    assert loads and loads[-1] is None, (
        f"expected a full enumeration, got {loads}"
    )


# ── 10. Inputs to the projection that are not state.db ───────────────────


def test_project_catalog_change_invalidates_the_hint(profiles, monkeypatch):
    """A projects.json change must invalidate the snapshot.

    The projection reads the project catalog to resolve each row's project_id
    and to drive the assigned-recovery pass, so dropping a project can move a
    session out of the window WITHOUT touching state.db. If the catalog were
    absent from the cache identity, a hint built before the change would still
    look current.
    """
    state, homes = profiles
    _make_state_db(homes["alpha"] / "state.db", [("assigned-one", "cli", "Assigned")])

    models.get_cli_sessions()  # prime with the empty catalog
    assert models.lookup_cli_session_row("assigned-one")["session_id"] == "assigned-one"

    # An external project-catalog change; state.db and the index are untouched.
    models.PROJECTS_FILE.write_text(
        '[{"id": "p1", "name": "P1", "profile": "alpha"}]', encoding="utf-8"
    )

    loads = []
    real_load = models._load_cli_sessions_uncached

    def _counting(*args, **kwargs):
        loads.append(kwargs.get("session_ids"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(models, "_load_cli_sessions_uncached", _counting)

    assert models.lookup_cli_session_row("assigned-one")["session_id"] == "assigned-one"
    assert loads and loads[-1] is None, (
        f"a project-catalog change must re-enumerate, got {loads}"
    )


def test_expired_streaming_hint_cannot_outlive_the_producer_ttl(profiles, monkeypatch):
    """Under a frozen key, an expired snapshot must defer to the enumeration.

    The streaming freeze hides state.db writes, so the key can stay the same
    while 40 newer sessions appear. Within the producer's freshness window
    that is exactly what it serves; past it the producer re-projects, and so
    must the lookup.
    """
    state, homes = profiles
    db = homes["alpha"] / "state.db"
    _make_state_db(db, [("old-one", "cli", "Old one")])

    marker = ("streaming", ("turn-1",))
    monkeypatch.setattr(models, "_cli_sessions_streaming_freeze_marker", lambda: marker)
    models.get_cli_sessions()  # prime under the frozen key

    # 40 newer sessions arrive; the frozen key does not move.
    conn = sqlite3.connect(str(db))
    try:
        for i in range(40):
            started = 1781024055.0 + 100 + i
            conn.execute(
                "INSERT OR REPLACE INTO sessions "
                "(id, source, model, message_count, started_at, title, cwd) "
                "VALUES (?, 'cli', 'test-model', 2, ?, ?, '/tmp')",
                (f"new-{i:02d}", started, f"New {i}"),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) "
                "VALUES (?, 'user', 'hi', ?)",
                (f"new-{i:02d}", started),
            )
        conn.commit()
    finally:
        conn.close()

    # Expire the frozen entry without changing its key.
    with models._CLI_SESSIONS_CACHE_LOCK:
        for key in list(models._CLI_SESSIONS_CACHE):
            entry = models._CLI_SESSIONS_CACHE[key]
            models._CLI_SESSIONS_CACHE[key] = (0.0, entry[1], entry[2])

    loads = []
    real_load = models._load_cli_sessions_uncached

    def _counting(*args, **kwargs):
        loads.append(kwargs.get("session_ids"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(models, "_load_cli_sessions_uncached", _counting)

    assert routes._lookup_cli_session_metadata("old-one") == {}
    # A full enumeration ran (the trailing ('old-one',) read is master's
    # pre-existing oneshot fallback, not the hint path).
    assert None in loads, f"expected a full enumeration, got {loads}"


# ── 11. The all-profiles identity is its own shape ────────────────────────


def test_all_profiles_stale_lkg_defers_after_an_external_write(profiles):
    """A write to a NON-active profile must invalidate the aggregate hint.

    The aggregate key's volatile component is the per-profile context tuple
    (index 3), not the stable profile-home list (index 4). Those two are easy
    to confuse, and confusing them makes every expired aggregate snapshot look
    current — an external write would then resurrect rows forever. This pins
    the correct index by exercising exactly that write.
    """
    state, homes = profiles
    db_alpha = homes["alpha"] / "state.db"
    db_bravo = homes["bravo"] / "state.db"
    _make_state_db(db_alpha, [("old-one", "cli", "Old one")])
    _make_state_db(db_bravo, [("bravo-one", "cli", "Bravo one")])

    primed = {r["session_id"] for r in models.get_cli_sessions(all_profiles=True)}
    assert {"old-one", "bravo-one"} <= primed

    # An external writer pushes newer sessions into the INACTIVE profile only.
    conn = sqlite3.connect(str(db_bravo))
    try:
        for i in range(40):
            started = 1781024055.0 + 100 + i
            conn.execute(
                "INSERT OR REPLACE INTO sessions "
                "(id, source, model, message_count, started_at, title, cwd) "
                "VALUES (?, 'cli', 'test-model', 2, ?, ?, '/tmp')",
                (f"new-{i:02d}", started, f"New {i}"),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) "
                "VALUES (?, 'user', 'hi', ?)",
                (f"new-{i:02d}", started),
            )
        conn.commit()
    finally:
        conn.close()

    loads = []
    real_load = models._load_cli_sessions_uncached

    def _counting(*args, **kwargs):
        loads.append(kwargs.get("session_ids"))
        return real_load(*args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(models, "_load_cli_sessions_uncached", _counting)
        # old-one is still in the window, so it answers — but it must be the
        # ENUMERATION that decided, not the pre-write snapshot.
        assert models.lookup_cli_session_row("old-one", all_profiles=True)["session_id"] == (
            "old-one"
        )
        assert None in loads, f"the aggregate snapshot survived an external write: {loads}"
    finally:
        monkeypatch.undo()


def test_aggregate_does_not_substitute_an_earlier_profiles_out_of_window_row(profiles):
    """The confirmation read must not swap the aggregate's owner.

    Same SID in both profiles, but alpha's copy has been pushed out of alpha's
    own window by newer sessions while bravo's copy is still in bravo's. The
    aggregate enumeration therefore answers with bravo's row. The bounded
    reader resolves per profile in aggregate order and would hand back alpha's
    out-of-window row — which is a different owner, not a confirmation.
    """
    state, homes = profiles
    db_alpha = homes["alpha"] / "state.db"
    db_bravo = homes["bravo"] / "state.db"
    _make_state_db(db_alpha, [("shared", "cli", "Alpha copy")])
    _make_state_db(db_bravo, [("shared", "cli", "Bravo copy")])

    # Bury alpha's copy outside alpha's window.
    conn = sqlite3.connect(str(db_alpha))
    try:
        for i in range(40):
            started = 1781024055.0 + 100 + i
            conn.execute(
                "INSERT OR REPLACE INTO sessions "
                "(id, source, model, message_count, started_at, title, cwd) "
                "VALUES (?, 'cli', 'test-model', 2, ?, ?, '/tmp')",
                (f"alpha-new-{i:02d}", started, f"Alpha {i}"),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) "
                "VALUES (?, 'user', 'hi', ?)",
                (f"alpha-new-{i:02d}", started),
            )
        conn.commit()
    finally:
        conn.close()

    enumerated = models.get_cli_sessions(all_profiles=True)
    owner = next(r for r in enumerated if r["session_id"] == "shared")
    assert owner["profile"] == "bravo", "bravo should own the aggregate row"

    got = models.lookup_cli_session_row("shared", all_profiles=True)
    assert got["profile"] == owner["profile"]
    assert got["title"] == owner["title"]
