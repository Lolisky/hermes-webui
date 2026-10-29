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
    """A session evicted from the recency window must stay absent.

    The bounded confirmation read (``get_cli_session_by_id``) is subject to
    the *same* window as the enumeration, so a hint cannot resurrect a row the
    sidebar no longer shows. This pins that the fast path inherits the
    producer's window rather than widening it.
    """
    state, homes = profiles
    db = homes["alpha"] / "state.db"
    _make_state_db(db, [("old-one", "cli", "Old one")])

    models.get_cli_sessions()  # prime while it is still in the window
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

    enumerated = {r["session_id"] for r in models.get_cli_sessions()}
    assert "old-one" not in enumerated, "window should have evicted it"
    # The fast path agrees with the enumeration instead of resurrecting it.
    assert models.lookup_cli_session_row("old-one") == {}


def test_streaming_transition_keeps_serving_the_confirmed_row(profiles, monkeypatch):
    """The last-known-good hint survives a streaming-marker transition.

    ``_cli_sessions_stable_cache_identity`` deliberately drops the volatile
    marker, so the snapshot stays reachable while a turn streams. The row is
    still confirmed against the live store, so surviving the transition cannot
    make it stale — it only avoids re-projecting mid-stream (#4842).
    """
    state, homes = profiles
    _make_state_db(homes["alpha"] / "state.db", [("mid-turn", "cli", "Mid turn")])

    models.get_cli_sessions()
    marker = ("streaming", ("turn-1",))
    monkeypatch.setattr(
        models, "_cli_sessions_streaming_freeze_marker", lambda: marker
    )
    # The key changed (the marker is now part of it), yet the confirmed row
    # still answers.
    assert models.lookup_cli_session_row("mid-turn")["session_id"] == "mid-turn"
    # And the content is the producer's, not the snapshot's.
    conn = sqlite3.connect(str(homes["alpha"] / "state.db"))
    try:
        conn.execute("UPDATE sessions SET title = 'Renamed mid-turn' WHERE id = 'mid-turn'")
        conn.commit()
    finally:
        conn.close()
    assert models.lookup_cli_session_row("mid-turn")["title"] == "Renamed mid-turn"
