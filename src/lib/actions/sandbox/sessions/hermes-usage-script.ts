// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Read-only SQLite introspection of the Hermes sessions table, used by the
 * `sandbox sessions usage` command. The script is authoritative for reading the
 * sessions database so that the host never has to parse raw SQLite output and
 * never needs to read messages, prompts, paths, credentials or response text.
 *
 * The database is opened with `mode=ro` and `PRAGMA query_only=1` so even an
 * accidental write is impossible. The only columns ever read are the session
 * identity columns (`id`, `parent_session_id`), the optional grouping metadata
 * (`model`, `billing_provider`) and the usage counters (`api_call_count`,
 * `tool_call_count`, `input_tokens`, `output_tokens`).
 */

export const HERMES_USAGE_SCRIPT: string = String.raw`
import json
import os
import sqlite3
import sys
from pathlib import Path

# Python-side safe-integer bound mirrors JavaScript's Number.MAX_SAFE_INTEGER.
JS_MAX_SAFE_INTEGER = 9007199254740991

# Maximum sessions the query may select. Exceeding this is an explicit error
# rather than silent truncation.
SESSION_LIMIT = 10000

# Counter columns grouped in the JSON output. api_call_count -> api_calls,
# tool_call_count -> tool_calls. input_tokens/output_tokens keep their names.
COUNTER_COLUMNS = {
    "api_calls": "api_call_count",
    "tool_calls": "tool_call_count",
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
}

# Order of counters in output objects (stable JSON field ordering).
COUNTER_NAMES = ["api_calls", "tool_calls", "input_tokens", "output_tokens"]


class UsageError(Exception):
    pass


def _coerce_counter(value):
    """Validate a counter value read from SQLite.

    Returns a non-negative Python int within JS safe-integer range, or None when
    the value is unknown (NULL). Any non-NULL corrupt value (negative,
    fractional, non-numeric TEXT, boolean, or overflowing) raises UsageError
    rather than being silently clamped or dropped to an unknown.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UsageError("invalid counter value")
    if isinstance(value, float) and not value.is_integer():
        raise UsageError("invalid counter value")
    val = int(value)
    if val < 0:
        raise UsageError("invalid counter value")
    if val > JS_MAX_SAFE_INTEGER:
        raise UsageError("counter value out of safe integer range")
    return val


def _coerce_text(value):
    """Coerce a grouping field to a non-empty string, defaulting to 'unknown'."""
    if value is None:
        return "unknown"
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else "unknown"
    return "unknown"


def _column_set(cursor, table):
    cursor.execute("PRAGMA table_info(%s)" % table)
    return {row[1] for row in cursor.fetchall()}


def _collect_descendants(cursor, root_id):
    """BFS over parent_session_id. Returns every descendant exactly once and
    never hangs: a seen-set guarantees termination even on cycles. Bounded so an
    unbounded fan-out cannot consume memory or race."""
    from collections import deque
    seen = set()
    frontier = deque([root_id])
    result = []
    while frontier:
        current = frontier.popleft()
        if current in seen:
            continue
        seen.add(current)
        result.append(current)
        if len(result) > SESSION_LIMIT:
            raise UsageError(
                "selection exceeds %d sessions; narrow the request with --session"
                % SESSION_LIMIT
            )
        cursor.execute(
            "SELECT id FROM sessions WHERE parent_session_id = ? ORDER BY id LIMIT ?",
            (current, SESSION_LIMIT + 1),
        )
        children = [row[0] for row in cursor.fetchall()]
        if len(children) > SESSION_LIMIT:
            raise UsageError(
                "selection exceeds %d sessions; narrow the request with --session"
                % SESSION_LIMIT
            )
        for child_id in children:
            if child_id not in seen:
                frontier.append(child_id)
    return result


def main(argv):
    if len(argv) < 4:
        raise UsageError("expected: db_path requested_session include_children")
    db_path, requested_session, include_children_raw = argv[1], argv[2], argv[3]
    if include_children_raw not in ("0", "1"):
        raise UsageError("include_children must be 0 or 1")
    include_children = include_children_raw == "1"
    if include_children and not requested_session:
        raise UsageError("--include-children requires --session")

    if not db_path:
        db_path = os.path.join(os.environ.get("HERMES_HOME", "/sandbox/.hermes"), "state.db")
    if not os.path.exists(db_path):
        raise UsageError("database not found")

    # Build a read-only URI even for paths containing '?', '#' or other
    # reserved characters.
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.execute("PRAGMA query_only=1")
        # Serialize the whole read under one transaction so the column probe,
        # descendant traversal and final SELECT observe a consistent snapshot.
        conn.execute("BEGIN")
        cursor = conn.cursor()
        try:
            columns = _column_set(cursor, "sessions")
        except sqlite3.Error:
            raise UsageError("sessions table unavailable")

        available = {}
        missing_columns = []
        for out_name, col in COUNTER_COLUMNS.items():
            if col in columns:
                available[out_name] = col
            else:
                missing_columns.append(col)

        for col in ("id", "parent_session_id"):
            if col not in columns:
                raise UsageError("sessions table is missing a required column")

        group_cols = []
        for col in ("model", "billing_provider"):
            if col in columns:
                group_cols.append(col)

        select_cols = ["id", "parent_session_id"] + group_cols
        for col in available.values():
            if col not in select_cols:
                select_cols.append(col)

        requested_ids = None
        if requested_session:
            cursor.execute(
                "SELECT id FROM sessions WHERE id = ? LIMIT 1",
                (requested_session,),
            )
            if cursor.fetchone() is None:
                raise UsageError("unknown session")
            if include_children:
                requested_ids = _collect_descendants(cursor, requested_session)
            else:
                requested_ids = [requested_session]

        where = ""
        params = []
        if requested_ids is not None:
            placeholders = ",".join("?" for _ in requested_ids)
            where = " WHERE id IN (%s)" % placeholders
            params = list(requested_ids)

        # Bound the SELECT with a LIMIT and detect overflow from the row count,
        # avoiding a separate unbounded COUNT that could race.
        cursor.execute(
            "SELECT %s FROM sessions%s ORDER BY id LIMIT ?"
            % (",".join(select_cols), where),
            params + [SESSION_LIMIT + 1],
        )
        rows = cursor.fetchall()
        if len(rows) > SESSION_LIMIT:
            raise UsageError(
                "selection exceeds %d sessions; narrow the request with --session"
                % SESSION_LIMIT
            )
    finally:
        conn.close()

    row_names = select_cols
    idx = {name: i for i, name in enumerate(row_names)}

    def group_value(row, name):
        if name in idx:
            return _coerce_text(row[idx[name]])
        return "unknown"

    if not rows:
        # Empty selection: zero totals, no incomplete fields, even when the
        # counter columns are absent (there is no data to be unknown).
        totals = {"api_calls": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0}
        result = {
            "schema_version": 1,
            "session_count": 0,
            "totals": totals,
            "models": [],
            "incomplete_fields": [],
            "cost_usd": None,
        }
        sys.stdout.write(json.dumps(result, separators=(",", ":")))
        sys.stdout.write("\n")
        return

    models = {}
    incomplete = set()
    for col in missing_columns:
        for out_name, c in COUNTER_COLUMNS.items():
            if c == col:
                incomplete.add(out_name)

    for row in rows:
        model = group_value(row, "model")
        provider = group_value(row, "billing_provider")
        key = (model, provider)
        entry = models.get(key)
        if entry is None:
            entry = {
                "model": model,
                "provider": provider,
                "session_count": 0,
                "api_calls": 0,
                "tool_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
            models[key] = entry
        entry["session_count"] += 1
        for out_name in COUNTER_NAMES:
            if out_name not in available:
                incomplete.add(out_name)
                entry[out_name] = None
                continue
            value = _coerce_counter(row[idx[available[out_name]]])
            if value is None:
                incomplete.add(out_name)
                entry[out_name] = None
            elif entry[out_name] is not None:
                entry[out_name] += value
                if entry[out_name] > JS_MAX_SAFE_INTEGER:
                    raise UsageError("counter value out of safe integer range")

    totals = {"api_calls": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0}
    for name in COUNTER_NAMES:
        if name in incomplete:
            totals[name] = None
            continue
        total = 0
        for entry in models.values():
            v = entry[name]
            if v is None:
                total = None
                break
            total += v
        if total is not None and total > JS_MAX_SAFE_INTEGER:
            raise UsageError("counter value out of safe integer range")
        totals[name] = total

    model_list = sorted(models.values(), key=lambda m: (m["model"], m["provider"]))
    incomplete_fields = sorted(name for name in COUNTER_NAMES if name in incomplete)

    result = {
        "schema_version": 1,
        "session_count": len(rows),
        "totals": totals,
        "models": model_list,
        "incomplete_fields": incomplete_fields,
        "cost_usd": None,
    }
    sys.stdout.write(json.dumps(result, separators=(",", ":")))
    sys.stdout.write("\n")


if __name__ == "__main__":
    try:
        main(sys.argv)
    except UsageError as exc:
        sys.stderr.write("usage-error: %s\n" % exc)
        sys.exit(2)
    except sqlite3.Error:
        sys.stderr.write("usage-error: failed to read sessions database\n")
        sys.exit(2)
    except Exception:  # noqa: BLE001 - generic guard; do not leak exception text
        sys.stderr.write("usage-error: failed to read sessions database\n")
        sys.exit(2)
`;
