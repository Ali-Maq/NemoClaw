# SPDX-License-Identifier: Apache-2.0
"""Independent, real-SQLite contract checks for HERMES_USAGE_SCRIPT.

Run with --script extracted.py or --compiled-module compiled-module.js.
This evaluator does not modify the candidate or inspect a live Hermes database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest


MAX_SAFE = 2**53 - 1
COUNTERS = {
    "api_calls": "api_call_count",
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "tool_calls": "tool_call_count",
}
SECRET = "FIXTURE_ONLY_MESSAGE_CONTENT_MUST_NOT_BE_READ_7a2c"
SCRIPT = ""
PYTHON = sys.executable

# Audit the real SQLite API. No result or database operation is mocked.
# The original connect function opens the actual temporary database.
AUDIT_WRAPPER = r'''
import json,os,sqlite3,sys,urllib.parse
original_connect=sqlite3.connect
audit={"connections":[],"message_reads":[],"writes":[],"query_only":[]}
def connect(*args,**kwargs):
    uri=str(args[0] if args else kwargs.get("database",""))
    audit["connections"].append({"uri":kwargs.get("uri",args[7] if len(args)>7 else False),"mode":urllib.parse.parse_qs(urllib.parse.urlsplit(uri).query).get("mode",[])})
    connection=original_connect(*args,**kwargs)
    def authorize(action,one,two,database,trigger):
        if action==sqlite3.SQLITE_READ and one=="messages":
            audit["message_reads"].append({"table":one,"column":two})
            return sqlite3.SQLITE_DENY
        if action in {sqlite3.SQLITE_INSERT,sqlite3.SQLITE_UPDATE,sqlite3.SQLITE_DELETE}:
            audit["writes"].append({"action":action,"table":one})
            return sqlite3.SQLITE_DENY
        if action==sqlite3.SQLITE_PRAGMA and str(one).lower()=="query_only" and two is not None:
            audit["query_only"].append(str(two).lower())
        return sqlite3.SQLITE_OK
    connection.set_authorizer(authorize)
    return connection
sqlite3.connect=connect
try:
    with open(os.environ["AYA_TEST_SCRIPT"],encoding="utf-8") as file:
        source=file.read()
    exec(compile(source,"<candidate>","exec"),{"__name__":"__main__"})
finally:
    with open(os.environ["AYA_TEST_AUDIT"],"w",encoding="utf-8") as file:
        json.dump(audit,file)
'''


def row(identity: str, parent: str | None = None, model: str | None = "m", provider: str | None = "p", **values: object) -> dict[str, object]:
    return {
        "id": identity,
        "parent_session_id": parent,
        "model": model,
        "billing_provider": provider,
        "api_call_count": 1,
        "input_tokens": 10,
        "output_tokens": 3,
        "tool_call_count": 2,
        **values,
    }


class UsageContract(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="aya-usage-contract-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "state.db"
        self.source = self.root / "candidate.py"
        self.source.write_text(SCRIPT)

    def database(self, rows: list[dict[str, object]], omit: set[str] | None = None, messages: bool = False) -> None:
        omitted = omit or set()
        columns = {
            "id": "TEXT PRIMARY KEY",
            "parent_session_id": "TEXT",
            "model": "TEXT",
            "billing_provider": "TEXT",
            **{name: "INTEGER" for name in COUNTERS.values()},
        }
        columns = {name: kind for name, kind in columns.items() if name not in omitted}
        with sqlite3.connect(self.db) as connection:
            connection.execute("CREATE TABLE sessions (" + ", ".join(name + " " + kind for name, kind in columns.items()) + ")")
            for value in rows:
                keys = [name for name in columns if name in value]
                connection.execute("INSERT INTO sessions (" + ",".join(keys) + ") VALUES (" + ",".join("?" for _ in keys) + ")", [value[name] for name in keys])
            if messages:
                connection.execute("CREATE TABLE messages (session_id TEXT, content TEXT, prompt TEXT)")
                connection.execute("INSERT INTO messages VALUES (?,?,?)", ("root", SECRET, SECRET))

    def invoke(self, session: str = "", children: bool = False, audited: bool = False) -> subprocess.CompletedProcess[str]:
        before = (hashlib.sha256(self.db.read_bytes()).hexdigest(), self.db.stat().st_mtime_ns) if self.db.exists() else None
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if audited:
            environment["AYA_TEST_SCRIPT"] = str(self.source)
            environment["AYA_TEST_AUDIT"] = str(self.root / "audit.json")
        result = subprocess.run(
            [PYTHON, "-c", AUDIT_WRAPPER if audited else SCRIPT, str(self.db), session, "1" if children else "0"],
            capture_output=True, text=True, timeout=8, env=environment,
        )
        self.assertLessEqual(len(result.stdout.encode()) + len(result.stderr.encode()), 1024 * 1024)
        self.assertNotIn(SECRET, result.stdout + result.stderr, "Message content was exposed")
        self.assertNotIn(str(self.db), result.stdout + result.stderr, "Database path was exposed")
        if before:
            self.assertEqual(before, (hashlib.sha256(self.db.read_bytes()).hexdigest(), self.db.stat().st_mtime_ns), "Database was modified")
        else:
            self.assertFalse(self.db.exists(), "Missing database was created")
        return result

    def report(self, session: str = "", children: bool = False, audited: bool = False) -> dict[str, object]:
        result = self.invoke(session, children, audited)
        self.assertEqual(result.returncode, 0, "Candidate failed a valid fixture: " + result.stderr[:500])
        data = json.loads(result.stdout, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        self.assertEqual(set(data), {"schema_version", "session_count", "totals", "models", "incomplete_fields", "cost_usd"})
        self.assertEqual(data["schema_version"], 1)
        self.assertIs(type(data["session_count"]), int)
        self.assertGreaterEqual(data["session_count"], 0)
        self.assertIsNone(data["cost_usd"])
        self.assertEqual(set(data["totals"]), set(COUNTERS))
        self.assertIsInstance(data["models"], list)
        self.assertEqual(data["models"], sorted(data["models"], key=lambda group: (group["model"], group["provider"])))
        self.assertEqual(len(data["incomplete_fields"]), len(set(data["incomplete_fields"])))
        self.assertLessEqual(set(data["incomplete_fields"]), set(COUNTERS))
        for group in data["models"]:
            self.assertEqual(set(group), {"model", "provider", "session_count", *COUNTERS})
            self.assertIs(type(group["session_count"]), int)
        for counters in [data["totals"], *data["models"]]:
            for name in COUNTERS:
                value = counters[name]
                if value is not None:
                    self.assertIs(type(value), int)
                    self.assertGreaterEqual(value, 0)
                    self.assertLessEqual(value, MAX_SAFE)
        return data

    def rejected(self, session: str = "", children: bool = False) -> None:
        result = self.invoke(session, children)
        self.assertNotEqual(result.returncode, 0, "Invalid input was accepted")
        self.assertTrue((result.stdout + result.stderr).strip(), "Failure needs an explicit error")
        self.assertLessEqual(len(result.stdout + result.stderr), 2048, "Unbounded failure output")
        self.assertNotIn("Traceback (most recent call last)", result.stdout + result.stderr, "Internal traceback exposed")

    def test_empty_database(self) -> None:
        self.database([])
        actual = self.report()
        self.assertEqual(actual["session_count"], 0)
        self.assertEqual(actual["totals"], dict.fromkeys(COUNTERS, 0))
        self.assertEqual(actual["models"], [])
        self.assertEqual(actual["incomplete_fields"], [])

    def test_empty_store_with_missing_counters_still_has_zero_counts(self) -> None:
        self.database([], omit=set(COUNTERS.values()))
        actual = self.report()
        self.assertEqual(actual["session_count"], 0)
        self.assertEqual(actual["totals"], dict.fromkeys(COUNTERS, 0))
        self.assertEqual(actual["incomplete_fields"], [])

    def test_known_aggregates_and_sorted_model_provider_groups(self) -> None:
        self.database([row("one", model="z", provider="b"), row("two", model="a", provider="b", input_tokens=20), row("three", model="a", provider="a", output_tokens=7), row("four", model="a", provider="b", api_call_count=4)])
        actual = self.report()
        self.assertEqual(actual["session_count"], 4)
        self.assertEqual(actual["totals"], {"api_calls": 7, "input_tokens": 50, "output_tokens": 16, "tool_calls": 8})
        self.assertEqual(actual["models"], [
            {"model": "a", "provider": "a", "session_count": 1, "api_calls": 1, "input_tokens": 10, "output_tokens": 7, "tool_calls": 2},
            {"model": "a", "provider": "b", "session_count": 2, "api_calls": 5, "input_tokens": 30, "output_tokens": 6, "tool_calls": 4},
            {"model": "z", "provider": "b", "session_count": 1, "api_calls": 1, "input_tokens": 10, "output_tokens": 3, "tool_calls": 2},
        ])

    def tree(self) -> None:
        self.database([row("root"), row("child", "root"), row("grandchild", "child"), row("other"), row("orphan", "missing")])

    def test_root_only_excludes_descendants(self) -> None:
        self.tree()
        actual = self.report("root")
        self.assertEqual(actual["session_count"], 1)
        self.assertEqual(actual["totals"]["input_tokens"], 10)

    def test_descendants_include_all_levels_once(self) -> None:
        self.tree()
        actual = self.report("root", True)
        self.assertEqual(actual["session_count"], 3)
        self.assertEqual(actual["totals"]["input_tokens"], 30)

    def test_default_includes_roots_children_and_orphans(self) -> None:
        self.tree()
        self.assertEqual(self.report()["session_count"], 5)

    def test_cycle_terminates_and_counts_once(self) -> None:
        self.database([row("a", "b"), row("b", "a"), row("c", "b"), row("outside")])
        actual = self.report("a", True)
        self.assertEqual(actual["session_count"], 3)
        self.assertEqual(actual["totals"]["api_calls"], 3)

    def test_self_cycle_counts_once(self) -> None:
        self.database([row("self", "self")])
        self.assertEqual(self.report("self", True)["session_count"], 1)

    def test_unknown_session_fails(self) -> None:
        self.database([row("known")])
        self.rejected("absent")

    def test_unknown_session_with_children_fails(self) -> None:
        self.database([row("known")])
        self.rejected("absent", True)

    def test_children_without_session_fails(self) -> None:
        self.database([row("known")])
        self.rejected("", True)

    def test_sql_injection_id_is_exact_data(self) -> None:
        identity = "x' OR 1=1; DROP TABLE messages; --"
        self.database([row(identity), row("other", input_tokens=999)], messages=True)
        self.assertEqual(self.report(identity)["totals"]["input_tokens"], 10)

    def test_missing_injection_id_does_not_select_all(self) -> None:
        self.database([row("known")])
        self.rejected("' OR 1=1 --")

    def test_descendant_filter_uses_bound_session_id(self) -> None:
        identity = "x' UNION SELECT id FROM sessions --"
        self.database([row(identity), row("child", identity), row("outside")])
        self.assertEqual(self.report(identity, True)["session_count"], 2)

    def test_null_counter_affects_only_relevant_group(self) -> None:
        self.database([row("one", model="known"), row("two", model="unknown", input_tokens=None)])
        actual = self.report()
        self.assertIsNone(actual["totals"]["input_tokens"])
        self.assertEqual(actual["models"][0]["input_tokens"], 10)
        self.assertIsNone(actual["models"][1]["input_tokens"])
        self.assertEqual(actual["incomplete_fields"], ["input_tokens"])
        self.assertEqual(actual["totals"]["api_calls"], 2)

    def test_all_counter_columns_missing_remain_unknown(self) -> None:
        self.database([row("one")], omit=set(COUNTERS.values()))
        actual = self.report()
        self.assertEqual(actual["totals"], dict.fromkeys(COUNTERS))
        self.assertEqual(set(actual["incomplete_fields"]), set(COUNTERS))

    def test_missing_model_and_provider_columns(self) -> None:
        self.database([row("one")], omit={"model", "billing_provider"})
        group = self.report()["models"][0]
        self.assertEqual((group["model"], group["provider"]), ("unknown", "unknown"))

    def test_null_model_and_provider(self) -> None:
        self.database([row("one", model=None, provider=None)])
        group = self.report()["models"][0]
        self.assertEqual((group["model"], group["provider"]), ("unknown", "unknown"))

    def test_zero_is_known_not_null(self) -> None:
        self.database([row("zero", **dict.fromkeys(COUNTERS.values(), 0))])
        actual = self.report()
        self.assertEqual(actual["totals"], dict.fromkeys(COUNTERS, 0))
        self.assertEqual(actual["incomplete_fields"], [])

    def test_missing_database_is_not_created(self) -> None:
        self.rejected()

    def test_missing_sessions_table_fails(self) -> None:
        with sqlite3.connect(self.db) as connection:
            connection.execute("CREATE TABLE unrelated (value INTEGER)")
        self.rejected()

    def test_missing_session_id_column_fails(self) -> None:
        self.database([], omit={"id"})
        self.rejected()

    def test_ten_thousand_sessions_are_not_truncated(self) -> None:
        self.database([row(str(index)) for index in range(10000)])
        actual = self.report()
        self.assertEqual(actual["session_count"], 10000)
        self.assertEqual(actual["totals"]["input_tokens"], 100000)

    def test_over_ten_thousand_sessions_fails_explicitly(self) -> None:
        self.database([row(str(index)) for index in range(10001)])
        self.rejected()

    def test_over_limit_descendants_fail(self) -> None:
        self.database([row("root"), *[row(str(index), "root") for index in range(10000)]])
        self.rejected("root", True)

    def test_metadata_only_read_only_connection(self) -> None:
        self.database([row("root")], messages=True)
        self.assertEqual(self.report(audited=True)["session_count"], 1)
        audit = json.loads((self.root / "audit.json").read_text())
        self.assertTrue(audit["connections"])
        self.assertTrue(all(item["uri"] is True and item["mode"] == ["ro"] for item in audit["connections"]))
        self.assertTrue(audit["query_only"], "PRAGMA query_only was not set")
        self.assertIn(audit["query_only"][-1], {"1", "on", "true"})
        self.assertEqual(audit["message_reads"], [])
        self.assertEqual(audit["writes"], [])


def counter_test(output_name: str, kind: str):
    column = COUNTERS[output_name]

    def check(self: UsageContract) -> None:
        if kind == "missing":
            self.database([row("one")], omit={column})
            actual = self.report()
            self.assertIsNone(actual["totals"][output_name])
            self.assertIsNone(actual["models"][0][output_name])
            self.assertEqual(actual["incomplete_fields"], [output_name])
        elif kind == "boundary":
            self.database([row("one", **{column: MAX_SAFE})])
            self.assertEqual(self.report()["totals"][output_name], MAX_SAFE)
        elif kind == "null":
            self.database([row("one", **{column: None})])
            actual = self.report()
            self.assertIsNone(actual["totals"][output_name])
            self.assertIsNone(actual["models"][0][output_name])
            self.assertEqual(actual["incomplete_fields"], [output_name])
        elif kind in {"group_overflow", "total_overflow"}:
            self.database([row("one", **{column: MAX_SAFE}), row("two", model="different" if kind == "total_overflow" else "m", **{column: 1})])
            self.rejected()
        else:
            value = {"negative": -1, "fractional": 1.5, "unsafe": MAX_SAFE + 1, "text": "not-a-counter"}[kind]
            self.database([row("one", **{column: value})])
            self.rejected()

    check.__name__ = "test_" + output_name + "_" + kind
    return check


for counter in COUNTERS:
    for variant in ["missing", "null", "boundary", "negative", "fractional", "unsafe", "text", "group_overflow", "total_overflow"]:
        setattr(UsageContract, "test_" + counter + "_" + variant, counter_test(counter, variant))


def main() -> int:
    global SCRIPT, PYTHON
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--script", type=Path, help="Extracted HERMES_USAGE_SCRIPT Python source")
    source.add_argument("--compiled-module", type=Path, help="tsc-compiled JS module exporting HERMES_USAGE_SCRIPT")
    parser.add_argument("--node", default="node")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--list", action="store_true", help="List contract cases without executing candidate code")
    args = parser.parse_args()
    PYTHON = args.python
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(UsageContract)
    if args.list:
        print(json.dumps({"count": suite.countTestCases(), "tests": [case.id().split(".")[-1] for case in suite]}, indent=2))
        return 0
    if args.script:
        SCRIPT = args.script.read_text()
    elif args.compiled_module:
        extractor = "const m=await import((await import('node:url')).pathToFileURL(process.argv[1]).href);const s=m.HERMES_USAGE_SCRIPT??m.default?.HERMES_USAGE_SCRIPT;if(typeof s!=='string')throw new Error('Missing HERMES_USAGE_SCRIPT');process.stdout.write(JSON.stringify(s));"
        result = subprocess.run([args.node, "--input-type=module", "-e", extractor, str(args.compiled_module.resolve())], capture_output=True, text=True, timeout=20, check=True)
        SCRIPT = json.loads(result.stdout)
    else:
        parser.error("Provide --script or --compiled-module")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "schema_version": 1,
        "candidate_script_sha256": hashlib.sha256(SCRIPT.encode()).hexdigest(),
        "tests_run": result.testsRun,
        "passed": result.testsRun - len(result.failures) - len(result.errors),
        "failures": [case.id().split(".")[-1] for case, _ in result.failures],
        "errors": [case.id().split(".")[-1] for case, _ in result.errors],
        "real_temporary_sqlite": True,
        "live_session_content_accessed": False,
        "scope": "Exported Python script contract; host CLI routing, executor timeout and output capture need separate checks",
    }
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
