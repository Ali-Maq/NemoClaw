// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from "vitest";
import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { HERMES_USAGE_SCRIPT } from "./hermes-usage-script";
import type {
  OpenShellSandboxBufferedCommandCompletion,
  OpenShellSandboxBufferedCommandExecutor,
} from "../../../adapters/openshell/sandbox-command";
import { createSandboxSessionsUsageAction } from "./usage";

vi.mock("../../../state/registry", () => ({
  getSandbox: (name: string) =>
    ({
      alpha: { agent: "hermes" },
      openclaw: { agent: "openclaw" },
    })[name as "alpha" | "openclaw"],
}));

vi.mock("../gateway-state", () => ({
  getKnownSandboxTargetGatewayName: () => "test-gateway",
}));

/**
 * Focused Vitest coverage for the sessions-usage action and its embedded
 * Python reader. Tests use real temporary SQLite databases created via Python
 * so the read path (mode=ro, PRAGMA query_only, counter coercion, descendant
 * traversal) is exercised end-to-end rather than against a mock.
 */

type Row = readonly [
  id: string,
  parent: string | null,
  model: string | null,
  provider: string | null,
  apiCalls: number | null,
  toolCalls: number | null,
  inputTokens: number | null,
  outputTokens: number | null,
];

interface Totals {
  api_calls: number | null;
  tool_calls: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
}

interface Report {
  schema_version: number;
  session_count: number;
  totals: Totals;
  models: Array<{ model: string; provider: string; session_count: number } & Totals>;
  incomplete_fields: string[];
  cost_usd: null;
}

const tempDirs: string[] = [];

/** Create a real temporary SQLite database conforming to the sessions schema. */
function makeTempDb(rows: readonly Row[], omitColumns: readonly string[] = []): string {
  const dir = mkdtempSync(join(tmpdir(), "nemoclaw-usage-"));
  tempDirs.push(dir);
  const dbPath = join(dir, "state.db");
  const columns: Array<[name: string, type: string]> = [
    ["id", "TEXT PRIMARY KEY"],
    ["parent_session_id", "TEXT"],
    ["model", "TEXT"],
    ["billing_provider", "TEXT"],
    ["api_call_count", "INTEGER"],
    ["tool_call_count", "INTEGER"],
    ["input_tokens", "INTEGER"],
    ["output_tokens", "INTEGER"],
  ].filter((col): col is [name: string, type: string] => !omitColumns.includes(col[0]));

  const columnNames = columns.map(([name]) => name);
  const columnDefs = columns.map(([name, type]) => `${name} ${type}`);

  const script = `
import json
import sqlite3
import sys
db = sys.argv[1]
rows = json.loads(sys.argv[2])
columns = ${JSON.stringify(columnNames)}
defs = ${JSON.stringify(columnDefs)}
con = sqlite3.connect(db)
con.execute("CREATE TABLE sessions (%s)" % ",".join(defs))
keyed = {"id":0,"parent_session_id":1,"model":2,"billing_provider":3,"api_call_count":4,"tool_call_count":5,"input_tokens":6,"output_tokens":7}
for r in rows:
    vals = [r[keyed[c]] for c in columns]
    ph = ",".join("?" for _ in columns)
    con.execute("INSERT INTO sessions (%s) VALUES (%s)" % (",".join(columns), ph), vals)
con.commit()
con.close()
`;
  const res = spawnSync("python3", ["-c", script, dbPath, JSON.stringify(rows)], {
    encoding: "utf8",
    timeout: 5000,
    input: "",
  });
  expect(res.status, res.stderr).toBe(0);
  return dbPath;
}

/** Run a follow-up SQL against an existing temp database. */
function runSql(dbPath: string, statement: string): void {
  const script = `import sqlite3,sys\ncon=sqlite3.connect(sys.argv[1])\ncon.execute(sys.argv[2])\ncon.commit()\ncon.close()`;
  const r = spawnSync("python3", ["-c", script, dbPath, statement], {
    encoding: "utf8",
    timeout: 5000,
    input: "",
  });
  expect(r.status, r.stderr).toBe(0);
}

/** Invoke the embedded Python reader as the action would (python3 -c SCRIPT). */
function runUsagePython(dbPath: string, sessionId = "", includeChildren = "0") {
  const res = spawnSync(
    "python3",
    ["-c", HERMES_USAGE_SCRIPT, dbPath, sessionId, includeChildren],
    { encoding: "utf8", timeout: 5000, maxBuffer: 1024 * 1024, input: "" },
  );
  return {
    code: res.status ?? -1,
    stdout: res.stdout ?? "",
    stderr: res.stderr ?? "",
  };
}

function completed(stdout: string, stderr = ""): OpenShellSandboxBufferedCommandCompletion {
  return {
    outcome: { kind: "completed", exitCode: 0 },
    stdout,
    stderr,
  };
}

function failed(code: number, stderr: string): OpenShellSandboxBufferedCommandCompletion {
  return { outcome: { kind: "completed", exitCode: code }, stdout: "", stderr };
}

function executorReturning(completion: OpenShellSandboxBufferedCommandCompletion) {
  const exec: OpenShellSandboxBufferedCommandExecutor = {
    runBuffered: vi.fn(async () => completion),
  };
  return exec;
}

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
  vi.restoreAllMocks();
});

describe("hermes-usage-script (embedded Python reader)", () => {
  it("aggregates known counters and sorts models by model then provider", () => {
    const db = makeTempDb([
      ["a", null, "gpt-4o", "openai", 5, 2, 100, 50],
      ["b", null, "claude-3", "anthropic", 7, 3, 200, 100],
      ["c", null, "gpt-4o", "openai", 3, 1, 40, 20],
    ]);
    const r = runUsagePython(db);
    expect(r.code).toBe(0);
    const out = JSON.parse(r.stdout) as Report;
    expect(out.session_count).toBe(3);
    expect(out.totals).toEqual({
      api_calls: 15,
      tool_calls: 6,
      input_tokens: 340,
      output_tokens: 170,
    });
    expect(out.models.map((m) => m.model)).toEqual(["claude-3", "gpt-4o"]);
    expect(out.cost_usd).toBeNull();
    expect(out.incomplete_fields).toEqual([]);
  });

  it("returns zero counts and no incomplete_fields for an empty store", () => {
    const db = makeTempDb([]);
    const r = runUsagePython(db);
    expect(r.code).toBe(0);
    const out = JSON.parse(r.stdout) as Report;
    expect(out.session_count).toBe(0);
    expect(out.totals).toEqual({ api_calls: 0, tool_calls: 0, input_tokens: 0, output_tokens: 0 });
    expect(out.models).toEqual([]);
    expect(out.incomplete_fields).toEqual([]);
  });

  it("returns zero counts for an empty store even when counter columns are absent", () => {
    const db = makeTempDb(
      [],
      [
        "model",
        "billing_provider",
        "api_call_count",
        "tool_call_count",
        "input_tokens",
        "output_tokens",
      ],
    );
    const r = runUsagePython(db);
    expect(r.code).toBe(0);
    const out = JSON.parse(r.stdout) as Report;
    expect(out.session_count).toBe(0);
    expect(out.totals).toEqual({ api_calls: 0, tool_calls: 0, input_tokens: 0, output_tokens: 0 });
    expect(out.models).toEqual([]);
    expect(out.incomplete_fields).toEqual([]);
  });

  it("scopes to a single session by exact id", () => {
    const db = makeTempDb([
      ["s1", null, "m", "p", 1, 0, 10, 10],
      ["s2", null, "m", "p", 9, 0, 90, 90],
    ]);
    const r = runUsagePython(db, "s1", "0");
    const out = JSON.parse(r.stdout) as Report;
    expect(out.session_count).toBe(1);
    expect(out.totals.api_calls).toBe(1);
  });

  it("includes each descendant once and terminates on cycles", () => {
    const db = makeTempDb([
      ["a", null, "m", "p", 1, 0, 10, 10],
      ["b", "a", "m", "p", 1, 0, 10, 10],
      ["c", "b", "m", "p", 1, 0, 10, 10],
      ["d", "b", "m", "p", 1, 0, 10, 10],
      ["e", "c", "m", "p", 1, 0, 10, 10],
      ["other", null, "m", "p", 1, 0, 10, 10],
    ]);
    runSql(db, "UPDATE sessions SET parent_session_id='c' WHERE id='a'");
    const r = runUsagePython(db, "a", "1");
    const out = JSON.parse(r.stdout) as Report;
    expect(out.session_count).toBe(5);
    expect(out.totals.api_calls).toBe(5);
  });

  it("fails on an unknown session without emitting JSON", () => {
    const db = makeTempDb([["s1", null, "m", "p", 1, 0, 1, 1]]);
    const r = runUsagePython(db, "nope", "0");
    expect(r.code).not.toBe(0);
    expect(r.stdout).not.toContain("{");
    expect(r.stderr).toContain("unknown session");
  });

  it("rejects --include-children without --session", () => {
    const db = makeTempDb([["s1", null, "m", "p", 1, 0, 1, 1]]);
    const r = runUsagePython(db, "", "1");
    expect(r.code).not.toBe(0);
    expect(r.stderr).toContain("--include-children requires --session");
  });

  it("marks NULL counters as unknown in totals and incomplete_fields", () => {
    const db = makeTempDb([["s1", null, "m", "p", null, 2, 100, 50]]);
    const r = runUsagePython(db);
    const out = JSON.parse(r.stdout) as Report;
    expect(out.totals.api_calls).toBeNull();
    expect(out.totals.tool_calls).toBe(2);
    expect(out.incomplete_fields).toContain("api_calls");
    expect(out.models[0].api_calls).toBeNull();
  });

  it("marks a missing counter column as unknown", () => {
    const db = makeTempDb([["s1", null, "m", "p", null, 2, 100, 50]], ["api_call_count"]);
    const r = runUsagePython(db);
    const out = JSON.parse(r.stdout) as Report;
    expect(out.totals.api_calls).toBeNull();
    expect(out.incomplete_fields).toContain("api_calls");
  });

  it("raises a usage error on a negative counter instead of returning unknown", () => {
    const db = makeTempDb([["s1", null, "m", "p", -5, 2, 100, 50]]);
    const r = runUsagePython(db);
    expect(r.code).not.toBe(0);
    expect(r.stdout).not.toContain("{");
    expect(r.stderr).toContain("invalid counter value");
  });

  it("raises a usage error on a fractional (REAL) counter", () => {
    const db = makeTempDb([["s1", null, "m", "p", 1.5, 2, 100, 50]]);
    const r = runUsagePython(db);
    expect(r.code).not.toBe(0);
    expect(r.stderr).toContain("invalid counter value");
  });

  it("raises a usage error on a non-numeric TEXT counter", () => {
    const db = makeTempDb([["s1", null, "m", "p", null, 2, 100, 50]]);
    runSql(db, "UPDATE sessions SET api_call_count='bad' WHERE id='s1'");
    const r = runUsagePython(db);
    expect(r.code).not.toBe(0);
    expect(r.stderr).toContain("invalid counter value");
  });

  it("raises a usage error on a counter beyond JS safe integer", () => {
    const db = makeTempDb([["s1", null, "m", "p", 9007199254740992, 0, 1, 1]]);
    const r = runUsagePython(db);
    expect(r.code).not.toBe(0);
    expect(r.stderr).toContain("counter value out of safe integer range");
  });

  it("raises a usage error when a group total overflows the safe integer range", () => {
    const db = makeTempDb([
      ["s1", null, "m", "p", 9007199254740991, 0, 1, 1],
      ["s2", null, "m", "p", 1, 0, 1, 1],
    ]);
    const r = runUsagePython(db);
    expect(r.code).not.toBe(0);
    expect(r.stderr).toContain("counter value out of safe integer range");
  });

  it("groups rows under 'unknown' when model/provider columns are missing", () => {
    const db = makeTempDb([["s1", null, null, null, 1, 0, 1, 1]], ["model", "billing_provider"]);
    const r = runUsagePython(db);
    expect(r.code).toBe(0);
    const out = JSON.parse(r.stdout) as Report;
    expect(out.models[0].model).toBe("unknown");
    expect(out.models[0].provider).toBe("unknown");
    expect(out.incomplete_fields).toEqual([]);
  });

  it("fails cleanly on a schema missing a required id column", () => {
    const db = makeTempDb([["s1", null, "m", "p", 1, 0, 1, 1]], ["id"]);
    const r = runUsagePython(db);
    expect(r.code).not.toBe(0);
    expect(r.stdout).not.toContain("{");
    expect(r.stderr).toContain("usage-error:");
    expect(r.stderr).not.toContain("no such column");
  });

  it("fails cleanly on a schema missing parent_session_id", () => {
    const db = makeTempDb([["s1", null, "m", "p", 1, 0, 1, 1]], ["parent_session_id"]);
    const r = runUsagePython(db);
    expect(r.code).not.toBe(0);
    expect(r.stderr).toContain("sessions table is missing a required column");
  });

  it("opens a database whose path contains reserved characters", () => {
    const dir = mkdtempSync(join(tmpdir(), "nemoclaw-usage-odd#?"));
    tempDirs.push(dir);
    const dbPath = join(dir, "state.db");
    const script = `
import sqlite3,sys
con=sqlite3.connect(sys.argv[1])
con.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, model TEXT, billing_provider TEXT, api_call_count INTEGER, tool_call_count INTEGER, input_tokens INTEGER, output_tokens INTEGER)")
con.execute("INSERT INTO sessions VALUES ('s1', NULL, 'm', 'p', 2, 0, 10, 5)")
con.commit()
con.close()
`;
    const res = spawnSync("python3", ["-c", script, dbPath], { encoding: "utf8", input: "" });
    expect(res.status).toBe(0);
    const r = runUsagePython(dbPath);
    expect(r.code).toBe(0);
    const out = JSON.parse(r.stdout) as Report;
    expect(out.totals.api_calls).toBe(2);
  });

  it("never leaks prompt, message, path, or credential content", () => {
    const db = makeTempDb([["secret-session", null, "m", "p", 3, 0, 1, 1]]);
    const r = runUsagePython(db);
    const out = JSON.parse(r.stdout) as Report;
    expect(JSON.stringify(out)).not.toContain("secret");
    expect(r.stdout).not.toContain("message");
    expect(r.stdout).not.toContain("prompt");
    expect(r.stdout).not.toContain("password");
    expect(r.stdout).not.toContain("/sandbox/");
  });
});

describe("createSandboxSessionsUsageAction", () => {
  it.each(["", " "])("rejects an empty explicit session %j before transport", async (sessionId) => {
    const executor = executorReturning(completed(""));
    const action = createSandboxSessionsUsageAction({ sandboxCommandExecutor: executor });
    await expect(action.run({ sandboxName: "alpha", sessionId })).rejects.toThrow(/non-empty id/);
    expect(executor.runBuffered).not.toHaveBeenCalled();
  });

  it.each([["SECRET"], ["api_calls", "api_calls"]])(
    "rejects invalid incomplete fields %j",
    async (...incomplete_fields) => {
      const bogus = JSON.stringify({
        schema_version: 1,
        session_count: 0,
        totals: { api_calls: 0, tool_calls: 0, input_tokens: 0, output_tokens: 0 },
        models: [],
        incomplete_fields,
        cost_usd: null,
      });
      const action = createSandboxSessionsUsageAction({
        sandboxCommandExecutor: executorReturning(completed(bogus)),
      });
      await expect(action.run({ sandboxName: "alpha" })).rejects.toThrow(
        /^Failed to parse sessions usage output$/,
      );
    },
  );

  it("returns a validated report via the buffered executor", async () => {
    const db = makeTempDb([["s1", null, "gpt-4o", "openai", 2, 0, 10, 5]]);
    const r = runUsagePython(db);
    const action = createSandboxSessionsUsageAction({
      sandboxCommandExecutor: executorReturning(completed(r.stdout, r.stderr)),
    });
    const { report } = await action.run({ sandboxName: "alpha" });
    expect(report.session_count).toBe(1);
    expect(report.totals.api_calls).toBe(2);
  });

  it("surfaces an unknown session as a rejected error", async () => {
    const db = makeTempDb([["s1", null, "m", "p", 1, 0, 1, 1]]);
    const r = runUsagePython(db, "missing", "0");
    const action = createSandboxSessionsUsageAction({
      sandboxCommandExecutor: executorReturning(failed(r.code, r.stderr)),
    });
    await expect(action.run({ sandboxName: "alpha", sessionId: "missing" })).rejects.toThrow(
      /unknown session/,
    );
  });

  it("rejects --include-children without --session before invoking transport", async () => {
    const action = createSandboxSessionsUsageAction({
      sandboxCommandExecutor: executorReturning(completed("")),
    });
    await expect(action.run({ sandboxName: "alpha", includeChildren: true })).rejects.toThrow(
      /--include-children requires --session/,
    );
  });

  it("uses a fixed generic error instead of leaking transport exception text", async () => {
    const exec: OpenShellSandboxBufferedCommandExecutor = {
      runBuffered: vi.fn(async () => {
        throw new Error("SECRET transport stack trace with internal paths");
      }),
    };
    const action = createSandboxSessionsUsageAction({ sandboxCommandExecutor: exec });
    await expect(action.run({ sandboxName: "alpha" })).rejects.toThrow(
      /^Failed to read sessions usage$/,
    );
  });

  it("uses a fixed generic error instead of echoing arbitrary stderr", async () => {
    const action = createSandboxSessionsUsageAction({
      sandboxCommandExecutor: executorReturning(
        failed(1, "usage-error: SECRET database password leaked from stderr\n"),
      ),
    });
    await expect(action.run({ sandboxName: "alpha" })).rejects.toThrow(
      /^Failed to read sessions usage$/,
    );
  });

  it("rejects a report with unknown fields", async () => {
    const bogus = JSON.stringify({
      schema_version: 1,
      session_count: 0,
      totals: {},
      models: [],
      incomplete_fields: [],
      cost_usd: null,
      secret: "leak",
    });
    const action = createSandboxSessionsUsageAction({
      sandboxCommandExecutor: executorReturning(completed(bogus)),
    });
    await expect(action.run({ sandboxName: "alpha" })).rejects.toThrow(
      /^Failed to parse sessions usage output$/,
    );
  });

  it("rejects a report with a non-null cost_usd", async () => {
    const bogus = JSON.stringify({
      schema_version: 1,
      session_count: 0,
      totals: { api_calls: 0, tool_calls: 0, input_tokens: 0, output_tokens: 0 },
      models: [],
      incomplete_fields: [],
      cost_usd: 123,
    });
    const action = createSandboxSessionsUsageAction({
      sandboxCommandExecutor: executorReturning(completed(bogus)),
    });
    await expect(action.run({ sandboxName: "alpha" })).rejects.toThrow(
      /^Failed to parse sessions usage output$/,
    );
  });

  it("rejects a report with negative totals", async () => {
    const bogus = JSON.stringify({
      schema_version: 1,
      session_count: 1,
      totals: { api_calls: -1, tool_calls: 0, input_tokens: 0, output_tokens: 0 },
      models: [],
      incomplete_fields: [],
      cost_usd: null,
    });
    const action = createSandboxSessionsUsageAction({
      sandboxCommandExecutor: executorReturning(completed(bogus)),
    });
    await expect(action.run({ sandboxName: "alpha" })).rejects.toThrow(
      /^Failed to parse sessions usage output$/,
    );
  });

  it("rejects a non-JSON stdout with a fixed error", async () => {
    const action = createSandboxSessionsUsageAction({
      sandboxCommandExecutor: executorReturning(completed("not json {{ secret")),
    });
    await expect(action.run({ sandboxName: "alpha" })).rejects.toThrow(
      /^Failed to parse sessions usage output$/,
    );
  });

  it("rejects a non-Hermes sandbox", async () => {
    const action = createSandboxSessionsUsageAction({
      sandboxCommandExecutor: executorReturning(completed("")),
    });
    await expect(action.run({ sandboxName: "openclaw" })).rejects.toThrow(/not a Hermes sandbox/);
  });

  it("rejects an unregistered sandbox", async () => {
    const action = createSandboxSessionsUsageAction({
      sandboxCommandExecutor: executorReturning(completed("")),
    });
    await expect(action.run({ sandboxName: "unknown" })).rejects.toThrow(/not registered/);
  });
});
