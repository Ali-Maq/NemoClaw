// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createCliOpenShellSandboxCommandExecutor } from "../../../adapters/openshell/sandbox-command-cli";
import { getKnownSandboxTargetGatewayName } from "../gateway-state";
import * as registry from "../../../state/registry";
import { REPOSITORY_ROOT } from "../../../core/repository-root";
import type { OpenShellSandboxBufferedCommandExecutor } from "../../../adapters/openshell/sandbox-command";
import { HERMES_USAGE_SCRIPT } from "./hermes-usage-script";

/**
 * Read-only usage summary of a Hermes sandbox's sessions table.
 *
 * The action is deliberately read-only: it never starts, restarts, stops or
 * alters the sandbox lifecycle, and it never changes gateway selection. It
 * simply reads the Hermes sessions SQLite database via a small Python helper
 * invoked through the existing OpenShell buffered command executor.
 */

export interface UsageCounterTotals {
  api_calls: number | null;
  tool_calls: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
}

export interface UsageModelEntry extends UsageCounterTotals {
  model: string;
  provider: string;
  session_count: number;
}

export interface UsageReport {
  schema_version: number;
  session_count: number;
  totals: UsageCounterTotals;
  models: UsageModelEntry[];
  incomplete_fields: string[];
  cost_usd: null;
}

export interface UsageRequest {
  sandboxName: string;
  sessionId?: string;
  includeChildren?: boolean;
}

export interface UsageResult {
  report: UsageReport;
}

export const USAGE_EXECUTION_TIMEOUT_MILLISECONDS = 20000;
export const USAGE_CAPTURE_MAX_BUFFER_BYTES = 1024 * 1024;

const JS_MAX_SAFE_INTEGER = 9007199254740991;

// Exact script messages that are safe to surface verbatim. Anything else read
// from the transport (raw stderr, exception text, arbitrary response data) is
// replaced with a fixed generic error.
const SAFE_SCRIPT_MESSAGES = new Set<string>([
  "unknown session",
  "database not found",
  "sessions table unavailable",
  "sessions table is missing a required column",
  "selection exceeds 10000 sessions; narrow the request with --session",
  "invalid counter value",
  "counter value out of safe integer range",
  "failed to read sessions database",
  "--include-children requires --session",
  "include_children must be 0 or 1",
  "expected: db_path requested_session include_children",
]);

/**
 * Resolve the named gateway for a sandbox. Returns null when the sandbox is not
 * registered; throws when the registered agent is not Hermes.
 */
function resolveUsageTarget(sandboxName: string): { gatewayName: string } | null {
  const sandbox = registry.getSandbox(sandboxName);
  if (!sandbox) {
    return null;
  }
  if (sandbox.agent !== "hermes") {
    throw new UsageCommandError("not-hermes", `Sandbox '${sandboxName}' is not a Hermes sandbox.`);
  }
  const gatewayName = getKnownSandboxTargetGatewayName(sandboxName);
  if (!gatewayName) {
    return null;
  }
  return { gatewayName };
}

class UsageCommandError extends Error {
  public readonly kind: "not-hermes" | "unregistered" | "failed";
  public constructor(kind: "not-hermes" | "unregistered" | "failed", message: string) {
    super(message);
    this.kind = kind;
  }
}

const COUNTER_NAMES = ["api_calls", "tool_calls", "input_tokens", "output_tokens"] as const;

function isSafeNonNegativeInt(value: unknown): value is number {
  return (
    typeof value === "number" &&
    Number.isInteger(value) &&
    value >= 0 &&
    value <= JS_MAX_SAFE_INTEGER
  );
}

function isCounterValue(value: unknown): value is number | null {
  return value === null || isSafeNonNegativeInt(value);
}

function validateTotals(value: unknown): UsageCounterTotals {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new UsageCommandError("failed", "Failed to parse sessions usage output");
  }
  const obj = value as Record<string, unknown>;
  if (Object.keys(obj).length !== COUNTER_NAMES.length) {
    throw new UsageCommandError("failed", "Failed to parse sessions usage output");
  }
  const totals: UsageCounterTotals = {
    api_calls: null,
    tool_calls: null,
    input_tokens: null,
    output_tokens: null,
  };
  for (const name of COUNTER_NAMES) {
    if (!(name in obj) || !isCounterValue(obj[name])) {
      throw new UsageCommandError("failed", "Failed to parse sessions usage output");
    }
    totals[name] = obj[name] as number | null;
  }
  return totals;
}

function validateModels(value: unknown): UsageModelEntry[] {
  if (!Array.isArray(value)) {
    throw new UsageCommandError("failed", "Failed to parse sessions usage output");
  }
  return value.map((item) => {
    if (typeof item !== "object" || item === null || Array.isArray(item)) {
      throw new UsageCommandError("failed", "Failed to parse sessions usage output");
    }
    const obj = item as Record<string, unknown>;
    const allowed = new Set<string>([
      "model",
      "provider",
      "session_count",
      "api_calls",
      "tool_calls",
      "input_tokens",
      "output_tokens",
    ]);
    for (const key of Object.keys(obj)) {
      if (!allowed.has(key)) {
        throw new UsageCommandError("failed", "Failed to parse sessions usage output");
      }
    }
    if (typeof obj.model !== "string" || typeof obj.provider !== "string") {
      throw new UsageCommandError("failed", "Failed to parse sessions usage output");
    }
    if (!isSafeNonNegativeInt(obj.session_count)) {
      throw new UsageCommandError("failed", "Failed to parse sessions usage output");
    }
    return {
      model: obj.model,
      provider: obj.provider,
      session_count: obj.session_count,
      api_calls: validateCounter(obj.api_calls),
      tool_calls: validateCounter(obj.tool_calls),
      input_tokens: validateCounter(obj.input_tokens),
      output_tokens: validateCounter(obj.output_tokens),
    };
  });
}

function validateCounter(value: unknown): number | null {
  if (!isCounterValue(value)) {
    throw new UsageCommandError("failed", "Failed to parse sessions usage output");
  }
  return value;
}

function validateIncompleteFields(value: unknown): string[] {
  if (!Array.isArray(value)) {
    throw new UsageCommandError("failed", "Failed to parse sessions usage output");
  }
  for (const entry of value) {
    if (typeof entry !== "string" || !COUNTER_NAMES.some((name) => name === entry)) {
      throw new UsageCommandError("failed", "Failed to parse sessions usage output");
    }
  }
  if (new Set(value).size !== value.length) {
    throw new UsageCommandError("failed", "Failed to parse sessions usage output");
  }
  return value as string[];
}

/**
 * Strictly validate the parsed report against the stable contract. Only the
 * known fields are accepted; unknown fields, malformed types, negative or
 * unsafe totals, non-object models and any non-null cost are rejected. The
 * return value is rebuilt from validated fields only, so untrusted response
 * data is never spread into host output.
 */
function validateUsageReport(parsed: unknown): UsageReport {
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new UsageCommandError("failed", "Failed to parse sessions usage output");
  }
  const obj = parsed as Record<string, unknown>;
  const allowed = new Set<string>([
    "schema_version",
    "session_count",
    "totals",
    "models",
    "incomplete_fields",
    "cost_usd",
  ]);
  for (const key of Object.keys(obj)) {
    if (!allowed.has(key)) {
      throw new UsageCommandError("failed", "Failed to parse sessions usage output");
    }
  }
  if (obj.schema_version !== 1) {
    throw new UsageCommandError("failed", "Failed to parse sessions usage output");
  }
  if (!isSafeNonNegativeInt(obj.session_count)) {
    throw new UsageCommandError("failed", "Failed to parse sessions usage output");
  }
  if (obj.cost_usd !== null) {
    throw new UsageCommandError("failed", "Failed to parse sessions usage output");
  }
  return {
    schema_version: obj.schema_version,
    session_count: obj.session_count,
    totals: validateTotals(obj.totals),
    models: validateModels(obj.models),
    incomplete_fields: validateIncompleteFields(obj.incomplete_fields),
    cost_usd: null,
  };
}

export function createSandboxSessionsUsageAction(options: {
  sandboxCommandExecutor: OpenShellSandboxBufferedCommandExecutor;
}): {
  run(request: UsageRequest): Promise<UsageResult>;
} {
  const { sandboxCommandExecutor } = options;

  return {
    async run(request: UsageRequest): Promise<UsageResult> {
      const { sandboxName, sessionId, includeChildren } = request;

      if (sessionId !== undefined && sessionId.trim().length === 0) {
        throw new UsageCommandError("failed", "--session requires a non-empty id.");
      }

      if (includeChildren && !sessionId) {
        throw new UsageCommandError("failed", "--include-children requires --session <id>.");
      }

      const target = resolveUsageTarget(sandboxName);
      if (target === null) {
        throw new UsageCommandError("unregistered", `Sandbox '${sandboxName}' is not registered.`);
      }

      const command = [
        "python3",
        "-c",
        HERMES_USAGE_SCRIPT,
        "",
        sessionId ?? "",
        includeChildren ? "1" : "0",
      ];

      let result;
      try {
        result = await sandboxCommandExecutor.runBuffered({
          sandboxName,
          target: { kind: "named", gatewayName: target.gatewayName },
          command,
          timeoutMilliseconds: USAGE_EXECUTION_TIMEOUT_MILLISECONDS,
          timeoutKillSignal: "SIGKILL",
          outputLimitBytes: USAGE_CAPTURE_MAX_BUFFER_BYTES,
        });
      } catch {
        throw new UsageCommandError("failed", "Failed to read sessions usage");
      }

      if (result.outcome.kind !== "completed" || result.outcome.exitCode !== 0) {
        const scriptMessage = /usage-error:\s*(.+)$/m.exec(result.stderr)?.[1]?.trim();
        const safe =
          scriptMessage && SAFE_SCRIPT_MESSAGES.has(scriptMessage) ? scriptMessage : undefined;
        throw new UsageCommandError("failed", safe ?? "Failed to read sessions usage");
      }

      let parsed: unknown;
      try {
        parsed = JSON.parse(result.stdout) as unknown;
      } catch {
        throw new UsageCommandError("failed", "Failed to parse sessions usage output");
      }

      return { report: validateUsageReport(parsed) };
    },
  };
}

export const runSandboxSessionsUsage = createSandboxSessionsUsageAction({
  sandboxCommandExecutor: createCliOpenShellSandboxCommandExecutor({ hostCwd: REPOSITORY_ROOT }),
});
