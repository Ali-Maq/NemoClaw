// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flags } from "@oclif/core";

import {
  runSandboxSessionsUsage,
  type UsageReport,
} from "../../../lib/actions/sandbox/sessions/usage";
import { NemoClawCommand } from "../../../lib/cli/nemoclaw-oclif-command";

export default class SandboxSessionsUsageCommand extends NemoClawCommand {
  static id = "sandbox:sessions:usage";
  static strict = false;
  // Unknown `-`-prefixed tokens fall through to argv so run() can emit
  // actionable guidance instead of a raw "Nonexistent flag" error.
  static "--" = false;
  static summary = "Report measured model usage of a Hermes sandbox";
  static description = [
    "Summarize token and call counts recorded in a Hermes sandbox's sessions",
    "database, without exporting conversation content.",
    "",
    "Reads the Hermes sessions SQLite table in read-only mode. By default every",
    "session is aggregated; pass --session to scope to one session id and",
    "--include-children to also pull in its descendants (each exactly once,",
    "even across cycles).",
    "",
    "Model grouping uses per-session metadata (model/provider), not per-call",
    "attribution; a session whose model or provider is missing is grouped under",
    '"unknown". Per-session cost is not tracked, so cost_usd is always null',
    "and a price is never invented.",
  ].join("\n");
  static usage = ["<name> [--session <id>] [--include-children] [--json]"];
  static examples = [
    "<%= config.bin %> sandbox sessions usage alpha",
    "<%= config.bin %> sandbox sessions usage alpha --session abc123 --include-children",
    "<%= config.bin %> sandbox sessions usage alpha --json",
  ];
  static flags = {
    session: Flags.string({
      description: "Restrict the report to a single session id.",
    }),
    "include-children": Flags.boolean({
      description: "With --session, include all descendant sessions (each once).",
      default: false,
    }),
    json: Flags.boolean({
      description: "Print the report as JSON.",
      default: false,
    }),
  };

  public async run(): Promise<void> {
    const { flags, argv } = await this.parse(SandboxSessionsUsageCommand);
    const [sandboxName, ...rest] = argv as string[];

    if (!sandboxName) {
      this.failWithLines([`  Usage: ${SandboxSessionsUsageCommand.usage[0]}`], 2);
      return;
    }

    if (rest.length > 0) {
      const stray = rest.filter((token) => token.startsWith("-"));
      const extra = rest.filter((token) => !token.startsWith("-"));
      const lines = [`  Unexpected argument${extra.length > 0 ? "s" : ""}: ${rest.join(" ")}`];
      if (stray.length > 0) {
        lines.push(`  Unknown flag: ${stray.join(", ")}`);
      }
      lines.push(`  Usage: ${SandboxSessionsUsageCommand.usage[0]}`);
      this.failWithLines(lines, 2);
      return;
    }

    if (flags.session !== undefined && flags.session.trim().length === 0) {
      this.failWithLines(["  --session requires a non-empty id."], 2);
      return;
    }

    if (flags["include-children"] && !flags.session) {
      this.failWithLines(
        [
          "  --include-children requires --session <id>.",
          `  Usage: ${SandboxSessionsUsageCommand.usage[0]}`,
        ],
        2,
      );
      return;
    }

    try {
      const { report } = await runSandboxSessionsUsage.run({
        sandboxName,
        sessionId: flags.session,
        includeChildren: flags["include-children"],
      });

      if (flags.json) {
        this.log(JSON.stringify({ sandbox: sandboxName, ...report }));
        return;
      }

      this.printHumanReport(sandboxName, report);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.failWithLines([`  ${message}`], 1);
    }
  }

  private printHumanReport(sandboxName: string, report: UsageReport): void {
    const lines: string[] = [];
    lines.push(`Sessions usage for '${sandboxName}'`);
    lines.push(`${report.session_count} session${report.session_count === 1 ? "" : "s"}`);
    lines.push(this.formatCounter("API calls", report.totals.api_calls));
    lines.push(this.formatCounter("Tool calls", report.totals.tool_calls));
    lines.push(this.formatCounter("Input tokens", report.totals.input_tokens));
    lines.push(this.formatCounter("Output tokens", report.totals.output_tokens));
    lines.push("Cost: unavailable");
    if (report.incomplete_fields.length > 0) {
      lines.push(`Incomplete: ${report.incomplete_fields.join(", ")}`);
    }
    if (report.models.length > 0) {
      lines.push("");
      lines.push("By model:");
      for (const entry of report.models) {
        const label =
          entry.provider === "unknown" ? entry.model : `${entry.model} (${entry.provider})`;
        lines.push(
          `  ${label}: ${entry.session_count} session${entry.session_count === 1 ? "" : "s"}, ` +
            `calls ${this.formatInline(entry.api_calls)}, ` +
            `tool calls ${this.formatInline(entry.tool_calls)}, ` +
            `in ${this.formatInline(entry.input_tokens)}, ` +
            `out ${this.formatInline(entry.output_tokens)}`,
        );
      }
    }
    this.log(lines.join("\n"));
  }

  private formatCounter(label: string, value: number | null): string {
    if (value === null) {
      return `${label}: unknown`;
    }
    return `${label}: ${value}`;
  }

  private formatInline(value: number | null): string {
    return value === null ? "unknown" : String(value);
  }
}
