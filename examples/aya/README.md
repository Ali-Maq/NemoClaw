# Hermes session usage

An experimental host command for the Ali-Maq fork of NemoClaw. It adds usage visibility inspired by [Claude Code usage reporting](https://code.claude.com/docs/en/costs). It reads recorded counters; it does not estimate a bill.

After the contributor setup and CLI build, run from this checkout:

```sh
node bin/nemohermes.js aya-feature sessions usage --json
node bin/nemohermes.js aya-feature sessions usage --session SESSION_ID --include-children --json
```

The sandbox must already be running and registered as Hermes. This command does not start or restart it. Existing experimental portable-profile restrictions still apply.

## Output

The report contains session count, model calls, tool calls, input/output tokens, and grouping by session model/provider metadata. Missing counters remain null. Empty stores report zero. Corrupt counters, overflowing sums, unknown session IDs and selections over 10,000 sessions fail explicitly. Dollar cost is unavailable. Grouping is not per-call attribution.

SQLite opens read-only. Only the sessions table is queried; messages are not exported. Remote execution is limited to 20 seconds and 1 MiB of captured output.

## How AYA built this

Native NemoHermes used Crusoe DeepSeek-V4-Pro in the existing OpenShell sandbox. The first implementation received an independent 35/62 result. Hermes repaired it and passed the same 62 cases. A separate review found an empty-filter bug and overly permissive metadata validation; the controller fixed both and added four tests. The final focused suite has 34 passing tests.

The independent evaluator ran against real temporary SQLite databases in a separate container with networking disabled. A live command over the completed parent and child sessions matched a separate read-only SQL aggregate: 2 sessions, 51 model calls, 61 tool calls, 102,802 input tokens and 37,620 output tokens. These are the implementation session counters, not the cost or usage of the whole AYA run.

```sh
npx vitest run --project cli src/lib/actions/sandbox/sessions/usage.test.ts
python3 scripts/aya/test-session-usage.py --compiled-module dist/lib/actions/sandbox/sessions/hermes-usage-script.js --json-report usage-contract.json
```

[The contract](session-usage.json) was frozen before implementation. The tests were separate from the worker. This example adapts documented behavior; it does not copy Claude source or demonstrate binary decompilation. The worker shared the existing sandbox and fetched test tooling there, so this run does not prove a fresh isolated sandbox for every job.

This is a personal-fork experiment awaiting review, not an NVIDIA-supported feature.
