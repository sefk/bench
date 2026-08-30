# lms-monitor

Collects LM Studio activity into a local SQLite file so it can be graphed in
Grafana: per-model live phase (idle / prefill / decode), queue depth, TTL
countdown, and per-request throughput (tokens/sec, time-to-first-token,
prompt/completion token counts).

Not exposed by LM Studio: cache hit/miss and true per-slot occupancy within a
model (only aggregate model-level phase and queue depth are available).

## Run it

```
uv run collector.py
```

Writes to `var/lms_metrics.sqlite` by default (gitignored). Options:

```
uv run collector.py --db-path /path/to/db.sqlite --interval 1.0
```

Requires the `lms` CLI on `PATH` and the LM Studio server running
(`lms server start` or via the app).

## Wire it into Grafana

1. Install the [SQLite datasource plugin][sqlite-ds] if not already present
   (already installed in this machine's Grafana instance).
2. Add a datasource: type "SQLite", path pointing at this machine's
   `lms-monitor/var/lms_metrics.sqlite` (absolute path).
3. Import `grafana-dashboard.json` (Dashboards → Import → paste JSON), and
   pick the SQLite datasource you just added when prompted.

The collector needs to be running for the dashboard to show live data — it's
a foreground process, so run it in a terminal, `tmux`, or under whatever
process supervisor you like alongside `lms server`.

[sqlite-ds]: https://grafana.com/grafana/plugins/frser-sqlite-datasource/

## Schema

- `model_status(ts, model, status, queued, parallel, ttl_remaining_s)` — one
  row per model per poll interval (default 1s). `status` is one of `idle`,
  `processingPrompt` (prefill), `generating` (decode), etc., straight from
  `lms ps --json`.
- `predictions(ts, model, tokens_per_second, time_to_first_token_s,
  total_time_s, prompt_tokens, completion_tokens, stop_reason)` — one row per
  finished request, from `lms log stream --json --stats -s model`.
