# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Collects LM Studio model status and per-request stats into SQLite for Grafana.

Two collection loops, run as daemon threads:
  - poll_ps: polls `lms ps --json` on an interval for live per-model status
    (idle / processingPrompt / generating), queue depth, and TTL countdown.
  - tail_log_stream: tails `lms log stream --json --stats -s model` for
    per-request stats (tokens/sec, TTFT, prompt/completion token counts) as
    each prediction finishes.
"""
import argparse
import json
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).parent / "var" / "lms_metrics.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_status (
    ts REAL NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    queued INTEGER NOT NULL,
    parallel INTEGER NOT NULL,
    ttl_remaining_s REAL
);
CREATE INDEX IF NOT EXISTS idx_model_status_ts ON model_status(ts);

CREATE TABLE IF NOT EXISTS predictions (
    ts REAL NOT NULL,
    model TEXT NOT NULL,
    tokens_per_second REAL,
    time_to_first_token_s REAL,
    total_time_s REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    stop_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts);
"""


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def poll_ps(db_path: Path, interval: float, stop_event: threading.Event) -> None:
    conn = sqlite3.connect(db_path)
    while not stop_event.is_set():
        try:
            proc = subprocess.run(
                ["lms", "ps", "--json"], capture_output=True, text=True, timeout=5
            )
            models = json.loads(proc.stdout) if proc.stdout.strip() else []
            now = time.time()
            now_ms = now * 1000
            rows = []
            for m in models:
                last_used = m.get("lastUsedTime")
                ttl_ms = m.get("ttlMs")
                ttl_remaining = None
                if last_used and ttl_ms:
                    ttl_remaining = max(0.0, (last_used + ttl_ms - now_ms) / 1000.0)
                rows.append(
                    (
                        now,
                        m["identifier"],
                        m["status"],
                        m["queued"],
                        m["parallel"],
                        ttl_remaining,
                    )
                )
            if rows:
                conn.executemany(
                    "INSERT INTO model_status "
                    "(ts, model, status, queued, parallel, ttl_remaining_s) "
                    "VALUES (?,?,?,?,?,?)",
                    rows,
                )
                conn.commit()
        except Exception as e:
            print(f"[ps] error: {e}", file=sys.stderr)
        stop_event.wait(interval)
    conn.close()


def tail_log_stream(
    db_path: Path, stop_event: threading.Event, proc_holder: list
) -> None:
    conn = sqlite3.connect(db_path)
    proc = subprocess.Popen(
        ["lms", "log", "stream", "--json", "--stats", "-s", "model"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    proc_holder.append(proc)
    try:
        for line in proc.stdout:
            if stop_event.is_set():
                break
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            data = evt.get("data", {})
            if data.get("type") != "llm.prediction.output":
                continue
            stats = data.get("stats") or {}
            conn.execute(
                "INSERT INTO predictions "
                "(ts, model, tokens_per_second, time_to_first_token_s, "
                "total_time_s, prompt_tokens, completion_tokens, stop_reason) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    time.time(),
                    data.get("modelIdentifier"),
                    stats.get("tokensPerSecond"),
                    stats.get("timeToFirstTokenSec"),
                    stats.get("totalTimeSec"),
                    stats.get("promptTokensCount"),
                    stats.get("predictedTokensCount"),
                    stats.get("stopReason"),
                ),
            )
            conn.commit()
    finally:
        proc.terminate()
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--interval", type=float, default=1.0, help="lms ps poll interval, seconds"
    )
    args = parser.parse_args()

    init_db(args.db_path)
    stop_event = threading.Event()
    proc_holder = []
    threads = [
        threading.Thread(
            target=poll_ps, args=(args.db_path, args.interval, stop_event), daemon=True
        ),
        threading.Thread(
            target=tail_log_stream,
            args=(args.db_path, stop_event, proc_holder),
            daemon=True,
        ),
    ]
    for t in threads:
        t.start()

    def handle_stop(signum, frame):
        stop_event.set()
        for proc in proc_holder:
            proc.terminate()

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    print(f"Writing metrics to {args.db_path} (Ctrl-C to stop)")
    while not stop_event.is_set():
        time.sleep(0.5)
    for t in threads:
        t.join(timeout=5)


if __name__ == "__main__":
    main()
