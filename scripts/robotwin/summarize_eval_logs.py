"""Summarize RoboTwin evaluation logs into per-episode and per-task CSV files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TASK_RE = re.compile(r"^task_name:\s*(\S+)", re.MULTILINE)
RESULT_RE = re.compile(
    r"Success rate:\s*(\d+)/(\d+)\s*=>\s*([0-9.]+)%,\s*current seed:\s*(\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--episodes-csv", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    return parser.parse_args()


def parse_log(path: Path) -> tuple[str, list[dict[str, int]]]:
    text = ANSI_ESCAPE_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    task_match = TASK_RE.search(text)
    if task_match is None:
        raise ValueError(f"task_name is missing from {path}")

    task = task_match.group(1)
    episodes = []
    previous_successes = 0
    previous_trials = 0
    for successes_text, trials_text, _, seed_text in RESULT_RE.findall(text):
        successes = int(successes_text)
        trials = int(trials_text)
        if trials != previous_trials + 1:
            raise ValueError(f"non-consecutive trial count in {path}: {previous_trials} -> {trials}")
        outcome = successes - previous_successes
        if outcome not in (0, 1):
            raise ValueError(f"invalid success delta in {path}: {previous_successes} -> {successes}")
        episodes.append(
            {
                "task": task,
                "episode": trials,
                "seed": int(seed_text),
                "success": outcome,
            }
        )
        previous_successes = successes
        previous_trials = trials
    return task, episodes


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    log_paths = sorted(args.log_dir.glob("*_eval.log"))
    if not log_paths:
        raise FileNotFoundError(f"no *_eval.log files found under {args.log_dir}")

    episode_rows = []
    summary_rows = []
    for log_path in log_paths:
        task, episodes = parse_log(log_path)
        episode_rows.extend(episodes)
        successes = sum(row["success"] for row in episodes)
        trials = len(episodes)
        summary_rows.append(
            {
                "task": task,
                "successes": successes,
                "trials": trials,
                "success_rate": successes / trials if trials else 0.0,
            }
        )

    output_root = args.log_dir.parent
    episodes_csv = args.episodes_csv or output_root / "episodes.csv"
    summary_csv = args.summary_csv or output_root / "summary.csv"
    write_csv(episodes_csv, ["task", "episode", "seed", "success"], episode_rows)
    write_csv(summary_csv, ["task", "successes", "trials", "success_rate"], summary_rows)
    print(f"episodes={len(episode_rows)} tasks={len(summary_rows)}")
    print(f"episodes_csv={episodes_csv}")
    print(f"summary_csv={summary_csv}")


if __name__ == "__main__":
    main()
