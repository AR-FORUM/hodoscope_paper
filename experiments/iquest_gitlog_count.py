#!/usr/bin/env python3
"""
Count iQuest actions containing "git log".

Usage:
    python experiments/iquest_gitlog_count.py

Prints JSON to stdout.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hodoscope"))
from hodoscope.io import read_analysis_json

ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "data" / "analysis_files"


def main():
    doc = read_analysis_json(ANALYSIS_DIR / "swebench" / "iquest_samples.hodoscope.json")
    summaries = [s for s in doc["summaries"] if s.get("embedding") is not None]
    total = len(summaries)
    git_log = sum(1 for s in summaries if "git log" in s["action_text"].lower())

    json.dump({
        "total_iquest_actions": total,
        "git_log_actions": git_log,
        "pct": round(100 * git_log / total, 1),
    }, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
