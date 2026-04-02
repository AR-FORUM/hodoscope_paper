#!/usr/bin/env python3
"""Average action_text length across all iQuest/SWE-bench actions."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hodoscope"))
from hodoscope.io import read_analysis_json

d = Path(__file__).resolve().parent.parent / "data" / "analysis_files" / "swebench"
lengths = []
for f in sorted(d.glob("*.hodoscope.json")):
    for s in read_analysis_json(f)["summaries"]:
        if s.get("embedding") is not None:
            lengths.append(len(s.get("action_text", "")))
json.dump({"n_actions": len(lengths), "avg_chars": round(sum(lengths)/len(lengths))}, sys.stdout)
print()
