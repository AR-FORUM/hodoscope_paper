# Hodoscope: Unsupervised Monitoring for AI Misbehaviors

Ziqian Zhong, Shashwat Saxena, Aditi Raghunathan

📝 Arxiv *(to come)* | [🖇️ Livepaper](https://hodoscope.dev/blog/livepaper.html) | [📦 Package](https://github.com/AR-FORUM/hodoscope) | [🌐 Website](https://hodoscope.dev)

This repository contains the paper, data references, and replication scripts for *Hodoscope: Unsupervised Monitoring for AI Misbehaviors*. It is structured so that every table and inline number in the paper can be regenerated from the scripts in [`experiments/`](experiments/). A [live paper](#live-paper) version is also provided and are preferred for agentic replication.

## What's in this repo

```
hodoscope_paper/
├── experiments/            # Standalone replication scripts (see below)
│   ├── run_table2.py                # Table 2 + Table 4 (sampling strategy comparison)
│   ├── table3_supervised.py         # Table 3 (supervised monitor AUC / TPR)
│   ├── dataset_stats.py             # Table 5 (per-testbed action / oracle counts)
│   ├── run_discriminative_baseline.py  # Logistic regression & random forest baselines
│   ├── iquest_gitlog_count.py       # Inline: `git log` action count on iQuest/SWE-bench
│   ├── avg_action_length.py         # Inline: average stored action_text length
│   ├── monitor.py                   # Re-run the supervised monitor on a trajectory file
│   └── prompts/{v1,v2}.txt          # Baseline and augmented monitor prompts
├── hodoscope/              # Git submodule: the Hodoscope library (pip install -e hodoscope/)
└── data/                   # Precomputed analysis files and monitor runs (to be downloaded from HuggingFace)
    ├── analysis_files/             # Default pipeline (gpt-5.2 summarizer + Gemini embeddings)
    ├── analysis_files_nosummary/   # Ablation: embed raw action text directly
    ├── analysis_files_weak/        # Ablation: gpt-4o-mini + text-embedding-3-large
    └── supervised_monitor_runs/    # Pre-scored monitor outputs for Table 3
├── livepaper/              # Markdown version of paper (paper.md), replication specs, figures, visualizations
│   ├── paper.md            # Full paper text (all sections, appendices, references)
│   ├── specs.yaml          # Livepaper replication specs (one per inline number / table)
│   ├── livepaper.yaml      # Livepaper setup commands
│   └── assets/             # Figures (PNG/PDF) and interactive HTML visualizations
```

## Setup

```bash
# 1. Clone the repository with the hodoscope submodule
git clone --recurse-submodules https://github.com/AR-FORUM/hodoscope_paper.git
cd hodoscope_paper

# 2. Install the Hodoscope library + replication dependencies
pip install -e hodoscope/
pip install hdbscan scikit-learn     # needed by experiments/run_table2.py

# 3. Download the precomputed analysis files (~several GB) from HuggingFace
huggingface-cli download fjzzq2002/hodoscope-paper-data --repo-type dataset --local-dir data
```

All replication scripts load precomputed summaries and embeddings from `data/analysis_files*/`, so no LLM API keys are required for Tables 2–5. API keys are only needed if you want to re-run the full pipeline end-to-end on raw trajectories (see [Hodoscope README](hodoscope/README.md)) or to re-score trajectories with `experiments/monitor.py`.

**Runtime note.** Table 2 takes roughly 20 minutes per run on CPU (dominated by 10 t-SNE projections with 50 % subsampling per seed). Tables 3 and 5 complete in seconds.

## Reproducing the results

### Table 1 — Commit0 score drops after patching (external data)

The before/after scores in Table 1 come from the [OpenHands Index](https://github.com/OpenHands/openhands-index-results) commit history. To verify each row, compare `scores.json` at the re-run commit against its parent commit. Example for MiniMax-M2.5 (requires an authenticated `gh` CLI):

```bash
# Before (parent commit) → expected 50.0
gh api "repos/OpenHands/openhands-index-results/contents/results/MiniMax-M2.5/scores.json?ref=d5a5e9b487~1" \
  --jq '.content' | base64 -d | \
  python3 -c "import sys,json;d=json.load(sys.stdin);[print(e['score']) for e in d if e.get('benchmark')=='commit0']"

# After (re-run commit) → expected 18.8
gh api "repos/OpenHands/openhands-index-results/contents/results/MiniMax-M2.5/scores.json?ref=d5a5e9b487" \
  --jq '.content' | base64 -d | \
  python3 -c "import sys,json;d=json.load(sys.stdin);[print(e['score']) for e in d if e.get('benchmark')=='commit0']"
```

The remaining re-run commits are: MiniMax-M2.1 `b7fb8bc26b`, GPT-5.2-Codex `570ecf0664`, DeepSeek-V3.2-Reasoner `60df1f27d2`, Kimi-K2-Thinking `cf581d358b`. Replace the hash and `results/<Model>/` in the commands above to verify each row. Runtime values use the same commits with `e['average_runtime']` instead of `e['score']`.

### Table 2 + Table 4 — Discovery effort across three testbeds

Table 2 reports rank-to-first-hit and fraction-inspected for density-contrast FPS and its ablations; Table 4 reports the same runs with two additional metrics (% characters examined, hits@20). Both are produced by the same script under three data configurations:

```bash
# Main rows: Hodoscope, −diffing, −density, HDBSCAN, Uniform random, distance-only
python experiments/run_table2.py --seeds 10

# −summarization row (raw action text embedded directly)
python experiments/run_table2.py --seeds 10 --data-dir data/analysis_files_nosummary

# weaker models row (gpt-4o-mini summarizer + text-embedding-3-large)
python experiments/run_table2.py --seeds 10 --data-dir data/analysis_files_weak
```

Each invocation prints a final summary table for all three testbeds (Commit0, ImpossibleBench, iQuest/SWE-bench) along with hits@{20, 50, 100} and % characters examined. Compare against Tables 2 and 4 in [`livepaper/paper.md`](livepaper/paper.md).

t-SNE is stochastic and each seed subsamples 50 % per group, so exact ranks vary slightly across runs; the relative ordering and magnitudes should match. The oracle definitions (what counts as a "first hit") are in Appendix H of the paper and in `run_table2.py:oracle_commit0`, `oracle_impossiblebench`, `oracle_iquest`.

Optional: to compare against logistic regression / random forest discriminative baselines (not shown in the paper but referenced as a control):

```bash
python experiments/run_discriminative_baseline.py
```

### Table 3 — Supervised monitor improvements on ImpossibleBench

```bash
python experiments/table3_supervised.py
```

Prints JSON with baseline vs. augmented AUC and TPR@5%FPR for each of the four monitors (GPT-4o-mini, GPT-4.1, GPT-5.4-nano, GPT-5.4-mini), averaged over three acting models (gpt5, o3, o4-mini). The script reads pre-scored monitor outputs from `data/supervised_monitor_runs/{monitor}/{v1,v2}/{actor}_{attack,honest}.json`; no API calls are made.

The two prompt variants used to produce those monitor outputs are in [`experiments/prompts/v1.txt`](experiments/prompts/v1.txt) (baseline) and [`experiments/prompts/v2.txt`](experiments/prompts/v2.txt) (augmented with six Hodoscope-discovered exemplars).

To re-score a trajectory file with the monitor yourself (requires `OPENAI_API_KEY`):

```bash
pip install openai python-dotenv
python experiments/monitor.py path/to/trajectory.json --prompt v2 --model gpt-5.4-nano
```

### Table 5 — Dataset statistics

```bash
python experiments/dataset_stats.py
```

Prints JSON with per-testbed action counts, oracle-positive counts and percentages, and a grand total (`total_actions` across all testbeds, ~63,686).

### Inline values

A few inline numbers in the paper have their own one-line scripts:

```bash
# "79 / 4006 / 2.0%" — git log actions on iQuest
python experiments/iquest_gitlog_count.py

# "~1,857" — average stored action_text length on iQuest/SWE-bench
python experiments/avg_action_length.py
```

Experimental design constants (`N_SEEDS = 10`, `N_UNIFORM = 100`, `MAX_PER_GROUP = 500`, 50 % subsampling per seed) are defined near the top of `experiments/run_table2.py` and can be read directly.

## Interactive visualizations

The three interactive Hodoscope visualizations from Section 3 are checked into this repo under [`livepaper/assets/visualizations/`](livepaper/assets/visualizations/). Open them in a browser to see the full point clouds with density-difference overlays and click-to-inspect panels:

- [`commit0.html`](livepaper/assets/visualizations/commit0.html) — 15 models on Commit0, MiniMax-M2.5 overlay highlights git-history exploitation
- [`impossiblebench.html`](livepaper/assets/visualizations/impossiblebench.html) — GPT-5 on ImpossibleBench vs. vanilla SWE-bench
- [`swebench_iquest.html`](livepaper/assets/visualizations/swebench_iquest.html) — iQuest-Coder-V1 vs. four frontier models on SWE-bench

To regenerate any of these (or produce visualizations for new data) using the `hodoscope` CLI, see [`hodoscope/README.md`](hodoscope/README.md). The one-liner is `hodoscope viz data/analysis_files/commit0/ --group-by model -o commit0.html`.

## Live paper

[`livepaper/`](livepaper/) contains a more agent-replication-friendly version of the paper generated with the [livepaper](https://github.com/fjzzq2002/livepaper) harness. Please refer to the [livepaper version of the paper](https://hodoscope.dev/blog/livepaper.html) for more details.

## Citation

```bibtex
@article{zhong2026hodoscope,
  title   = {Hodoscope: Unsupervised Monitoring for AI Misbehaviors},
  author  = {Zhong, Ziqian and Saxena, Shashwat and Raghunathan, Aditi},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```
