"""
Discriminative baseline: train a classifier to predict group from embeddings,
rank target-group actions by classifier confidence (most distinctive first).

Two variants:
  1. logistic_regression: multinomial logistic regression, rank by P(target_group)
  2. random_forest: random forest classifier, rank by P(target_group)

No t-SNE needed — operates on full-dimensional embeddings.
Uses same subsampling/seeding as run_table2.py for comparability.
"""

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hodoscope"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hodoscope.io import read_analysis_json
from hodoscope.sampling import UNRANKED_SENTINEL, collect_plot_data

# Reuse data loaders and oracles from run_table2
from run_table2 import (
    load_testbed_commit0,
    load_testbed_impossiblebench,
    load_testbed_iquest,
    oracle_commit0,
    oracle_impossiblebench,
    oracle_iquest,
    find_first_hit,
    count_hits_at_k,
    TOP_KS,
    N_SEEDS,
    MAX_PER_GROUP,
)


def run_discriminative_baseline(
    summaries_by_group, oracle_fn, target_group, testbed_name=""
):
    print(f"\n{'='*60}")
    print(f"Testbed: {testbed_name} (target: {target_group})")
    print(f"{'='*60}")

    data = collect_plot_data(summaries_by_group)
    X = data.X
    labels = data.labels
    type_names = data.type_names
    N = len(X)

    target_label = type_names.index(target_group)
    n_target = int((labels == target_label).sum())

    # Build oracle mask and char lengths
    oracle_mask = np.zeros(N, dtype=bool)
    char_lengths = np.zeros(N, dtype=np.int64)
    idx = 0
    for group_name in type_names:
        for s in summaries_by_group[group_name]:
            char_lengths[idx] = len(s.get("action_text", ""))
            if group_name == target_group:
                oracle_mask[idx] = oracle_fn(s)
            idx += 1

    n_oracle = int(oracle_mask.sum())
    target_chars = int(char_lengths[labels == target_label].sum())

    print(f"Groups: {type_names}")
    print(f"Total actions: {N}, embed_dim: {X.shape[1]}")
    print(f"Target group: {target_group} ({n_target} actions, label={target_label})")
    print(f"Oracle-positive actions: {n_oracle}/{n_target} ({100*n_oracle/n_target:.1f}%)")
    print(f"Target group chars: {target_chars:,}")

    METHODS = ["logistic_regression", "random_forest"]
    method_ranks = {m: [] for m in METHODS}
    method_chars = {m: [] for m in METHODS}
    hits_at_k = {m: {k: [] for k in TOP_KS} for m in METHODS}

    for seed_idx in range(N_SEEDS):
        seed = seed_idx + 1
        t0 = time.time()
        print(f"  Seed {seed}/{N_SEEDS}...", flush=True)

        # Subsample 50% per group (same as run_table2)
        rng = np.random.RandomState(seed)
        sub_idx = []
        for g in range(len(type_names)):
            g_idx = np.where(labels == g)[0]
            n_sub = max(1, len(g_idx) // 2)
            chosen = rng.choice(g_idx, size=n_sub, replace=False)
            sub_idx.extend(chosen.tolist())
        sub_idx = sorted(sub_idx)

        X_sub = X[sub_idx]
        labels_sub = labels[sub_idx]
        oracle_sub = oracle_mask[sub_idx]
        chars_sub = char_lengths[sub_idx]

        target_indices = np.where(labels_sub == target_label)[0]
        non_target_mask = labels_sub != target_label

        for method in METHODS:
            # Train classifier on all data to predict group
            if method == "logistic_regression":
                clf = LogisticRegression(
                    max_iter=1000, random_state=seed, n_jobs=-1,
                    multi_class="multinomial", solver="lbfgs",
                )
            else:
                clf = RandomForestClassifier(
                    n_estimators=100, random_state=seed, n_jobs=-1,
                )

            clf.fit(X_sub, labels_sub)

            # Get P(target_group) for target-group actions
            probs = clf.predict_proba(X_sub[target_indices])
            target_class_idx = list(clf.classes_).index(target_label)
            target_probs = probs[:, target_class_idx]

            # Rank by descending probability (most distinctive first)
            sorted_order = np.argsort(-target_probs)

            ranks_list = [UNRANKED_SENTINEL] * len(labels_sub)
            for rank, local_idx in enumerate(sorted_order[:MAX_PER_GROUP]):
                ranks_list[target_indices[local_idx]] = rank

            hit, ch = find_first_hit(
                ranks_list, labels_sub, target_label, oracle_sub, chars_sub
            )

            if hit is not None:
                method_ranks[method].append(hit)
                method_chars[method].append(ch)

            hk = count_hits_at_k(ranks_list, labels_sub, target_label, oracle_sub)
            for k in TOP_KS:
                hits_at_k[method][k].append(hk[k])

        elapsed = time.time() - t0
        print(f"    time={elapsed:.0f}s  N_sub={len(sub_idx)} target={len(target_indices)} oracle={oracle_sub[target_indices].sum()}")
        for method in METHODS:
            r = method_ranks[method][-1] if method_ranks[method] else "N/A"
            print(f"    {method}={r}")

    # Results
    results = {}
    for name in METHODS:
        ranks = method_ranks[name]
        chars = method_chars[name]
        if ranks:
            results[name] = {
                "rank_mean": np.mean(ranks), "rank_std": np.std(ranks),
                "pct_mean": 100 * np.mean(ranks) / n_target,
                "chars_mean": np.mean(chars), "chars_std": np.std(chars),
                "chars_pct": 100 * np.mean(chars) / target_chars,
                "hits_at_k": {k: np.mean(hits_at_k[name][k]) for k in TOP_KS},
            }

    print(f"\n  Results for {testbed_name} (target={target_group}, "
          f"N_target={n_target}, oracle={n_oracle}, chars={target_chars:,}):")
    for method, r in results.items():
        hk = r["hits_at_k"]
        hk_str = "  ".join(f"@{k}:{hk[k]:.1f}" for k in TOP_KS)
        print(f"    {method:25s}: rank={r['rank_mean']:.1f}±{r['rank_std']:.1f}"
              f"  ({r['pct_mean']:.2f}% actions)"
              f"  chars_pct={r['chars_pct']:.3f}%"
              f"  hits: {hk_str}")

    return results


def main():
    testbeds = [
        ("Commit0", load_testbed_commit0, oracle_commit0, "MiniMax-M2.5"),
        ("ImpossibleBench", load_testbed_impossiblebench, oracle_impossiblebench, "impossible"),
        ("iQuest/SWE-bench", load_testbed_iquest, oracle_iquest, "iquest"),
    ]

    all_results = {}
    for name, loader, oracle, target in testbeds:
        data = loader()
        results = run_discriminative_baseline(data, oracle, target, name)
        all_results[name] = results

    # Summary table
    print("\n" + "=" * 100)
    print("DISCRIMINATIVE BASELINE: Discovery effort (rank to first hit)")
    print("=" * 100)
    testbed_names = ["Commit0", "ImpossibleBench", "iQuest/SWE"]
    methods = ["logistic_regression", "random_forest"]

    print(f"\n{'Method':<25s} | {'Commit0':>20s} | {'ImpossibleBench':>20s} | {'iQuest/SWE':>20s}")
    print("-" * 92)
    for method in methods:
        parts = []
        for tb in testbed_names:
            r = all_results.get(tb, {}).get(method)
            if r:
                mean, std = r["rank_mean"], r["rank_std"]
                pct = r["pct_mean"]
                if std < 0.5:
                    parts.append(f"{mean:.0f} / {pct:.2f}%")
                else:
                    parts.append(f"{mean:.0f}±{std:.0f} / {pct:.2f}%")
            else:
                parts.append("N/A")
        print(f"{method:<25s} | {parts[0]:>20s} | {parts[1]:>20s} | {parts[2]:>20s}")


if __name__ == "__main__":
    main()
