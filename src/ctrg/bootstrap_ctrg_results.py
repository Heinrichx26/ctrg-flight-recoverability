from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RANKINGS = [
    ("ctrg_gap_max", "CTRG max-gap", False),
    ("out_dep_delay", "Delay-only", False),
    ("available_turn", "Slack-only", True),
    ("pred_recover", "Observed-path risk", True),
]


def to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin({"true", "1", "yes"})


def top_failure_rate(sample: pd.DataFrame, score: str, ascending: bool, frac: float, fail_col: str) -> float:
    n = max(1, int(np.ceil(len(sample) * frac)))
    if ascending:
        top = sample.nsmallest(n, score, keep="first")
    else:
        top = sample.nlargest(n, score, keep="first")
    return float(top[fail_col].mean())


def bootstrap_global(df: pd.DataFrame, horizon: int, reps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fail_col = f"fail_h{horizon}"
    groups = {a: g for a, g in df.groupby("airport", sort=False)}
    airports = np.array(list(groups.keys()), dtype=object)
    rows = []
    for score, label, ascending in RANKINGS:
        for frac in (0.05, 0.10, 0.20):
            vals = []
            for _ in range(reps):
                sampled = rng.choice(airports, size=len(airports), replace=True)
                sample = pd.concat([groups[a] for a in sampled], ignore_index=True)
                vals.append(top_failure_rate(sample, score, ascending, frac, fail_col))
            rows.append(
                {
                    "scope": "all_contexts",
                    "context_cluster": "",
                    "context_label": "",
                    "ranking": label,
                    "slice": f"top_{int(frac * 100)}pct",
                    "failure_rate_boot_mean": float(np.mean(vals)),
                    "failure_rate_ci_low": float(np.quantile(vals, 0.025)),
                    "failure_rate_ci_high": float(np.quantile(vals, 0.975)),
                    "bootstrap_reps": int(reps),
                    "airport_blocks": int(len(airports)),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_context(df: pd.DataFrame, horizon: int, reps: int, seed: int) -> pd.DataFrame:
    if "context_cluster" not in df.columns:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    fail_col = f"fail_h{horizon}"
    rows = []
    for (cluster, label), context_df in df.groupby(["context_cluster", "context_label"], sort=True):
        groups = {a: g for a, g in context_df.groupby("airport", sort=False)}
        airports = np.array(list(groups.keys()), dtype=object)
        if len(airports) < 2:
            continue
        for score, ranking, ascending in RANKINGS:
            vals = []
            for _ in range(reps):
                sampled = rng.choice(airports, size=len(airports), replace=True)
                sample = pd.concat([groups[a] for a in sampled], ignore_index=True)
                vals.append(top_failure_rate(sample, score, ascending, 0.10, fail_col))
            rows.append(
                {
                    "scope": "airport_context",
                    "context_cluster": int(cluster),
                    "context_label": label,
                    "ranking": ranking,
                    "slice": "top_10pct",
                    "failure_rate_boot_mean": float(np.mean(vals)),
                    "failure_rate_ci_low": float(np.quantile(vals, 0.025)),
                    "failure_rate_ci_high": float(np.quantile(vals, 0.975)),
                    "bootstrap_reps": int(reps),
                    "airport_blocks": int(len(airports)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    episode_path = results_dir / "episode_scores.csv"
    context_path = results_dir.parents[0] / "summary_tables" / "airport_context_membership_2025.csv"
    cols = [
        "airport",
        f"fail_h{args.horizon}",
        "supported",
        "stressed",
        "ctrg_gap_max",
        "out_dep_delay",
        "available_turn",
        "pred_recover",
    ]
    df = pd.read_csv(episode_path, usecols=cols, low_memory=False)
    for col in [f"fail_h{args.horizon}", "ctrg_gap_max", "out_dep_delay", "available_turn", "pred_recover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[to_bool(df["supported"]) & to_bool(df["stressed"])].copy()
    if context_path.exists():
        context = pd.read_csv(context_path)[["airport", "context_cluster", "context_label"]]
        df = df.merge(context, on="airport", how="left")

    global_boot = bootstrap_global(df, args.horizon, args.reps, args.seed)
    context_boot = bootstrap_context(df, args.horizon, args.reps, args.seed + 17)
    out = pd.concat([global_boot, context_boot], ignore_index=True)
    out.to_csv(results_dir / "top_slice_bootstrap.csv", index=False)
    print(out.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
