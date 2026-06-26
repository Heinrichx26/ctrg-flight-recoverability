from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from run_ctrg_experiment import (
    MODEL_CATEGORICAL,
    MODEL_NUMERIC,
    ExperimentConfig,
    airport_table,
    bootstrap_top_slice,
    cause_table,
    delay_band_table,
    monthly_table,
    predict_tree,
    split_train_eval,
    top_slice_table,
    train_model,
    train_tree_model,
)


ROOT = Path(__file__).resolve().parents[2]

BASE_USECOLS = [
    "episode_id",
    "tail",
    "carrier",
    "airport",
    "dest",
    "flight_date",
    "sched_dep_dt",
    "hour",
    "month",
    "day_of_week",
    "in_arr_delay",
    "out_dep_delay",
    "available_turn",
    "planned_turn",
    "turn_slack",
    "taxi_out",
    "taxi_in",
    "distance_group",
    "dep_time_blk",
    "arr_time_blk",
    "is_cancelled",
    "is_diverted",
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "late_aircraft_delay",
    "same_tail_turn_index",
    "airport_hour_dep_rate",
    "airport_hour_mean_dep_delay",
    "airport_day_weather_delay_share",
    "airport_day_late_aircraft_share",
    "airport_day_cancel_share",
]


def log(message: str) -> None:
    print(message, flush=True)


def load_turnarounds(path: Path, horizon: int) -> pd.DataFrame:
    needed = sorted(
        set(
            BASE_USECOLS
            + MODEL_NUMERIC
            + MODEL_CATEGORICAL
            + [
                f"recover_h{horizon}",
                f"fail_h{horizon}",
                f"endpoint_obs_h{horizon}",
            ]
        )
    )
    header = pd.read_csv(path, nrows=0)
    usecols = [c for c in needed if c in set(header.columns)]
    turn = pd.read_csv(path, usecols=usecols, low_memory=False)
    turn["sched_dep_dt"] = pd.to_datetime(turn["sched_dep_dt"], errors="coerce")
    for col in [
        *MODEL_NUMERIC,
        f"recover_h{horizon}",
        f"fail_h{horizon}",
        f"endpoint_obs_h{horizon}",
        "is_cancelled",
        "is_diverted",
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "late_aircraft_delay",
    ]:
        if col in turn.columns:
            turn[col] = pd.to_numeric(turn[col], errors="coerce")
    for col in MODEL_CATEGORICAL + ["episode_id", "tail"]:
        if col in turn.columns:
            turn[col] = turn[col].fillna("UNK").astype(str)
    return turn


def empty_donor_summary(test: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_id": test["episode_id"].to_numpy(),
            "donor_count": np.zeros(len(test), dtype=np.int16),
        }
    )


def build_donor_scores_fast(
    test: pd.DataFrame,
    donor_window_minutes: int,
    max_donors_per_episode: int,
    horizon: int,
    write_edges_path: Path | None = None,
) -> tuple[pd.DataFrame, int]:
    test = test.copy().reset_index(drop=True)
    stale_cols = [
        "donor_count",
        "donor_pred_mean",
        "donor_pred_max",
        "donor_actual_recover_mean",
        "donor_actual_recover_any",
        "donor_median_time_gap",
        "donor_median_available_turn",
        "ctrg_gap_mean",
        "ctrg_gap_max",
        "supported",
        "stressed",
        "severe_start_delay",
        "structural_brittle",
        "recoverable_despite_severe",
    ]
    test = test.drop(columns=[col for col in stale_cols if col in test.columns])

    n = len(test)
    donor_count = np.zeros(n, dtype=np.int16)
    donor_pred_mean = np.full(n, np.nan, dtype=float)
    donor_pred_max = np.full(n, np.nan, dtype=float)
    donor_actual_mean = np.full(n, np.nan, dtype=float)
    donor_actual_any = np.full(n, np.nan, dtype=float)
    donor_median_time_gap = np.full(n, np.nan, dtype=float)
    donor_median_available_turn = np.full(n, np.nan, dtype=float)

    window_ns = np.int64(donor_window_minutes) * np.int64(60_000_000_000)
    edge_count = 0
    edge_file = None
    if write_edges_path is not None:
        edge_file = write_edges_path.open("w", encoding="utf-8")
        edge_file.write(
            "episode_id,donor_episode_id,time_gap,donor_pred_recover,donor_actual_recover\n"
        )

    try:
        groups = list(test.groupby(["airport", "carrier"], sort=False).indices.items())
        for group_number, (key, idx_values) in enumerate(groups, start=1):
            idx = np.asarray(idx_values, dtype=np.int64)
            g = test.iloc[idx].sort_values("sched_dep_dt")
            orig_idx = g.index.to_numpy(dtype=np.int64)
            sched = g["sched_dep_dt"].to_numpy(dtype="datetime64[ns]").astype("int64")
            tail = g["tail"].astype(str).to_numpy()
            avail = pd.to_numeric(g["available_turn"], errors="coerce").to_numpy(dtype=float)
            cancel = pd.to_numeric(g["is_cancelled"], errors="coerce").fillna(0).to_numpy(dtype=float)
            divert = pd.to_numeric(g["is_diverted"], errors="coerce").fillna(0).to_numpy(dtype=float)
            dist = pd.to_numeric(g["distance_group"], errors="coerce").to_numpy(dtype=float)
            pred = pd.to_numeric(g["pred_recover"], errors="coerce").to_numpy(dtype=float)
            actual = pd.to_numeric(g[f"recover_h{horizon}"], errors="coerce").to_numpy(dtype=float)
            delay = pd.to_numeric(g["out_dep_delay"], errors="coerce").to_numpy(dtype=float)
            valid_base = (avail >= 35) & (cancel == 0) & (divert == 0) & np.isfinite(pred)
            stressed_positions = np.flatnonzero(delay >= 15)
            if len(stressed_positions) == 0:
                continue
            for pos in stressed_positions:
                lo = int(np.searchsorted(sched, sched[pos] - window_ns, side="left"))
                hi = int(np.searchsorted(sched, sched[pos] + window_ns, side="right"))
                if hi <= lo:
                    continue
                cand = np.arange(lo, hi, dtype=np.int64)
                mask = valid_base[cand] & (tail[cand] != tail[pos])
                if np.isfinite(dist[pos]):
                    mask &= (~np.isfinite(dist[cand])) | (np.abs(dist[cand] - dist[pos]) <= 2)
                cand = cand[mask]
                if cand.size == 0:
                    continue
                time_gap = np.abs(sched[cand] - sched[pos]).astype(float) / 60_000_000_000.0
                slack_gap = np.abs(avail[cand] - avail[pos])
                order = np.lexsort((slack_gap, time_gap))
                cand = cand[order[:max_donors_per_episode]]
                time_gap = time_gap[order[:max_donors_per_episode]]
                row_id = orig_idx[pos]
                donor_count[row_id] = int(cand.size)
                donor_pred_mean[row_id] = float(np.nanmean(pred[cand]))
                donor_pred_max[row_id] = float(np.nanmax(pred[cand]))
                donor_actual_mean[row_id] = float(np.nanmean(actual[cand]))
                donor_actual_any[row_id] = float(np.nanmax(actual[cand]))
                donor_median_time_gap[row_id] = float(np.nanmedian(time_gap))
                donor_median_available_turn[row_id] = float(np.nanmedian(avail[cand]))
                edge_count += int(cand.size)
                if edge_file is not None:
                    episode_id = str(g.iloc[pos]["episode_id"])
                    donor_ids = g.iloc[cand]["episode_id"].astype(str).to_numpy()
                    for donor_id, gap, donor_pred, donor_actual in zip(
                        donor_ids, time_gap, pred[cand], actual[cand]
                    ):
                        edge_file.write(
                            f"{episode_id},{donor_id},{gap:.6f},{donor_pred:.10f},{donor_actual:.0f}\n"
                        )
            if group_number % 50 == 0 or group_number == len(groups):
                supported_so_far = int((donor_count > 0).sum())
                log(
                    f"Donor groups {group_number}/{len(groups)}; "
                    f"supported stressed episodes so far: {supported_so_far:,}"
                )
    finally:
        if edge_file is not None:
            edge_file.close()

    summary = pd.DataFrame(
        {
            "episode_id": test["episode_id"].to_numpy(),
            "donor_count": donor_count,
            "donor_pred_mean": donor_pred_mean,
            "donor_pred_max": donor_pred_max,
            "donor_actual_recover_mean": donor_actual_mean,
            "donor_actual_recover_any": donor_actual_any,
            "donor_median_time_gap": donor_median_time_gap,
            "donor_median_available_turn": donor_median_available_turn,
        }
    )
    merged = test.merge(summary, on="episode_id", how="left")
    merged["donor_count"] = merged["donor_count"].fillna(0).astype(int)
    merged["ctrg_gap_mean"] = merged["donor_pred_mean"] - merged["pred_recover"]
    merged["ctrg_gap_max"] = merged["donor_pred_max"] - merged["pred_recover"]
    merged["supported"] = merged["donor_count"] > 0
    merged["stressed"] = merged["out_dep_delay"] >= 15
    merged["severe_start_delay"] = merged["out_dep_delay"] >= 60
    merged["structural_brittle"] = (
        merged["stressed"] & merged["supported"] & (merged["ctrg_gap_max"] <= 0)
    )
    merged["recoverable_despite_severe"] = (
        merged["severe_start_delay"]
        & merged["supported"]
        & (merged["donor_pred_max"] >= 0.70)
    )
    return merged, edge_count


def context_tables(test_graph: pd.DataFrame, horizon: int, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_path = root / "results" / "ctrg" / "summary_tables" / "airport_context_membership_2025.csv"
    if not context_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    context = pd.read_csv(context_path)[["airport", "context_cluster", "context_label"]]
    df = test_graph.merge(context, on="airport", how="left")
    rows = []
    for (cluster, label), g in df.groupby(["context_cluster", "context_label"], dropna=False):
        stressed = g[g["stressed"]]
        supported = stressed[stressed["supported"]]
        if len(stressed) == 0:
            continue
        rows.append(
            {
                "context_cluster": int(cluster) if pd.notna(cluster) else -1,
                "context_label": label,
                "airports": int(g["airport"].nunique()),
                "test_episodes": int(len(g)),
                "stressed_episodes": int(len(stressed)),
                "supported_stressed": int(len(supported)),
                "supported_share_stressed": float(len(supported) / len(stressed)),
                "failure_rate_supported_stressed": float(supported[f"fail_h{horizon}"].mean()) if len(supported) else np.nan,
                "mean_gap_max_supported": float(supported["ctrg_gap_max"].mean()) if len(supported) else np.nan,
                "severe_high_rewire_share": float(supported["recoverable_despite_severe"].mean()) if len(supported) else np.nan,
                "structural_brittle_share": float(supported["structural_brittle"].mean()) if len(supported) else np.nan,
                "median_donor_count_supported": float(supported["donor_count"].median()) if len(supported) else np.nan,
            }
        )
    context_summary = pd.DataFrame(rows).sort_values("context_cluster")

    top_rows = []
    for (cluster, label), g in df.groupby(["context_cluster", "context_label"], dropna=False):
        supported = g[(g["supported"]) & (g["stressed"])].copy()
        if supported.empty:
            continue
        base_fail = supported[f"fail_h{horizon}"].mean()
        for score, ranking, ascending in [
            ("ctrg_gap_max", "CTRG max-gap", False),
            ("out_dep_delay", "Delay-only", False),
            ("available_turn", "Slack-only", True),
            ("pred_recover", "Observed-path risk", True),
        ]:
            ranked = supported.sort_values(score, ascending=ascending)
            n = max(1, int(math.ceil(len(ranked) * 0.10)))
            top = ranked.head(n)
            top_rows.append(
                {
                    "context_cluster": int(cluster) if pd.notna(cluster) else -1,
                    "context_label": label,
                    "ranking": ranking,
                    "slice": "top_10pct",
                    "n": int(n),
                    "failure_rate": float(top[f"fail_h{horizon}"].mean()),
                    "failure_lift_vs_context": float(top[f"fail_h{horizon}"].mean() / base_fail) if base_fail > 0 else np.nan,
                    "mean_gap_max": float(top["ctrg_gap_max"].mean()),
                    "mean_start_delay": float(top["out_dep_delay"].mean()),
                    "mean_donor_pred_max": float(top["donor_pred_max"].mean()),
                }
            )
    context_top = pd.DataFrame(top_rows).sort_values(
        ["context_cluster", "ranking"]
    )
    return context_summary, context_top


def summarize(
    config: ExperimentConfig,
    turn: pd.DataFrame,
    train: pd.DataFrame,
    test_graph: pd.DataFrame,
    donor_edge_count: int,
    auc: float,
    ap: float,
) -> dict:
    horizon = config.horizon
    stressed = test_graph[test_graph["stressed"]]
    supported = stressed[stressed["supported"]]
    severe = supported[supported["severe_start_delay"]]
    moderate = supported[(supported["out_dep_delay"] >= 15) & (supported["out_dep_delay"] < 60)]
    return {
        "config": asdict(config),
        "counts": {
            "raw_turnarounds": int(len(turn)),
            "train_endpoint_episodes": int(len(train)),
            "test_endpoint_episodes": int(len(test_graph)),
            "test_stressed_episodes": int(len(stressed)),
            "supported_stressed_episodes": int(len(supported)),
            "donor_edges": int(donor_edge_count),
        },
        "model": {
            "recover_auc": float(auc),
            "recover_average_precision": float(ap),
        },
        "support": {
            "supported_share_among_stressed": float(len(supported) / len(stressed)) if len(stressed) else np.nan,
            "median_donor_count_supported": float(supported["donor_count"].median()) if len(supported) else np.nan,
            "median_donor_time_gap_minutes": float(supported["donor_median_time_gap"].median()) if len(supported) else np.nan,
        },
        "counterfactual_patterns": {
            "supported_stressed_failure_rate": float(supported[f"fail_h{horizon}"].mean()) if len(supported) else np.nan,
            "supported_stressed_recovery_rate": float(supported[f"recover_h{horizon}"].mean()) if len(supported) else np.nan,
            "mean_gap_max_supported": float(supported["ctrg_gap_max"].mean()) if len(supported) else np.nan,
            "severe_high_rewire_share": float(supported["recoverable_despite_severe"].mean()) if len(supported) else np.nan,
            "structural_brittle_share": float(supported["structural_brittle"].mean()) if len(supported) else np.nan,
            "severe_mean_donor_pred_max": float(severe["donor_pred_max"].mean()) if len(severe) else np.nan,
            "moderate_mean_donor_pred_max": float(moderate["donor_pred_max"].mean()) if len(moderate) else np.nan,
            "severe_count": int(len(severe)),
            "moderate_count": int(len(moderate)),
        },
    }


def run(args: argparse.Namespace) -> None:
    results_dir = ROOT / "results" / "ctrg" / args.mode
    results_dir.mkdir(parents=True, exist_ok=True)
    turn = load_turnarounds(Path(args.turnarounds), args.horizon)
    log(f"Loaded reconstructed turnarounds: {len(turn):,}")

    train, test = split_train_eval(turn, args.mode, args.horizon)
    if train.empty or test.empty:
        raise RuntimeError("Empty train or test split.")
    log(f"Endpoint split: train={len(train):,}, test={len(test):,}")

    linear_model = train_model(train, args.horizon, args.random_state)
    tree_model = train_tree_model(train, args.horizon, args.random_state)
    test = test.copy()
    test["pred_recover_linear"] = linear_model.predict_proba(
        test[MODEL_NUMERIC + MODEL_CATEGORICAL]
    )[:, 1]
    test["pred_recover_tree"] = predict_tree(tree_model, test)
    test["pred_recover"] = 0.5 * test["pred_recover_linear"] + 0.5 * test["pred_recover_tree"]
    y = test[f"recover_h{args.horizon}"].astype(int)
    auc = roc_auc_score(y, test["pred_recover"]) if y.nunique() > 1 else np.nan
    ap = average_precision_score(y, test["pred_recover"]) if y.nunique() > 1 else np.nan
    log(f"Observed-path recovery model: AUC={auc:.3f}, AP={ap:.3f}")

    edge_path = results_dir / "donor_edges.csv" if args.write_donor_edges else None
    test_graph, donor_edge_count = build_donor_scores_fast(
        test,
        donor_window_minutes=args.donor_window_minutes,
        max_donors_per_episode=args.max_donors_per_episode,
        horizon=args.horizon,
        write_edges_path=edge_path,
    )
    log(
        f"Donor graph built: supported stressed="
        f"{int(((test_graph['stressed']) & (test_graph['supported'])).sum()):,}, "
        f"edges={donor_edge_count:,}"
    )

    config = ExperimentConfig(
        mode=args.mode,
        years=args.years,
        months=args.months,
        airports=args.airports,
        horizon=args.horizon,
        min_turn=args.min_turn,
        max_turn=args.max_turn,
        donor_window_minutes=args.donor_window_minutes,
        max_donors_per_episode=args.max_donors_per_episode,
        random_state=args.random_state,
    )

    top_table = top_slice_table(test_graph, args.horizon)
    monthly = monthly_table(test_graph, args.horizon)
    airport = airport_table(test_graph, args.horizon)
    delay_band = delay_band_table(test_graph, args.horizon)
    cause = cause_table(test_graph, args.horizon)
    context_summary, context_top = context_tables(test_graph, args.horizon, ROOT)
    summary = summarize(config, turn, train, test_graph, donor_edge_count, auc, ap)

    keep_episode_cols = [
        "episode_id",
        "tail",
        "carrier",
        "airport",
        "dest",
        "flight_date",
        "sched_dep_dt",
        "month",
        "hour",
        "out_dep_delay",
        "available_turn",
        "distance_group",
        f"recover_h{args.horizon}",
        f"fail_h{args.horizon}",
        "pred_recover",
        "donor_count",
        "donor_pred_mean",
        "donor_pred_max",
        "donor_actual_recover_mean",
        "donor_actual_recover_any",
        "donor_median_time_gap",
        "donor_median_available_turn",
        "ctrg_gap_mean",
        "ctrg_gap_max",
        "supported",
        "stressed",
        "severe_start_delay",
        "structural_brittle",
        "recoverable_despite_severe",
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "late_aircraft_delay",
    ]
    test_graph[[c for c in keep_episode_cols if c in test_graph.columns]].to_csv(
        results_dir / "episode_scores.csv", index=False
    )
    top_table.to_csv(results_dir / "top_slice_table.csv", index=False)
    airport.to_csv(results_dir / "airport_table.csv", index=False)
    monthly.to_csv(results_dir / "monthly_table.csv", index=False)
    delay_band.to_csv(results_dir / "delay_band_table.csv", index=False)
    cause.to_csv(results_dir / "cause_table.csv", index=False)
    context_summary.to_csv(results_dir / "airport_context_result_table.csv", index=False)
    context_top.to_csv(results_dir / "airport_context_top_slice_table.csv", index=False)
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if args.bootstrap_reps > 0:
        boot = bootstrap_top_slice(
            test_graph,
            args.horizon,
            args.random_state,
            n_boot=args.bootstrap_reps,
        )
        boot.to_csv(results_dir / "top_slice_bootstrap.csv", index=False)
    else:
        pd.DataFrame().to_csv(results_dir / "top_slice_bootstrap.csv", index=False)
    log(json.dumps(summary, indent=2))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="smoke")
    parser.add_argument("--turnarounds", required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2025])
    parser.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--airports", nargs="+", default=[])
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--min-turn", type=int, default=35)
    parser.add_argument("--max-turn", type=int, default=720)
    parser.add_argument("--donor-window-minutes", type=int, default=120)
    parser.add_argument("--max-donors-per-episode", type=int, default=20)
    parser.add_argument("--bootstrap-reps", type=int, default=150)
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--write-donor-edges", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
