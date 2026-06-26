from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from resume_ctrg_from_turnarounds import build_donor_scores_fast, load_turnarounds
from run_ctrg_experiment import (
    MODEL_CATEGORICAL,
    MODEL_NUMERIC,
    predict_tree,
    split_train_eval,
    train_model,
    train_tree_model,
)


ROOT = Path(__file__).resolve().parents[2]


BASE_NUMERIC = [
    "in_arr_delay",
    "out_dep_delay",
    "available_turn",
    "planned_turn",
    "turn_slack",
    "taxi_out",
    "taxi_in",
    "distance_group",
    "same_tail_turn_index",
    "airport_hour_dep_rate",
    "airport_hour_mean_dep_delay",
    "airport_day_weather_delay_share",
    "airport_day_late_aircraft_share",
    "airport_day_cancel_share",
    "hour",
    "day_of_week",
]

BASE_CATEGORICAL = ["airport", "carrier", "dep_time_blk", "arr_time_blk"]

METHOD_NOTES = {
    "CTRG max-gap": {
        "family": "Counterfactual feasible-continuation graph",
        "problem_dimension": "Feasible rewire recoverability certificate",
    },
    "CTRG risk-gap certificate": {
        "family": "Counterfactual feasible-continuation graph",
        "problem_dimension": "Joint observed-path risk and feasible rewire recoverability",
    },
    "Observed-path ensemble": {
        "family": "Supervised sequence-outcome recovery prediction",
        "problem_dimension": "Pure failure-risk ranking along observed tail chains",
    },
    "ST propagation learner": {
        "family": "Spatiotemporal delay-propagation learning",
        "problem_dimension": "Airport, carrier, and route delay pressure",
    },
    "Multi-horizon hazard ensemble": {
        "family": "Finite-horizon hazard/recovery dynamics",
        "problem_dimension": "Recovery timing across short and longer horizons",
    },
    "Analog counterfactual matching": {
        "family": "Nearest-neighbor counterfactual support matching",
        "problem_dimension": "Recoverability from similar historical contexts",
    },
    "Network resilience learner": {
        "family": "Temporal airline-network resilience modeling",
        "problem_dimension": "Route diversity, concentration, and network centrality",
    },
    "Capacity-greedy recovery proxy": {
        "family": "Optimization and agent-based recovery proxy",
        "problem_dimension": "Local feasible donor supply under capacity pressure",
    },
    "Uncertainty-aware ensemble": {
        "family": "Uncertainty-aware decision support",
        "problem_dimension": "Risk ranking penalized by model disagreement",
    },
}

RECOMMENDED_ROLES = {
    "CTRG max-gap": "Proposed certificate metric",
    "CTRG risk-gap certificate": "Proposed integrated certificate sensitivity",
    "Observed-path ensemble": "Observed-path supervised recovery comparator",
    "ST propagation learner": "Propagation-learning comparator",
    "Multi-horizon hazard ensemble": "Dynamic recovery/hazard comparator",
    "Analog counterfactual matching": "Analog matching comparator",
    "Network resilience learner": "Airline-network resilience comparator",
    "Capacity-greedy recovery proxy": "Recovery optimization proxy",
    "Uncertainty-aware ensemble": "Uncertainty-aware learning comparator",
}


def log(message: str) -> None:
    print(message, flush=True)


def fit_hgb(
    train: pd.DataFrame,
    target: str,
    numeric: list[str],
    categorical: list[str],
    random_state: int,
):
    work = train[train[target].notna()].copy()
    y = work[target].astype(int)
    X_num = work[numeric].apply(pd.to_numeric, errors="coerce")
    X_cat = pd.get_dummies(work[categorical].fillna("UNK").astype(str), dummy_na=True)
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    med = X.median(numeric_only=True)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(med)
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=160,
        min_samples_leaf=35,
        l2_regularization=0.03,
        random_state=random_state,
    )
    clf.fit(X, y)
    return clf, X.columns.tolist(), med, numeric, categorical


def predict_hgb(model_tuple, df: pd.DataFrame) -> np.ndarray:
    clf, columns, med, numeric, categorical = model_tuple
    X_num = df[numeric].apply(pd.to_numeric, errors="coerce")
    X_cat = pd.get_dummies(df[categorical].fillna("UNK").astype(str), dummy_na=True)
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).reindex(columns=columns, fill_value=0)
    X = X.fillna(med)
    return clf.predict_proba(X)[:, 1]


def add_spatiotemporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour_bin"] = out["sched_dep_dt"].dt.floor("h")
    out["is_delayed15"] = (pd.to_numeric(out["out_dep_delay"], errors="coerce") >= 15).astype(float)
    keys = [
        ("airport", "airport"),
        ("carrier", "carrier"),
        ("airport_carrier", ["airport", "carrier"]),
        ("route", ["airport", "dest"]),
    ]
    for prefix, group_cols in keys:
        cols = group_cols if isinstance(group_cols, list) else [group_cols]
        stats = (
            out.groupby(cols + ["hour_bin"], dropna=False)
            .agg(
                dep_count=("episode_id", "size"),
                mean_delay=("out_dep_delay", "mean"),
                delayed_share=("is_delayed15", "mean"),
                mean_taxi_out=("taxi_out", "mean"),
            )
            .reset_index()
            .sort_values(cols + ["hour_bin"])
        )
        for col in ["dep_count", "mean_delay", "delayed_share", "mean_taxi_out"]:
            stats[f"{prefix}_lag1_{col}"] = stats.groupby(cols, dropna=False)[col].shift(1)
        stats = stats[cols + ["hour_bin"] + [c for c in stats.columns if c.startswith(f"{prefix}_lag1_")]]
        out = out.merge(stats, on=cols + ["hour_bin"], how="left")
    return out


def pagerank_from_routes(route_counts: pd.DataFrame, damping: float = 0.85, steps: int = 40) -> dict[str, float]:
    nodes = sorted(set(route_counts["airport"].astype(str)) | set(route_counts["dest"].astype(str)))
    if not nodes:
        return {}
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    out_links: dict[str, list[tuple[str, float]]] = {node: [] for node in nodes}
    for origin, g in route_counts.groupby("airport"):
        total = float(g["count"].sum())
        if total <= 0:
            continue
        out_links[str(origin)] = [(str(r.dest), float(r["count"]) / total) for _, r in g.iterrows()]
    pr = np.full(n, 1.0 / n)
    base = (1.0 - damping) / n
    for _ in range(steps):
        nxt = np.full(n, base)
        dangling = 0.0
        for node in nodes:
            j = idx[node]
            links = out_links.get(node, [])
            if not links:
                dangling += pr[j]
                continue
            for dest, weight in links:
                nxt[idx[dest]] += damping * pr[j] * weight
        nxt += damping * dangling / n
        pr = nxt
    return {node: float(pr[idx[node]]) for node in nodes}


def add_network_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = train.copy()
    test = test.copy()
    route_counts = (
        train.groupby(["airport", "dest"], dropna=False).size().rename("count").reset_index()
    )
    pr = pagerank_from_routes(route_counts)
    route_profile = (
        route_counts.groupby("airport")
        .agg(route_count=("dest", "nunique"), departures=("count", "sum"), max_route_count=("count", "max"))
        .reset_index()
    )
    entropy_rows = []
    for airport, g in route_counts.groupby("airport"):
        p = g["count"] / g["count"].sum()
        entropy = float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
        entropy_rows.append({"airport": airport, "route_entropy": entropy})
    route_profile = route_profile.merge(pd.DataFrame(entropy_rows), on="airport", how="left")
    route_profile["top_route_share"] = route_profile["max_route_count"] / route_profile["departures"]
    route_profile["airport_pagerank"] = route_profile["airport"].astype(str).map(pr).fillna(0.0)

    carrier_profile = (
        train.groupby(["airport", "carrier"], dropna=False)
        .agg(carrier_airport_departures=("episode_id", "size"), carrier_airport_tails=("tail", "nunique"))
        .reset_index()
    )
    carrier_profile["tail_density"] = (
        carrier_profile["carrier_airport_tails"] / carrier_profile["carrier_airport_departures"].clip(lower=1)
    )

    for frame in [train, test]:
        frame["airport"] = frame["airport"].astype(str)
        frame["carrier"] = frame["carrier"].astype(str)
    train = train.merge(route_profile, on="airport", how="left").merge(
        carrier_profile, on=["airport", "carrier"], how="left"
    )
    test = test.merge(route_profile, on="airport", how="left").merge(
        carrier_profile, on=["airport", "carrier"], how="left"
    )
    numeric = [
        "route_count",
        "route_entropy",
        "top_route_share",
        "airport_pagerank",
        "carrier_airport_departures",
        "carrier_airport_tails",
        "tail_density",
    ]
    return train, test, numeric


def analog_counterfactual_scores(train: pd.DataFrame, test: pd.DataFrame, k: int = 20) -> np.ndarray:
    scores = np.full(len(test), np.nan, dtype=float)
    features = [
        "hour",
        "out_dep_delay",
        "available_turn",
        "distance_group",
        "airport_hour_dep_rate",
        "airport_hour_mean_dep_delay",
        "airport_day_weather_delay_share",
        "airport_day_late_aircraft_share",
    ]
    for key, pool in train.groupby(["airport", "carrier"], dropna=False):
        query_index = test.index[(test["airport"] == key[0]) & (test["carrier"] == key[1])]
        if len(query_index) == 0:
            continue
        pool = pool[(pool["available_turn"] >= 35) & (pool["is_cancelled"] == 0) & (pool["is_diverted"] == 0)].copy()
        if len(pool) < 5:
            continue
        q = test.loc[query_index].copy()
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        X_pool = pipe.fit_transform(pool[features].apply(pd.to_numeric, errors="coerce"))
        X_q = pipe.transform(q[features].apply(pd.to_numeric, errors="coerce"))
        nn = NearestNeighbors(n_neighbors=min(k, len(pool)), metric="euclidean")
        nn.fit(X_pool)
        _, ind = nn.kneighbors(X_q)
        donor_pred = pool["pred_recover"].to_numpy(dtype=float)
        donor_max = np.nanmax(donor_pred[ind], axis=1)
        scores[query_index.to_numpy()] = donor_max - q["pred_recover"].to_numpy(dtype=float)
    return scores


def capacity_greedy_score(df: pd.DataFrame) -> np.ndarray:
    p_fail = 1.0 - df["pred_recover"].to_numpy(dtype=float)
    donor_count = df["donor_count"].fillna(0).to_numpy(dtype=float)
    time_gap = df["donor_median_time_gap"].fillna(df["donor_median_time_gap"].median()).to_numpy(dtype=float)
    donor_turn = df["donor_median_available_turn"].fillna(df["donor_median_available_turn"].median()).to_numpy(dtype=float)
    delay = df["out_dep_delay"].fillna(df["out_dep_delay"].median()).to_numpy(dtype=float)
    count_penalty = 1.0 / np.sqrt(np.maximum(donor_count, 1.0))
    time_pressure = time_gap / np.nanpercentile(time_gap, 95)
    turn_relief = donor_turn / np.nanpercentile(donor_turn, 95)
    delay_pressure = delay / np.nanpercentile(delay, 95)
    return 0.55 * p_fail + 0.20 * count_penalty + 0.15 * time_pressure + 0.10 * delay_pressure - 0.10 * turn_relief


def uncertainty_scores(train: pd.DataFrame, test: pd.DataFrame, random_state: int) -> np.ndarray:
    numeric = BASE_NUMERIC
    categorical = BASE_CATEGORICAL
    ycol = "fail_h4"
    preds = []
    rng = np.random.default_rng(random_state)
    train = train[train[ycol].notna()].reset_index(drop=True)
    for i in range(5):
        sample_idx = rng.choice(len(train), size=int(0.75 * len(train)), replace=True)
        sample = train.iloc[sample_idx].copy()
        model = fit_hgb(sample, ycol, numeric, categorical, random_state + i + 11)
        preds.append(predict_hgb(model, test))
    pred = np.vstack(preds)
    return pred.mean(axis=0) + 0.5 * pred.std(axis=0)


def evaluate_scores(df: pd.DataFrame, methods: dict[str, np.ndarray], horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fail_col = f"fail_h{horizon}"
    y = df[fail_col].astype(int).to_numpy()
    summary_rows = []
    top_rows = []
    base_failure = float(np.mean(y))
    for name, raw_score in methods.items():
        score = np.asarray(raw_score, dtype=float)
        valid = np.isfinite(score)
        eval_y = y[valid]
        eval_score = score[valid]
        if len(eval_y) == 0 or len(np.unique(eval_y)) < 2:
            auc = np.nan
            ap = np.nan
        else:
            auc = roc_auc_score(eval_y, eval_score)
            ap = average_precision_score(eval_y, eval_score)
        summary_rows.append(
            {
                "method": name,
                "family": METHOD_NOTES[name]["family"],
                "problem_dimension": METHOD_NOTES[name]["problem_dimension"],
                "evaluated_episodes": int(valid.sum()),
                "missing_share": float(1.0 - valid.mean()),
                "failure_auc": float(auc),
                "failure_average_precision": float(ap),
                "base_failure_rate": base_failure,
            }
        )
        order = np.argsort(-eval_score)
        for frac in (0.05, 0.10, 0.20):
            n = max(1, int(math.ceil(len(order) * frac)))
            picked_index = order[:n]
            picked = eval_y[picked_index]
            picked_df = df.iloc[picked_index]
            failure_rate = float(np.mean(picked))
            top_rows.append(
                {
                    "method": name,
                    "slice": f"top_{int(frac * 100)}pct",
                    "n": int(n),
                    "failure_rate": failure_rate,
                    "failure_lift_vs_supported": float(failure_rate / base_failure) if base_failure > 0 else np.nan,
                    "mean_start_delay": float(picked_df["out_dep_delay"].mean()) if "out_dep_delay" in picked_df else np.nan,
                    "mean_ctrg_gap_max": float(picked_df["ctrg_gap_max"].mean()) if "ctrg_gap_max" in picked_df else np.nan,
                    "mean_donor_pred_max": float(picked_df["donor_pred_max"].mean()) if "donor_pred_max" in picked_df else np.nan,
                    "donor_actual_recover_mean": float(picked_df["donor_actual_recover_mean"].mean())
                    if "donor_actual_recover_mean" in picked_df
                    else np.nan,
                    "severe_high_rewire_share": float(picked_df["recoverable_despite_severe"].mean())
                    if "recoverable_despite_severe" in picked_df
                    else np.nan,
                    "structural_brittle_share": float(picked_df["structural_brittle"].mean())
                    if "structural_brittle" in picked_df
                    else np.nan,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(top_rows)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turnarounds", required=True)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--donor-window-minutes", type=int, default=120)
    parser.add_argument("--max-donors-per-episode", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "ctrg" / "comparison_methods_diagnostic"))
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    turn = load_turnarounds(Path(args.turnarounds), args.horizon)
    log(f"Loaded turnarounds: {len(turn):,}")
    train, test = split_train_eval(turn, "smoke", args.horizon)
    log(f"Endpoint split: train={len(train):,}, test={len(test):,}")

    linear_model = train_model(train, args.horizon, args.random_state)
    tree_model = train_tree_model(train, args.horizon, args.random_state)
    for frame in [train, test]:
        frame["pred_recover_linear"] = linear_model.predict_proba(frame[MODEL_NUMERIC + MODEL_CATEGORICAL])[:, 1]
        frame["pred_recover_tree"] = predict_tree(tree_model, frame)
        frame["pred_recover"] = 0.5 * frame["pred_recover_linear"] + 0.5 * frame["pred_recover_tree"]
        frame[f"fail_h{args.horizon}"] = pd.to_numeric(frame[f"fail_h{args.horizon}"], errors="coerce")

    test_graph, edge_count = build_donor_scores_fast(
        test,
        donor_window_minutes=args.donor_window_minutes,
        max_donors_per_episode=args.max_donors_per_episode,
        horizon=args.horizon,
    )
    eval_df = test_graph[(test_graph["stressed"]) & (test_graph["supported"])].copy().reset_index(drop=True)
    log(f"Evaluation episodes: {len(eval_df):,}; donor edges: {edge_count:,}")

    st_all = add_spatiotemporal_features(pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True))
    st_train = st_all[st_all["split"] == "train"].copy()
    st_test = st_all[st_all["split"] == "test"].copy()
    st_features = BASE_NUMERIC + [c for c in st_all.columns if "_lag1_" in c]
    st_model = fit_hgb(st_train, f"fail_h{args.horizon}", st_features, BASE_CATEGORICAL, args.random_state + 101)
    st_score_all = predict_hgb(st_model, st_test)

    hazard_scores = []
    for h, weight in [(2, 0.45), (4, 0.35), (6, 0.20)]:
        target = f"recover_h{h}"
        if target not in train.columns:
            continue
        h_model = fit_hgb(train[train[f"endpoint_obs_h{h}"] == 1], target, BASE_NUMERIC, BASE_CATEGORICAL, args.random_state + h)
        hazard_scores.append(weight * predict_hgb(h_model, test))
    mh_score_all = 1.0 - np.sum(hazard_scores, axis=0)

    train_net, test_net, net_features = add_network_features(train, test)
    net_model = fit_hgb(
        train_net,
        f"fail_h{args.horizon}",
        BASE_NUMERIC + net_features,
        BASE_CATEGORICAL,
        args.random_state + 202,
    )
    net_score_all = predict_hgb(net_model, test_net)

    analog_score_all = analog_counterfactual_scores(train, test)
    uncertainty_score_all = uncertainty_scores(train, test, args.random_state + 303)

    score_frame = test_graph[
        [
            "episode_id",
            "out_dep_delay",
            "donor_pred_max",
            "donor_actual_recover_mean",
            "ctrg_gap_max",
            "recoverable_despite_severe",
            "structural_brittle",
        ]
    ].copy()
    score_frame["CTRG max-gap"] = test_graph["ctrg_gap_max"].to_numpy(dtype=float)
    score_frame["CTRG risk-gap certificate"] = (
        (1.0 - test_graph["pred_recover"].to_numpy(dtype=float))
        + test_graph["ctrg_gap_max"].fillna(0).to_numpy(dtype=float)
    )
    score_frame["Observed-path ensemble"] = 1.0 - test_graph["pred_recover"].to_numpy(dtype=float)
    score_frame["ST propagation learner"] = st_score_all
    score_frame["Multi-horizon hazard ensemble"] = mh_score_all
    score_frame["Analog counterfactual matching"] = analog_score_all
    score_frame["Network resilience learner"] = net_score_all
    score_frame["Capacity-greedy recovery proxy"] = capacity_greedy_score(test_graph)
    score_frame["Uncertainty-aware ensemble"] = uncertainty_score_all

    eval_scores = eval_df[["episode_id", f"fail_h{args.horizon}"]].merge(score_frame, on="episode_id", how="left")
    methods = {
        name: eval_scores[name].to_numpy(dtype=float)
        for name in ["CTRG max-gap", "CTRG risk-gap certificate", *METHOD_NOTES.keys()]
        if name in eval_scores.columns
    }
    summary, top = evaluate_scores(eval_scores, methods, args.horizon)
    top10 = top[top["slice"] == "top_10pct"].copy()
    condensed = summary.merge(
        top10[
            [
                "method",
                "failure_rate",
                "failure_lift_vs_supported",
                "mean_ctrg_gap_max",
                "mean_donor_pred_max",
                "severe_high_rewire_share",
                "structural_brittle_share",
            ]
        ],
        on="method",
        how="left",
    )
    condensed.insert(
        1,
        "recommended_role",
        condensed["method"].map(RECOMMENDED_ROLES).fillna("Comparison method"),
    )

    summary.to_csv(out_dir / "comparison_method_diagnostic_summary.csv", index=False)
    top.to_csv(out_dir / "comparison_method_diagnostic_top_slices.csv", index=False)
    eval_scores.to_csv(out_dir / "comparison_method_diagnostic_episode_scores.csv", index=False)
    condensed.to_csv(out_dir / "comparison_method_table.csv", index=False)
    (out_dir / "comparison_method_notes.json").write_text(
        json.dumps(METHOD_NOTES, indent=2), encoding="utf-8"
    )
    log(summary.round(4).to_string(index=False))
    log(top[top["slice"] == "top_10pct"].round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
