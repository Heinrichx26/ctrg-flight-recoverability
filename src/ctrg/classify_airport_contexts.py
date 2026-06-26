from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from run_ctrg_experiment import download_month


USECOLS = [
    "Year",
    "Month",
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepDelayMinutes",
    "DepDel15",
    "ArrDelayMinutes",
    "ArrDel15",
    "Cancelled",
    "Diverted",
    "TaxiOut",
    "Distance",
    "DistanceGroup",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]

NUMERIC_COLS = [
    "DepDelayMinutes",
    "DepDel15",
    "ArrDelayMinutes",
    "ArrDel15",
    "Cancelled",
    "Diverted",
    "TaxiOut",
    "Distance",
    "DistanceGroup",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]

LCC_CARRIERS = {"WN", "NK", "F9", "G4", "B6", "SY", "MX", "XP"}


def read_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV in {path}")
        with zf.open(csv_names[0]) as fh:
            header = pd.read_csv(fh, nrows=0)
        usecols = [c for c in USECOLS if c in set(header.columns)]
        with zf.open(csv_names[0]) as fh:
            df = pd.read_csv(fh, usecols=usecols, low_memory=False)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def entropy_share(s: pd.Series) -> float:
    p = s.value_counts(normalize=True)
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def top_share(s: pd.Series) -> float:
    p = s.value_counts(normalize=True)
    if p.empty:
        return np.nan
    return float(p.iloc[0])


def summarize_airports(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLS:
        if col not in df:
            df[col] = np.nan
    df["is_lcc"] = df["Reporting_Airline"].isin(LCC_CARRIERS).astype(float)
    df["is_delayed"] = (df["DepDelayMinutes"].fillna(0) >= 15).astype(float)
    df["is_longhaul"] = (df["Distance"].fillna(0) >= 1500).astype(float)
    df["weather_delay_flag"] = (df["WeatherDelay"].fillna(0) > 0).astype(float)
    df["late_aircraft_flag"] = (df["LateAircraftDelay"].fillna(0) > 0).astype(float)
    df["nas_delay_flag"] = (df["NASDelay"].fillna(0) > 0).astype(float)
    df["date"] = pd.to_datetime(df["FlightDate"], errors="coerce")
    daily = df.groupby(["Origin", "date"]).size().rename("daily_departures").reset_index()
    daily_stats = daily.groupby("Origin")["daily_departures"].agg(["mean", "std"]).rename(
        columns={"mean": "mean_daily_departures", "std": "sd_daily_departures"}
    )
    base = df.groupby("Origin").agg(
        departures=("Origin", "size"),
        carriers=("Reporting_Airline", "nunique"),
        tails=("Tail_Number", "nunique"),
        routes=("Dest", "nunique"),
        dep_delay15_rate=("is_delayed", "mean"),
        mean_dep_delay=("DepDelayMinutes", "mean"),
        arr_delay15_rate=("ArrDel15", "mean"),
        cancel_rate=("Cancelled", "mean"),
        divert_rate=("Diverted", "mean"),
        mean_taxi_out=("TaxiOut", "mean"),
        mean_distance=("Distance", "mean"),
        longhaul_share=("is_longhaul", "mean"),
        lcc_share=("is_lcc", "mean"),
        weather_delay_share=("weather_delay_flag", "mean"),
        late_aircraft_share=("late_aircraft_flag", "mean"),
        nas_delay_share=("nas_delay_flag", "mean"),
        carrier_top_share=("Reporting_Airline", top_share),
        carrier_entropy=("Reporting_Airline", entropy_share),
    )
    base = base.join(daily_stats)
    base["daily_cv"] = base["sd_daily_departures"] / base["mean_daily_departures"]
    base = base.replace([np.inf, -np.inf], np.nan)
    return base.reset_index().rename(columns={"Origin": "airport"})


def select_eligible(summary: pd.DataFrame, min_departures: int, min_routes: int, min_carriers: int) -> pd.DataFrame:
    return summary[
        (summary["departures"] >= min_departures)
        & (summary["routes"] >= min_routes)
        & (summary["carriers"] >= min_carriers)
    ].copy()


def cluster_airports(eligible: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [
        "departures",
        "carriers",
        "routes",
        "dep_delay15_rate",
        "cancel_rate",
        "mean_taxi_out",
        "mean_distance",
        "longhaul_share",
        "lcc_share",
        "weather_delay_share",
        "late_aircraft_share",
        "nas_delay_share",
        "carrier_top_share",
        "carrier_entropy",
        "daily_cv",
    ]
    X = eligible[features].copy()
    X = X.fillna(X.median(numeric_only=True))
    Z = StandardScaler().fit_transform(np.log1p(X.where(X >= 0, X)))
    choices = []
    max_k = min(8, len(eligible) - 1)
    for k in range(3, max_k + 1):
        km = KMeans(n_clusters=k, n_init=30, random_state=random_state)
        labels = km.fit_predict(Z)
        score = silhouette_score(Z, labels)
        choices.append({"k": k, "silhouette": float(score), "inertia": float(km.inertia_)})
    choice_df = pd.DataFrame(choices)
    best_k = int(choice_df.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    km = KMeans(n_clusters=best_k, n_init=50, random_state=random_state)
    labels = km.fit_predict(Z)
    out = eligible.copy()
    out["context_cluster"] = labels
    profiles = out.groupby("context_cluster")[features + ["airport"]].agg(
        {
            **{f: "median" for f in features},
            "airport": "count",
        }
    ).rename(columns={"airport": "airport_count"})
    profiles = profiles.reset_index()
    return out, choice_df


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--min-departures", type=int, default=12000)
    parser.add_argument("--min-routes", type=int, default=20)
    parser.add_argument("--min-carriers", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=2026)
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "ctrg" / "raw_bts"
    results_dir = root / "results" / "ctrg" / "airport_contexts"
    results_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for m in range(1, 13):
        path = raw_dir / f"bts_otp_{args.year}_{m:02d}.zip"
        if not path.exists():
            path = download_month(args.year, m, raw_dir)
        frames.append(read_zip(path))
        print(f"Loaded {path.name}", flush=True)
    df = pd.concat(frames, ignore_index=True)
    summary = summarize_airports(df)
    eligible = select_eligible(summary, args.min_departures, args.min_routes, args.min_carriers)
    clustered, choice = cluster_airports(eligible, args.random_state)

    summary.to_csv(results_dir / f"airport_universe_{args.year}.csv", index=False)
    eligible.to_csv(results_dir / f"airport_eligible_{args.year}.csv", index=False)
    clustered.to_csv(results_dir / f"airport_context_clusters_{args.year}.csv", index=False)
    choice.to_csv(results_dir / f"cluster_selection_{args.year}.csv", index=False)
    meta = {
        "year": args.year,
        "min_departures": args.min_departures,
        "min_routes": args.min_routes,
        "min_carriers": args.min_carriers,
        "airport_universe_count": int(len(summary)),
        "eligible_airport_count": int(len(eligible)),
        "selected_clusters": int(clustered["context_cluster"].nunique()),
        "eligible_airports": clustered.sort_values("departures", ascending=False)["airport"].tolist(),
    }
    (results_dir / f"airport_context_meta_{args.year}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
