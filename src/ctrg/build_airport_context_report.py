from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CONTEXT_DIR = ROOT / "results" / "ctrg" / "airport_contexts"
OUT_DIR = ROOT / "results" / "ctrg" / "summary_tables"

CONTEXT_LABELS = {
    0: "Large route-diverse congested gateways",
    1: "Multi-carrier regional markets",
    2: "Carrier-dominant point-to-point airports",
    3: "Long-stage geographic gateway airports",
}

PROFILE_FEATURES = [
    "departures",
    "routes",
    "carriers",
    "tails",
    "dep_delay15_rate",
    "cancel_rate",
    "mean_taxi_out",
    "mean_distance",
    "longhaul_share",
    "lcc_share",
    "carrier_top_share",
    "carrier_entropy",
    "weather_delay_share",
    "late_aircraft_share",
    "nas_delay_share",
    "daily_cv",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(CONTEXT_DIR / "airport_universe_2025.csv")
    eligible = pd.read_csv(CONTEXT_DIR / "airport_eligible_2025.csv")
    clustered = pd.read_csv(CONTEXT_DIR / "airport_context_clusters_2025.csv")

    clustered["context_label"] = clustered["context_cluster"].map(CONTEXT_LABELS)
    clustered["context_label"] = clustered["context_label"].fillna(
        "Data-derived airport context"
    )

    eligible_set = set(eligible["airport"])
    universe = universe.copy()
    universe["estimability_status"] = np.where(
        universe["airport"].isin(eligible_set),
        "Estimable population",
        "Sparse-support population",
    )
    universe = universe.merge(
        clustered[["airport", "context_cluster", "context_label"]],
        on="airport",
        how="left",
    )
    universe["mutually_exclusive_context"] = universe["context_label"].fillna(
        "Sparse-support airport outside main recoverability estimation"
    )

    total_departures = float(universe["departures"].sum())
    eligible_departures = float(eligible["departures"].sum())
    sparse = universe[universe["estimability_status"] == "Sparse-support population"]

    sampling_frame = pd.DataFrame(
        [
            {
                "year": 2025,
                "bts_departure_airports": int(len(universe)),
                "estimable_airports": int(len(eligible)),
                "sparse_support_airports": int(len(sparse)),
                "estimable_departures": int(eligible_departures),
                "total_departures": int(total_departures),
                "estimable_departure_share_pct": float(
                    100.0 * eligible_departures / total_departures
                ),
                "median_departures_sparse_airports": float(
                    sparse["departures"].median()
                ),
                "minimum_departures_rule": 12000,
                "minimum_routes_rule": 20,
                "minimum_carriers_rule": 4,
                "selected_context_clusters": int(
                    clustered["context_cluster"].nunique()
                ),
            }
        ]
    )

    profiles = (
        clustered.groupby(["context_cluster", "context_label"], as_index=False)
        .agg(
            airports=("airport", "count"),
            departures=("departures", "sum"),
            median_departures=("departures", "median"),
            median_routes=("routes", "median"),
            median_carriers=("carriers", "median"),
            median_tails=("tails", "median"),
            median_dep_delay15_rate=("dep_delay15_rate", "median"),
            median_cancel_rate=("cancel_rate", "median"),
            median_taxi_out=("mean_taxi_out", "median"),
            median_distance=("mean_distance", "median"),
            median_longhaul_share=("longhaul_share", "median"),
            median_lcc_share=("lcc_share", "median"),
            median_carrier_top_share=("carrier_top_share", "median"),
            median_carrier_entropy=("carrier_entropy", "median"),
            median_weather_delay_share=("weather_delay_share", "median"),
            median_late_aircraft_share=("late_aircraft_share", "median"),
            median_nas_delay_share=("nas_delay_share", "median"),
            median_daily_cv=("daily_cv", "median"),
        )
        .sort_values("context_cluster")
    )
    profiles["departure_share_of_estimable_pct"] = (
        100.0 * profiles["departures"] / profiles["departures"].sum()
    )

    members = clustered[
        ["airport", "context_cluster", "context_label", *PROFILE_FEATURES]
    ].sort_values(["context_cluster", "departures"], ascending=[True, False])

    universe_out = universe[
        [
            "airport",
            "estimability_status",
            "mutually_exclusive_context",
            "departures",
            "routes",
            "carriers",
            "dep_delay15_rate",
            "mean_distance",
            "lcc_share",
            "carrier_top_share",
            "carrier_entropy",
        ]
    ].sort_values(["estimability_status", "departures"], ascending=[True, False])

    sampling_frame.to_csv(OUT_DIR / "airport_sampling_frame_2025.csv", index=False)
    profiles.to_csv(OUT_DIR / "airport_context_profiles_2025.csv", index=False)
    members.to_csv(OUT_DIR / "airport_context_membership_2025.csv", index=False)
    universe_out.to_csv(
        OUT_DIR / "airport_universe_mutually_exclusive_contexts_2025.csv",
        index=False,
    )

    method_text = (
        "The 2025 airport sampling frame was defined from all departure airports "
        "appearing in the public Bureau of Transportation Statistics on-time "
        "performance records. Airports entered the estimable population when they "
        "had at least 12,000 departures, 20 nonstop destination markets, and four "
        "reporting carriers during the study year. These rules require continuous "
        "daily operating support, route diversity, and carrier diversity before "
        "airport-level recoverability patterns are estimated. The resulting "
        f"{len(eligible)} airports account for {eligible_departures / total_departures:.1%} "
        "of all departures in the public sampling frame. Airports below these "
        "support conditions are retained in the sampling-frame audit and are "
        "reported as sparse-support airports.\n\n"
        "The estimable airports were then assigned to mutually exclusive operating "
        "contexts using standardized airport-level descriptors of volume, route "
        "breadth, carrier diversity, delay exposure, taxi-out time, stage length, "
        "low-cost-carrier share, delay-cause composition, and daily demand "
        "variability. The number of contexts was selected by silhouette score over "
        "candidate k-means partitions. Four contexts were selected. Each estimable "
        "airport belongs to exactly one context, and every airport in the original "
        "sampling frame is represented either by one of these contexts or by the "
        "sparse-support audit category."
    )
    (OUT_DIR / "airport_context_methods_text_2025.md").write_text(
        method_text, encoding="utf-8"
    )

    manifest = {
        "outputs": [
            "airport_sampling_frame_2025.csv",
            "airport_context_profiles_2025.csv",
            "airport_context_membership_2025.csv",
            "airport_universe_mutually_exclusive_contexts_2025.csv",
            "airport_context_methods_text_2025.md",
        ],
        "context_labels": CONTEXT_LABELS,
    }
    (OUT_DIR / "airport_context_report_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
