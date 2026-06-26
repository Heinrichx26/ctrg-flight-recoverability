from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "ctrg"
FULL = RESULTS / "full"
SUMMARY = RESULTS / "summary_tables"
DIAGNOSTIC = RESULTS / "comparison_methods_diagnostic"
TABLE_DIR = ROOT / "results" / "display" / "tables"
FIG_DIR = ROOT / "results" / "display" / "figures"


def tex_int(value: float | int) -> str:
    return f"{int(round(float(value))):,}".replace(",", "{,}")


def tex_pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}"


def tex_num(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def bold(text: str) -> str:
    return rf"\textbf{{{text}}}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "sampling": pd.read_csv(SUMMARY / "airport_sampling_frame_2025.csv"),
        "profiles": pd.read_csv(SUMMARY / "airport_context_profiles_2025.csv"),
        "yearly": pd.read_csv(SUMMARY / "yearly_robustness_summary.csv"),
        "context": pd.read_csv(FULL / "airport_context_result_table.csv"),
        "context_top": pd.read_csv(FULL / "airport_context_top_slice_table.csv"),
        "top": pd.read_csv(FULL / "top_slice_table.csv"),
        "boot": pd.read_csv(FULL / "top_slice_bootstrap.csv"),
        "delay": pd.read_csv(FULL / "delay_band_table.csv"),
        "cause": pd.read_csv(FULL / "cause_table.csv"),
        "monthly": pd.read_csv(FULL / "monthly_table.csv"),
        "yearly_context": pd.read_csv(SUMMARY / "yearly_context_robustness_summary.csv"),
        "diagnostic_slices": pd.read_csv(DIAGNOSTIC / "comparison_method_diagnostic_top_slices.csv"),
        "diagnostic": pd.read_csv(DIAGNOSTIC / "comparison_method_table.csv"),
    }


def table_sampling_contexts(data: dict[str, pd.DataFrame]) -> None:
    sampling = data["sampling"].iloc[0]
    profiles = data["profiles"].sort_values("context_cluster")
    sparse_departures = int(sampling.total_departures - sampling.estimable_departures)
    sparse_share = 100.0 - float(sampling.estimable_departure_share_pct)
    rows = [
        [
            "Public departure-airport frame",
            tex_int(sampling.bts_departure_airports),
            tex_int(sampling.total_departures),
            r"100.00\% public",
            "All origin airports observed in the yearly BTS record",
        ],
        [
            bold("Estimable population"),
            bold(tex_int(sampling.estimable_airports)),
            bold(tex_int(sampling.estimable_departures)),
            bold(rf"{float(sampling.estimable_departure_share_pct):.2f}\% public"),
            bold("At least 12,000 departures, 20 destination markets, and 4 reporting carriers"),
        ],
        [
            "Sparse-support audit population",
            tex_int(sampling.sparse_support_airports),
            tex_int(sparse_departures),
            rf"{sparse_share:.2f}\% public",
            f"Outside at least one estimability rule; median annual departures {float(sampling.median_departures_sparse_airports):,.1f}".replace(",", "{,}"),
        ],
    ]
    for _, row in profiles.iterrows():
        rows.append(
            [
                row.context_label,
                tex_int(row.airports),
                tex_int(row.departures),
                rf"{float(row.departure_share_of_estimable_pct):.2f}\% estimable",
                f"Median routes {float(row.median_routes):.1f}; median carriers {float(row.median_carriers):.1f}; median departure-delay-15 rate {100.0 * float(row.median_dep_delay15_rate):.1f}\\%",
            ]
        )

    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    text = rf"""\begin{{table}}[!tbp]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{1.02}}
\caption{{Public sampling frame and mutually exclusive airport contexts}}
\label{{tab:public-sampling-contexts}}
\begin{{tabularx}}{{\linewidth}}{{@{{}}L{{0.27\linewidth}}r r L{{0.14\linewidth}}Y@{{}}}}
\toprule
Layer or context & Airports & Departures & Departure share & Rule or profile \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\tablenote{{Bold entries mark the retained estimable population. Context rows partition the estimable population, so their departure shares use the estimable population as the denominator.}}
\end{{table}}
"""
    write_text(TABLE_DIR / "tab_public_sampling_contexts.tex", text)


def table_experimental_scale(data: dict[str, pd.DataFrame]) -> None:
    yearly = data["yearly"].copy()
    rows = []
    for _, row in yearly.iterrows():
        rows.append(
            [
                row.analysis,
                tex_int(row.turnarounds),
                tex_int(row.test_episodes),
                f"{tex_int(row.stressed_episodes)} / {tex_int(row.supported_stressed)}",
                tex_pct(row.support_share),
                tex_int(row.donor_edges),
                f"{tex_num(row.recover_auc, 3)} / {tex_num(row.recover_ap, 3)}",
                tex_num(row.mean_gap_max, 3),
            ]
        )
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    text = rf"""\begin{{table}}[!tbp]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{1.02}}
\caption{{Experimental scale and graph support}}
\label{{tab:experimental-scale}}
\begin{{tabularx}}{{\linewidth}}{{@{{}}L{{0.16\linewidth}}C C C C C C C@{{}}}}
\toprule
Analysis & Turnarounds & Test episodes & Stressed / supported & Support (\%) & Donor edges & AUC / AP & Mean gap \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\tablenote{{AUC denotes area under the receiver operating characteristic curve, and AP denotes average precision for the observed-path recovery scorer. Higher support means broader certified donor coverage; larger mean gap means greater feasible-rewire recoverability space among supported stressed episodes.}}
\end{{table}}
"""
    write_text(TABLE_DIR / "tab_experimental_scale.tex", text)


def table_top_slice_comparison(data: dict[str, pd.DataFrame]) -> None:
    top = data["top"]
    boot = data["boot"]
    name_map = {
        "CTRG max-gap": "CTRG max-gap",
        "Observed-path recoverability": "Observed-path risk",
        "Delay-only": "Delay-only",
        "Slack-only": "Slack-only",
    }
    rows_raw = []
    for source_name, label in name_map.items():
        tr = top[(top.ranking == source_name) & (top.slice == "top_10pct")].iloc[0]
        br_name = "Observed-path risk" if source_name == "Observed-path recoverability" else source_name
        br = boot[(boot.scope == "all_contexts") & (boot.ranking == br_name) & (boot.slice == "top_10pct")].iloc[0]
        rows_raw.append(
            {
                "label": label,
                "failure": float(tr.failure_rate),
                "ci_low": float(br.failure_rate_ci_low),
                "ci_high": float(br.failure_rate_ci_high),
                "lift": float(tr.failure_lift_vs_supported),
                "gap": float(tr.mean_gap_max),
                "rewire": float(tr.mean_donor_pred_max),
            }
        )
    max_failure = max(r["failure"] for r in rows_raw)
    max_gap = max(r["gap"] for r in rows_raw)
    max_rewire = max(r["rewire"] for r in rows_raw)
    rows = []
    for r in rows_raw:
        failure_text = f"{tex_pct(r['failure'])} [{tex_pct(r['ci_low'])}, {tex_pct(r['ci_high'])}]"
        if r["failure"] == max_failure:
            failure_text = bold(failure_text)
        gap_text = tex_num(r["gap"], 3)
        if r["gap"] == max_gap:
            gap_text = bold(gap_text)
        rewire_text = tex_num(r["rewire"], 3)
        if r["rewire"] == max_rewire:
            rewire_text = bold(rewire_text)
        rows.append([r["label"], failure_text, tex_num(r["lift"], 2), gap_text, rewire_text])
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    text = rf"""\begin{{table}}[!tbp]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.04}}
\caption{{Top-10\% ranking comparison in the 2025 full experiment}}
\label{{tab:top-slice-comparison}}
\begin{{tabularx}}{{\linewidth}}{{@{{}}L{{0.26\linewidth}}r r r r@{{}}}}
\toprule
Ranking rule & Failed-exit rate (\%, 95\% CI) & Lift & Mean gap & Mean feasible-rewire score \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\tablenote{{Higher failed-exit rate means stronger enrichment of realized brittle outcomes in the selected slice. Lift is the failed-exit rate divided by the reference failed-exit rate among all supported stressed episodes. Higher mean gap and feasible-rewire score mean stronger CTRG recoverability evidence. Bold marks the largest value in each outcome column. CI denotes confidence interval from airport-block bootstrap resampling.}}
\end{{table}}
"""
    write_text(TABLE_DIR / "tab_top_slice_comparison.tex", text)


def table_context_heterogeneity(data: dict[str, pd.DataFrame]) -> None:
    context = data["context"]
    top = data["context_top"]
    rows = []
    short = {
        "Large route-diverse congested gateways": "Large route-diverse gateways",
        "Multi-carrier regional markets": "Multi-carrier regional markets",
        "Carrier-dominant point-to-point airports": "Carrier-dominant point-to-point",
        "Long-stage geographic gateway airports": "Long-stage geographic gateways",
    }
    for _, crow in context.sort_values("context_cluster").iterrows():
        subset = top[(top.context_cluster == crow.context_cluster) & (top.slice == "top_10pct")]
        vals = {
            "CTRG": float(subset[subset.ranking == "CTRG max-gap"].failure_rate.iloc[0]),
            "Delay": float(subset[subset.ranking == "Delay-only"].failure_rate.iloc[0]),
            "Slack": float(subset[subset.ranking == "Slack-only"].failure_rate.iloc[0]),
        }
        max_val = max(vals.values())
        ctrg = tex_pct(vals["CTRG"])
        delay = tex_pct(vals["Delay"])
        slack = tex_pct(vals["Slack"])
        if vals["CTRG"] == max_val:
            ctrg = bold(ctrg)
        if vals["Delay"] == max_val:
            delay = bold(delay)
        if vals["Slack"] == max_val:
            slack = bold(slack)
        rows.append(
            [
                short[crow.context_label],
                tex_int(crow.airports),
                tex_pct(crow.supported_share_stressed),
                tex_pct(crow.failure_rate_supported_stressed),
                ctrg,
                delay,
                slack,
                tex_num(vals["CTRG"] / vals["Delay"], 2),
            ]
        )
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    text = rf"""\begin{{table}}[!tbp]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.04}}
\caption{{Airport-context heterogeneity in the 2025 full experiment}}
\label{{tab:context-heterogeneity}}
\begin{{tabularx}}{{\linewidth}}{{@{{}}L{{0.22\linewidth}}C C C C C C C@{{}}}}
\toprule
Airport context & Airports & Support (\%) & Reference failed-exit (\%) & CTRG top 10\% & Delay top 10\% & Slack top 10\% & Ratio \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\tablenote{{Higher top-slice failed-exit rate means stronger enrichment within that ranking rule. Ratio is the CTRG top-10\% failed-exit rate divided by the delay-only top-10\% failed-exit rate. Bold marks the largest value among CTRG, delay-only, and slack-only rankings in each context.}}
\end{{table}}
"""
    write_text(TABLE_DIR / "tab_context_heterogeneity.tex", text)


def table_method_diagnostic(data: dict[str, pd.DataFrame]) -> None:
    diag = data["diagnostic"]
    keep = [
        "CTRG max-gap",
        "Observed-path ensemble",
        "ST propagation learner",
        "Multi-horizon hazard ensemble",
        "Network resilience learner",
        "Capacity-greedy recovery proxy",
    ]
    diag = diag[diag.method.isin(keep)].copy()
    diag["order"] = diag.method.map({name: i for i, name in enumerate(keep)})
    diag = diag.sort_values("order")
    max_failure = diag.failure_rate.max()
    max_gap = diag.mean_ctrg_gap_max.max()
    max_rewire = diag.severe_high_rewire_share.max()
    rows = []
    labels = {
        "CTRG max-gap": "CTRG max-gap",
        "ST propagation learner": "Spatiotemporal propagation learner",
    }
    for _, row in diag.iterrows():
        failure = tex_pct(row.failure_rate)
        gap = tex_num(row.mean_ctrg_gap_max, 3)
        rewire = tex_pct(row.severe_high_rewire_share)
        if row.failure_rate == max_failure:
            failure = bold(failure)
        if row.mean_ctrg_gap_max == max_gap:
            gap = bold(gap)
        if row.severe_high_rewire_share == max_rewire:
            rewire = bold(rewire)
        rows.append(
            [
                labels.get(row.method, row.method),
                row.problem_dimension,
                failure,
                gap,
                rewire,
            ]
        )
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    text = rf"""\begin{{table}}[!tbp]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{1.04}}
\caption{{Recent comparison-method diagnostic under public-data constraints}}
\label{{tab:method-diagnostic}}
\begin{{tabularx}}{{\linewidth}}{{@{{}}L{{0.24\linewidth}}Y r r r@{{}}}}
\toprule
Method family & Problem dimension & Failed-exit rate (\%) & Mean CTRG gap & Severe high-rewire (\%) \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\tablenote{{The diagnostic uses the same held-out supported stressed episodes in the focused method-family test. Higher failed-exit rate indicates stronger pure risk enrichment; higher mean CTRG gap and severe high-rewire share indicate stronger recoverability-gap evidence. Bold marks the largest value in each outcome column.}}
\end{{table}}
"""
    write_text(TABLE_DIR / "tab_method_diagnostic.tex", text)


def table_temporal_robustness(data: dict[str, pd.DataFrame]) -> None:
    yearly = data["yearly"]
    rows = []
    for _, row in yearly.iterrows():
        rows.append(
            [
                row.analysis,
                tex_int(row.supported_stressed),
                tex_pct(row.support_share),
                tex_pct(row.reference_failed_exit),
                tex_num(row.mean_gap_max, 3),
                bold(f"{tex_pct(row.ctrg_top10_failure)} [{tex_pct(row.ctrg_top10_ci_low)}, {tex_pct(row.ctrg_top10_ci_high)}]"),
                tex_pct(row.delay_top10_failure),
                tex_pct(row.slack_top10_failure),
            ]
        )
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    text = rf"""\begin{{table}}[!tbp]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{1.04}}
\caption{{Temporal robustness under the same airport set and graph rules}}
\label{{tab:temporal-robustness}}
\begin{{tabularx}}{{\linewidth}}{{@{{}}L{{0.14\linewidth}}C C C C L{{0.19\linewidth}}C C@{{}}}}
\toprule
Analysis & Supported stressed & Support (\%) & Reference failed-exit (\%) & Mean gap & CTRG top 10\% (\%, 95\% CI) & Delay top 10\% & Slack top 10\% \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\tablenote{{Bold CTRG entries are the temporal robustness target. The 2024 run uses the same 84-airport set, horizon, donor window, donor cap, and train--test logic as the 2025 main run.}}
\end{{table}}
"""
    write_text(TABLE_DIR / "tab_temporal_robustness.tex", text)


def supplement_table_context_profiles(data: dict[str, pd.DataFrame]) -> None:
    profiles = data["profiles"].sort_values("context_cluster")
    rows = []
    for _, row in profiles.iterrows():
        rows.append(
            [
                row.context_label,
                tex_int(row.airports),
                tex_int(row.median_departures),
                tex_num(row.median_routes, 1),
                tex_num(row.median_carriers, 1),
                tex_pct(row.median_dep_delay15_rate),
                tex_pct(row.median_cancel_rate),
                tex_num(row.median_taxi_out, 1),
                tex_num(row.median_distance, 1),
                tex_pct(row.median_carrier_top_share),
            ]
        )
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    text = rf"""\begin{{table}}[!tbp]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{1.04}}
\caption{{Airport-context descriptors in the 2025 estimable population}}
\label{{tab:supp-context-profiles}}
\begin{{tabularx}}{{\linewidth}}{{@{{}}L{{0.24\linewidth}}r r r r r r r r r@{{}}}}
\toprule
Airport context & Airports & Median departures & Median routes & Median carriers & Delay-15 (\%) & Cancel (\%) & Taxi-out (min) & Distance (mi) & Top carrier (\%) \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\tablenote{{Delay-15 is the median share of departures delayed by at least 15 minutes. Cancel is the median cancellation share. Taxi-out is the median taxi-out time. Distance is the median route stage length. Top carrier is the median share of departures operated by the largest carrier at an airport.}}
\end{{table}}
"""
    write_text(TABLE_DIR / "supp_tab_context_profiles.tex", text)


def supplement_table_monthly_stability(data: dict[str, pd.DataFrame]) -> None:
    monthly = data["monthly"].sort_values("month")
    max_support = monthly.supported_share_stressed.max()
    max_failure = monthly.failure_rate_supported_stressed.max()
    max_gap = monthly.mean_gap_max_supported.max()
    rows = []
    for _, row in monthly.iterrows():
        support = tex_pct(row.supported_share_stressed)
        failure = tex_pct(row.failure_rate_supported_stressed)
        gap = tex_num(row.mean_gap_max_supported, 3)
        if row.supported_share_stressed == max_support:
            support = bold(support)
        if row.failure_rate_supported_stressed == max_failure:
            failure = bold(failure)
        if row.mean_gap_max_supported == max_gap:
            gap = bold(gap)
        rows.append(
            [
                tex_int(row.month),
                tex_int(row.test_episodes),
                tex_int(row.stressed_episodes),
                tex_int(row.supported_stressed),
                support,
                failure,
                gap,
                tex_pct(row.severe_high_rewire_share),
                tex_pct(row.structural_brittle_share),
            ]
        )
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    text = rf"""\begin{{table}}[!tbp]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{1.04}}
\caption{{Monthly stability in the 2025 held-out period}}
\label{{tab:supp-monthly-stability}}
\begin{{tabularx}}{{\linewidth}}{{@{{}}r r r r r r r r r@{{}}}}
\toprule
Month & Test episodes & Stressed & Supported & Support (\%) & Failed exit (\%) & Mean gap & Severe high-rewire (\%) & Structural-brittle (\%) \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\tablenote{{The held-out period contains months 7--12. Support is the share of stressed episodes with at least one retained donor continuation. Severe high-rewire denotes episodes with outbound departure delay of at least 60 minutes and feasible-rewire recoverability of at least 0.70. Structural-brittle denotes supported stressed episodes whose best donor score is no higher than the observed-path score. Bold marks the highest support, failed-exit rate, and mean gap across months.}}
\end{{table}}
"""
    write_text(TABLE_DIR / "supp_tab_monthly_stability.tex", text)


def supplement_table_delay_bands(data: dict[str, pd.DataFrame]) -> None:
    delay = data["delay"].copy()
    max_failure = delay.failure_rate.max()
    max_gap = delay.mean_gap_max.max()
    max_structural = delay.structural_brittle_share.max()
    rows = []
    for _, row in delay.iterrows():
        failure = tex_pct(row.failure_rate)
        gap = tex_num(row.mean_gap_max, 3)
        structural = tex_pct(row.structural_brittle_share)
        if row.failure_rate == max_failure:
            failure = bold(failure)
        if row.mean_gap_max == max_gap:
            gap = bold(gap)
        if row.structural_brittle_share == max_structural:
            structural = bold(structural)
        rows.append(
            [
                str(row.delay_band_min),
                tex_int(row.episodes),
                failure,
                tex_num(row.mean_pred_recover, 3),
                tex_num(row.mean_donor_pred_max, 3),
                gap,
                tex_pct(row.high_rewire_share),
                structural,
            ]
        )
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    text = rf"""\begin{{table}}[!tbp]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.04}}
\caption{{Delay-band recoverability patterns in 2025}}
\label{{tab:supp-delay-bands}}
\begin{{tabularx}}{{\linewidth}}{{@{{}}L{{0.12\linewidth}}r r r r r r r@{{}}}}
\toprule
Delay band (min) & Episodes & Failed exit (\%) & Observed-path score & Feasible-rewire score & Mean gap & High-rewire (\%) & Structural-brittle (\%) \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\tablenote{{Delay band is based on outbound departure delay at the focal episode. High-rewire denotes feasible-rewire recoverability of at least 0.70. Larger observed-path and feasible-rewire scores mean higher estimated four-turn recovery probability. Larger mean gap means greater feasible continuation availability beyond the observed path. Bold marks the highest failed-exit rate, mean gap, and structural-brittle share.}}
\end{{table}}
"""
    write_text(TABLE_DIR / "supp_tab_delay_bands.tex", text)


def supplement_table_delay_causes(data: dict[str, pd.DataFrame]) -> None:
    cause = data["cause"].sort_values("episodes", ascending=False)
    max_failure = cause.failure_rate.max()
    max_gap = cause.mean_gap_max.max()
    max_structural = cause.structural_brittle_share.max()
    rows = []
    for _, row in cause.iterrows():
        failure = tex_pct(row.failure_rate)
        gap = tex_num(row.mean_gap_max, 3)
        structural = tex_pct(row.structural_brittle_share)
        if row.failure_rate == max_failure:
            failure = bold(failure)
        if row.mean_gap_max == max_gap:
            gap = bold(gap)
        if row.structural_brittle_share == max_structural:
            structural = bold(structural)
        rows.append(
            [
                row.dominant_delay_cause,
                tex_int(row.episodes),
                failure,
                tex_num(row.mean_pred_recover, 3),
                tex_num(row.mean_donor_pred_max, 3),
                gap,
                tex_pct(row.severe_high_rewire_share),
                structural,
            ]
        )
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    text = rf"""\begin{{table}}[!tbp]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.04}}
\caption{{Delay-cause recoverability patterns in 2025}}
\label{{tab:supp-delay-causes}}
\begin{{tabularx}}{{\linewidth}}{{@{{}}L{{0.24\linewidth}}r r r r r r r@{{}}}}
\toprule
Dominant delay cause & Episodes & Failed exit (\%) & Observed-path score & Feasible-rewire score & Mean gap & Severe high-rewire (\%) & Structural-brittle (\%) \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\tablenote{{Dominant delay cause follows the largest reported BTS delay component for the focal departure. National Air System denotes the BTS system-delay category. Higher severe high-rewire share indicates more severe-delay episodes with high feasible-rewire recoverability. Higher structural-brittle share indicates more episodes with limited improvement from compatible donors. Bold marks the highest failed-exit rate, mean gap, and structural-brittle share.}}
\end{{table}}
"""
    write_text(TABLE_DIR / "supp_tab_delay_causes.tex", text)


def supplement_table_yearly_context(data: dict[str, pd.DataFrame]) -> None:
    yearly_context = data["yearly_context"].sort_values(["context_cluster", "year"])
    rows = []
    for _, row in yearly_context.iterrows():
        ctrg = tex_pct(row.ctrg_top10_failure)
        delay = tex_pct(row.delay_top10_failure)
        slack = tex_pct(row.slack_top10_failure)
        max_rank = max(row.ctrg_top10_failure, row.delay_top10_failure, row.slack_top10_failure)
        if row.ctrg_top10_failure == max_rank:
            ctrg = bold(ctrg)
        if row.delay_top10_failure == max_rank:
            delay = bold(delay)
        if row.slack_top10_failure == max_rank:
            slack = bold(slack)
        rows.append(
            [
                str(int(row.year)),
                row.context_label,
                tex_int(row.airports),
                tex_int(row.stressed_episodes),
                tex_pct(row.supported_share),
                tex_pct(row.reference_failed_exit),
                tex_num(row.mean_gap, 3),
                ctrg,
                delay,
                slack,
            ]
        )
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    text = rf"""\begin{{table}}[!tbp]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{1.04}}
\caption{{Context-level temporal robustness}}
\label{{tab:supp-yearly-context}}
\begin{{tabularx}}{{\linewidth}}{{@{{}}r L{{0.22\linewidth}}r r r r r r r r@{{}}}}
\toprule
Year & Airport context & Airports & Stressed & Support (\%) & Reference failed-exit (\%) & Mean gap & CTRG top 10\% & Delay top 10\% & Slack top 10\% \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\tablenote{{Reference failed-exit is the failed-exit rate among all supported stressed episodes in the context and year. The top-10\% columns report failed-exit rates after ranking supported stressed episodes by CTRG recoverability gap, outbound departure delay, or low available turn time. Higher top-10\% values indicate stronger enrichment of realized brittle outcomes. Bold marks the largest top-10\% value within each context-year row.}}
\end{{table}}
"""
    write_text(TABLE_DIR / "supp_tab_yearly_context.tex", text)


def supplement_table_method_slices(data: dict[str, pd.DataFrame]) -> None:
    slices = data["diagnostic_slices"]
    keep = [
        "CTRG max-gap",
        "Observed-path ensemble",
        "ST propagation learner",
        "Multi-horizon hazard ensemble",
        "Network resilience learner",
        "Capacity-greedy recovery proxy",
    ]
    labels = {
        "ST propagation learner": "Spatiotemporal propagation learner",
        "top_5pct": "5",
        "top_10pct": "10",
        "top_20pct": "20",
    }
    slices = slices[slices.method.isin(keep)].copy()
    order = {name: i for i, name in enumerate(keep)}
    slice_order = {"top_5pct": 0, "top_10pct": 1, "top_20pct": 2}
    slices["order"] = slices.method.map(order)
    slices["slice_order"] = slices.slice.map(slice_order)
    slice_max_failure = slices.groupby("slice").failure_rate.transform("max")
    slice_max_gap = slices.groupby("slice").mean_ctrg_gap_max.transform("max")
    slice_max_rewire = slices.groupby("slice").severe_high_rewire_share.transform("max")
    rows = []
    for _, row in slices.sort_values(["order", "slice_order"]).iterrows():
        failure = tex_pct(row.failure_rate)
        gap = tex_num(row.mean_ctrg_gap_max, 3)
        severe = tex_pct(row.severe_high_rewire_share)
        if row.failure_rate == slice_max_failure.loc[row.name]:
            failure = bold(failure)
        if row.mean_ctrg_gap_max == slice_max_gap.loc[row.name]:
            gap = bold(gap)
        if row.severe_high_rewire_share == slice_max_rewire.loc[row.name]:
            severe = bold(severe)
        rows.append(
            [
                labels.get(row.method, row.method),
                labels.get(row.slice, row.slice),
                tex_int(row.n),
                failure,
                tex_num(row.failure_lift_vs_supported, 2),
                tex_num(row.mean_start_delay, 1),
                gap,
                tex_num(row.mean_donor_pred_max, 3),
                severe,
            ]
        )
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    text = rf"""\begin{{table}}[!tbp]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{1.04}}
\caption{{Comparison-method capacity-slice diagnostic}}
\label{{tab:supp-method-slices}}
\begin{{tabularx}}{{\linewidth}}{{@{{}}L{{0.24\linewidth}}r r r r r r r r@{{}}}}
\toprule
Method family & Slice (\%) & Episodes & Failed exit (\%) & Lift & Start delay (min) & Mean CTRG gap & Feasible-rewire score & Severe high-rewire (\%) \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\tablenote{{Slice is the selected top share among supported stressed episodes in the focused diagnostic sample. Lift is the failed-exit rate divided by the reference failed-exit rate in that sample. Start delay is the mean outbound departure delay at the focal episode. Higher mean CTRG gap and feasible-rewire score indicate stronger recoverability-gap evidence. Bold marks the largest failed-exit rate, mean CTRG gap, and severe high-rewire share within each slice size.}}
\end{{table}}
"""
    write_text(TABLE_DIR / "supp_tab_method_slices.tex", text)


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def place_bottom_legend(fig, handles, labels, *, ncol: int, y: float = 0.015) -> None:
    """Place one compact legend below the axes area."""
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label and label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)
    if unique_handles:
        fig.legend(
            unique_handles,
            unique_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, y),
            ncol=ncol,
            frameon=False,
            columnspacing=0.9,
            handlelength=1.5,
            handletextpad=0.45,
            borderaxespad=0.0,
        )


def figure_sampling_contexts(data: dict[str, pd.DataFrame]) -> None:
    setup_plot_style()
    sampling = data["sampling"].iloc[0]
    profiles = data["profiles"].sort_values("context_cluster")
    context = data["context"].sort_values("context_cluster")
    sparse_departures = int(sampling.total_departures - sampling.estimable_departures)
    labels = [
        "Large gateways",
        "Regional markets",
        "Carrier-dominant",
        "Long-stage",
        "Sparse audit",
    ]
    departures = list(profiles.departures / 1_000_000) + [sparse_departures / 1_000_000]
    support_labels = ["Large gateways", "Regional markets", "Carrier-dominant", "Long-stage"]
    support = list(context.supported_share_stressed * 100)
    colors = ["#3B6EA8", "#579B8E", "#C27C3A", "#8E5A9F", "#8A8A8A"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=False)
    ax = axes[0]
    y = range(len(labels))
    ax.barh(y, departures, color=colors, edgecolor="#333333", linewidth=0.4)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Departures in 2025 (million)")
    ax.set_title("(a) Sampling-frame coverage")
    for idx, val in enumerate(departures):
        ax.text(val + 0.03, idx, f"{val:.2f}", va="center", fontsize=7)
    ax.set_xlim(0, max(departures) * 1.20)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)

    ax = axes[1]
    x = range(len(support_labels))
    ax.bar(x, support, color=colors[:4], edgecolor="#333333", linewidth=0.4)
    ax.axhline(88.7255, color="#333333", linewidth=0.8, linestyle="--", label="Overall")
    ax.set_xticks(list(x), ["Large\ngateways", "Regional\nmarkets", "Carrier-\ndominant", "Long-stage"])
    ax.set_ylabel("Supported stressed episodes (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(b) Donor-support coverage")
    for idx, val in enumerate(support):
        ax.text(idx, val + 2, f"{val:.1f}", ha="center", fontsize=7)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.26, wspace=0.28)
    place_bottom_legend(fig, handles, legend_labels, ncol=1, y=0.035)
    fig.savefig(FIG_DIR / "fig_sampling_context_support.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_top_slice(data: dict[str, pd.DataFrame]) -> None:
    setup_plot_style()
    top = data["top"].copy()
    boot = data["boot"].copy()
    methods = [
        ("CTRG max-gap", "CTRG max-gap", "#3B6EA8", "o"),
        ("Observed-path recoverability", "Observed-path risk", "#8E5A9F", "s"),
        ("Delay-only", "Delay-only", "#C27C3A", "^"),
        ("Slack-only", "Slack-only", "#579B8E", "D"),
    ]
    x_map = {"top_5pct": 5, "top_10pct": 10, "top_20pct": 20}
    fig, ax = plt.subplots(figsize=(4.9, 3.55), constrained_layout=False)
    for table_name, boot_name, color, marker in methods:
        subset = top[top.ranking == table_name].copy()
        subset["x"] = subset.slice.map(x_map)
        subset = subset.sort_values("x")
        y = subset.failure_rate * 100
        x = subset.x
        bs = boot[(boot.scope == "all_contexts") & (boot.ranking == boot_name)].copy()
        bs["x"] = bs.slice.map(x_map)
        bs = bs.sort_values("x")
        lower = y.values - (bs.failure_rate_ci_low.values * 100)
        upper = (bs.failure_rate_ci_high.values * 100) - y.values
        ax.errorbar(
            x,
            y,
            yerr=[lower, upper],
            label=boot_name,
            color=color,
            marker=marker,
            linewidth=1.5,
            markersize=4,
            capsize=2,
        )
    ax.axhline(11.319857, color="#333333", linestyle="--", linewidth=0.8, label="Reference rate")
    ax.set_xlabel("Selected slice among supported stressed episodes (%)")
    ax.set_ylabel("Failed-exit rate (%)")
    ax.set_xticks([5, 10, 20])
    ax.set_ylim(8, 43)
    ax.grid(color="#DDDDDD", linewidth=0.5)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.subplots_adjust(left=0.13, right=0.98, top=0.94, bottom=0.34)
    place_bottom_legend(fig, handles, legend_labels, ncol=2, y=0.025)
    fig.savefig(FIG_DIR / "fig_top_slice_separation.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_context_heterogeneity(data: dict[str, pd.DataFrame]) -> None:
    setup_plot_style()
    top = data["context_top"]
    context = data["context"].sort_values("context_cluster")
    labels = ["Large\ngateways", "Regional\nmarkets", "Carrier-\ndominant", "Long-stage\ngateways"]
    methods = [
        ("CTRG max-gap", "CTRG", "#3B6EA8"),
        ("Delay-only", "Delay", "#C27C3A"),
        ("Slack-only", "Slack", "#579B8E"),
    ]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.1),
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )

    ax = axes[0]
    width = 0.23
    x = list(range(len(labels)))
    for offset_idx, (method, label, color) in enumerate(methods):
        vals = []
        for _, row in context.iterrows():
            vals.append(
                float(
                    top[
                        (top.context_cluster == row.context_cluster)
                        & (top.ranking == method)
                        & (top.slice == "top_10pct")
                    ].failure_rate.iloc[0]
                )
                * 100
            )
        positions = [v + (offset_idx - 1) * width for v in x]
        ax.bar(positions, vals, width=width, label=label, color=color, edgecolor="#333333", linewidth=0.4)
        for px, val in zip(positions, vals):
            ax.text(px, val + 0.8, f"{val:.1f}", ha="center", fontsize=5.7)
    reference = list(context.failure_rate_supported_stressed * 100)
    ax.scatter(
        x,
        reference,
        label="Reference",
        marker="D",
        s=20,
        facecolor="white",
        edgecolor="#333333",
        linewidth=0.8,
        zorder=4,
    )
    for px, val in zip(x, reference):
        ax.text(px, val - 1.6, f"{val:.1f}", ha="center", va="top", fontsize=6.3, color="#333333")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Top-10% failed-exit rate (%)")
    ax.set_title("(a) Enrichment relative to context reference")
    ax.set_ylim(0, 47)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    handles_left, labels_left = ax.get_legend_handles_labels()

    ax = axes[1]
    support = list(context.supported_share_stressed * 100)
    donors = list(context.median_donor_count_supported)
    colors = ["#3B6EA8", "#579B8E", "#C27C3A", "#8E5A9F"]
    bars = ax.bar(x, support, color=colors, edgecolor="#333333", linewidth=0.4)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Supported stressed episodes (%)")
    ax.set_title("(b) Support depth")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    for bar, val in zip(bars, support):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.8, f"{val:.1f}", ha="center", fontsize=6.5)
    ax2 = ax.twinx()
    ax2.plot(x, donors, color="#333333", marker="o", linewidth=1.2, markersize=3.5, label="Median donors")
    ax2.set_ylabel("Median donor count")
    ax2.set_ylim(0, max(donors) * 1.35)
    for px, val in zip(x, donors):
        ax2.text(px, val + 0.5, f"{val:.0f}", ha="center", fontsize=6.3, color="#333333")
    handles_right, labels_right = ax2.get_legend_handles_labels()
    fig.subplots_adjust(left=0.08, right=0.93, top=0.89, bottom=0.24, wspace=0.34)
    place_bottom_legend(
        fig,
        handles_left + handles_right,
        labels_left + labels_right,
        ncol=5,
        y=0.025,
    )
    fig.savefig(FIG_DIR / "fig_context_heterogeneity.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_counterfactual_patterns(data: dict[str, pd.DataFrame]) -> None:
    setup_plot_style()
    delay = data["delay"].copy()
    cause = data["cause"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(7.45, 3.35), constrained_layout=False)
    delay_tick_labels = ["15-\n29", "30-\n59", "60-\n119", "120-\n239", "240+"]

    ax = axes[0]
    x = list(range(len(delay)))
    ax.plot(x, delay.mean_pred_recover, marker="o", color="#8E5A9F", linewidth=1.5, label="Observed-path")
    ax.plot(x, delay.mean_donor_pred_max, marker="s", color="#3B6EA8", linewidth=1.5, label="Feasible rewire")
    ax.bar(x, delay.mean_gap_max, color="#C27C3A", alpha=0.35, label="Gap")
    ax.set_xticks(x, delay_tick_labels)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Recoverability score or gap")
    ax.set_xlabel("Departure-delay band (min)")
    ax.set_title("(a) Recovery scores")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    handles_a, labels_a = ax.get_legend_handles_labels()

    ax = axes[1]
    failure_vals = delay.failure_rate * 100
    high_rewire_vals = delay.high_rewire_share * 100
    bars = ax.bar(
        x,
        failure_vals,
        color="#C27C3A",
        alpha=0.75,
        edgecolor="#333333",
        linewidth=0.4,
        label="Failed exit",
    )
    ax.set_xticks(x, delay_tick_labels)
    ax.set_ylabel("Failed-exit rate (%)")
    ax.set_xlabel("Departure-delay band (min)")
    ax.set_title("(b) Failure and rewire")
    ax.set_ylim(0, max(failure_vals) * 1.45)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    for bar, val in zip(bars, failure_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.1f}", ha="center", fontsize=6.2)
    ax2 = ax.twinx()
    ax2.plot(x, high_rewire_vals, color="#3B6EA8", marker="s", linewidth=1.3, markersize=3.2, label="High rewire")
    ax2.set_ylabel("")
    ax2.set_ylim(0, 100)
    lines, line_labels = ax.get_legend_handles_labels()
    lines2, line_labels2 = ax2.get_legend_handles_labels()
    handles_b = lines + lines2
    labels_b = line_labels + line_labels2

    ax = axes[2]
    order = cause.sort_values("structural_brittle_share", ascending=False)
    labels = [
        "Weather" if v == "Weather" else
        "System\ndelay" if v == "National Air System" else
        "Unreported" if v.startswith("Unreported") else
        "Late\naircraft" if v == "Late aircraft" else
        v
        for v in order.dominant_delay_cause
    ]
    vals = order.structural_brittle_share * 100
    gap_vals = order.mean_gap_max
    pos = list(range(len(order)))
    ax.barh(pos, vals, color="#579B8E", edgecolor="#333333", linewidth=0.4)
    ax.set_yticks(pos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Structural-brittle share (%)")
    ax.set_title("(c) Cause-specific brittleness")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    ax.set_xlim(0, max(vals) * 1.35)
    for idx, val in enumerate(vals):
        ax.text(val + 0.08, idx, f"{val:.1f}", va="center", fontsize=6.3)
    ax2 = ax.twiny()
    ax2.plot(gap_vals, pos, color="#333333", marker="o", linewidth=1.2, markersize=3.2, label="Mean gap")
    ax2.set_xlabel("Mean gap")
    ax2.set_xlim(0, max(gap_vals) * 1.35)
    handles_c, labels_c = ax2.get_legend_handles_labels()
    fig.subplots_adjust(left=0.07, right=0.96, top=0.88, bottom=0.27, wspace=0.62)
    place_bottom_legend(
        fig,
        handles_a + handles_b + handles_c,
        labels_a + labels_b + labels_c,
        ncol=6,
        y=0.02,
    )
    fig.savefig(FIG_DIR / "fig_counterfactual_patterns.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = load_data()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    table_sampling_contexts(data)
    table_experimental_scale(data)
    table_top_slice_comparison(data)
    table_context_heterogeneity(data)
    table_method_diagnostic(data)
    table_temporal_robustness(data)
    supplement_table_context_profiles(data)
    supplement_table_monthly_stability(data)
    supplement_table_delay_bands(data)
    supplement_table_delay_causes(data)
    supplement_table_yearly_context(data)
    supplement_table_method_slices(data)
    figure_sampling_contexts(data)
    figure_top_slice(data)
    figure_context_heterogeneity(data)
    figure_counterfactual_patterns(data)


if __name__ == "__main__":
    main()
