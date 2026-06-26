from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def read_summary(results_dir: Path) -> dict:
    return json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))


def read_top10(results_dir: Path) -> dict[str, dict]:
    top = pd.read_csv(results_dir / "top_slice_table.csv")
    out = {}
    for ranking in ["CTRG max-gap", "Delay-only", "Slack-only", "Observed-path recoverability"]:
        row = top[(top["ranking"] == ranking) & (top["slice"] == "top_10pct")].iloc[0]
        out[ranking] = row.to_dict()
    return out


def read_bootstrap_top10(results_dir: Path) -> dict[str, dict]:
    path = results_dir / "top_slice_bootstrap.csv"
    if not path.exists():
        return {}
    boot = pd.read_csv(path)
    out = {}
    for ranking in ["CTRG max-gap", "Delay-only", "Slack-only", "Observed-path risk"]:
        row = boot[(boot["ranking"] == ranking) & (boot["slice"] == "top_10pct")]
        if not row.empty:
            out[ranking] = row.iloc[0].to_dict()
    return out


def yearly_row(label: str, results_dir: Path) -> dict:
    summary = read_summary(results_dir)
    top10 = read_top10(results_dir)
    boot = read_bootstrap_top10(results_dir)
    counts = summary["counts"]
    model = summary["model"]
    support = summary["support"]
    patterns = summary["counterfactual_patterns"]
    ctrg_boot = boot.get("CTRG max-gap", {})
    delay_boot = boot.get("Delay-only", {})
    slack_boot = boot.get("Slack-only", {})
    observed_boot = boot.get("Observed-path risk", {})
    return {
        "analysis": label,
        "turnarounds": counts["raw_turnarounds"],
        "test_episodes": counts["test_endpoint_episodes"],
        "stressed_episodes": counts["test_stressed_episodes"],
        "supported_stressed": counts["supported_stressed_episodes"],
        "support_share": support["supported_share_among_stressed"],
        "donor_edges": counts["donor_edges"],
        "recover_auc": model["recover_auc"],
        "recover_ap": model["recover_average_precision"],
        "reference_failed_exit": patterns["supported_stressed_failure_rate"],
        "mean_gap_max": patterns["mean_gap_max_supported"],
        "severe_high_rewire_share": patterns["severe_high_rewire_share"],
        "structural_brittle_share": patterns["structural_brittle_share"],
        "ctrg_top10_failure": top10["CTRG max-gap"]["failure_rate"],
        "ctrg_top10_ci_low": ctrg_boot.get("failure_rate_ci_low"),
        "ctrg_top10_ci_high": ctrg_boot.get("failure_rate_ci_high"),
        "delay_top10_failure": top10["Delay-only"]["failure_rate"],
        "delay_top10_ci_low": delay_boot.get("failure_rate_ci_low"),
        "delay_top10_ci_high": delay_boot.get("failure_rate_ci_high"),
        "slack_top10_failure": top10["Slack-only"]["failure_rate"],
        "slack_top10_ci_low": slack_boot.get("failure_rate_ci_low"),
        "slack_top10_ci_high": slack_boot.get("failure_rate_ci_high"),
        "observed_path_top10_failure": top10["Observed-path recoverability"]["failure_rate"],
        "observed_path_top10_ci_low": observed_boot.get("failure_rate_ci_low"),
        "observed_path_top10_ci_high": observed_boot.get("failure_rate_ci_high"),
    }


def context_rows(year: int, results_dir: Path) -> list[dict]:
    context = pd.read_csv(results_dir / "airport_context_result_table.csv")
    top = pd.read_csv(results_dir / "airport_context_top_slice_table.csv")
    rows = []
    for _, row in context.iterrows():
        subset = top[
            (top["context_cluster"] == row.context_cluster)
            & (top["slice"] == "top_10pct")
        ]
        def rate(ranking: str) -> float:
            return float(subset[subset["ranking"] == ranking]["failure_rate"].iloc[0])

        rows.append(
            {
                "year": year,
                "context_cluster": int(row.context_cluster),
                "context_label": row.context_label,
                "airports": int(row.airports),
                "stressed_episodes": int(row.stressed_episodes),
                "supported_share": float(row.supported_share_stressed),
                "reference_failed_exit": float(row.failure_rate_supported_stressed),
                "mean_gap": float(row.mean_gap_max_supported),
                "ctrg_top10_failure": rate("CTRG max-gap"),
                "delay_top10_failure": rate("Delay-only"),
                "slack_top10_failure": rate("Slack-only"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dir", default=str(ROOT / "results" / "ctrg" / "full"))
    parser.add_argument("--main-label", default="2025 main")
    parser.add_argument("--main-year", type=int, default=2025)
    parser.add_argument("--robust-dir", default=str(ROOT / "results" / "ctrg" / "robust_2024"))
    parser.add_argument("--robust-label", default="2024 robustness")
    parser.add_argument("--robust-year", type=int, default=2024)
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "ctrg" / "summary_tables"))
    args = parser.parse_args()

    main_dir = Path(args.main_dir)
    robust_dir = Path(args.robust_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yearly = pd.DataFrame(
        [
            yearly_row(args.main_label, main_dir),
            yearly_row(args.robust_label, robust_dir),
        ]
    )
    yearly.to_csv(out_dir / "yearly_robustness_summary.csv", index=False)

    contexts = pd.DataFrame(
        context_rows(args.main_year, main_dir)
        + context_rows(args.robust_year, robust_dir)
    )
    contexts.to_csv(out_dir / "yearly_context_robustness_summary.csv", index=False)

    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
