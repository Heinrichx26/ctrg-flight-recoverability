# CTRG Flight Recoverability

This repository contains the reproducible code for Counterfactual Tail-Swap Recoverability Graphs (CTRG), a public-record method for measuring recoverability in aircraft delay chains.

The code uses U.S. Bureau of Transportation Statistics (BTS) Airline On-Time Performance records. The repository does not redistribute source flight records or derived experiment outputs. Users should obtain the public records from the BTS TranStats portal before running the scripts:

- BTS Airline On-Time Performance records: https://transtats.bts.gov/ONTIME/
- BTS pre-zipped monthly files used by the scripts: https://transtats.bts.gov/PREZIP/

## Repository Layout

```text
src/ctrg/     Analysis code
data/         Local data directory created by scripts
results/      Local output directory created by scripts
```

## Environment

Python 3.11 or newer is recommended.

```bash
pip install -r requirements.txt
```

## Data Preparation Prerequisite

Prepare the BTS monthly zip files locally before running any workflow. The scripts read local files and do not download public records automatically.

Place each monthly zip file in:

```text
data/ctrg/raw_bts/
```

Use this repository-local naming pattern:

```text
data/ctrg/raw_bts/bts_otp_YYYY_MM.zip
```

For example, the January 2024 BTS pre-zipped file should be saved as:

```text
data/ctrg/raw_bts/bts_otp_2024_01.zip
```

The corresponding BTS source file is named:

```text
On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2024_1.zip
```

If a required local file is missing, the scripts stop and report the expected path and the BTS public source portal.

## Quick Smoke Test

After preparing the January 2024 BTS zip locally, run a small diagnostic workflow on five airports:

```bash
python src/ctrg/run_ctrg_experiment.py \
  --mode smoke \
  --years 2024 \
  --months 1 \
  --airports ATL DFW DEN ORD LAX
```

Outputs are written to `results/ctrg/smoke`.

## Full Reproduction Workflow

The full workflow uses complete public years. It may take substantial time and disk space because the BTS monthly files and reconstructed turnaround tables are large. Prepare all required monthly zip files in `data/ctrg/raw_bts` before running the commands.

1. Classify the 2025 airport sampling frame.

```bash
python src/ctrg/classify_airport_contexts.py --year 2025
python src/ctrg/build_airport_context_report.py
```

2. Build 2025 and 2024 turnaround tables for the estimable airport set. Pass the airport list reported in `results/ctrg/summary_tables/airport_context_membership_2025.csv`.

```bash
python src/ctrg/build_turnarounds_for_period.py \
  --years 2025 \
  --months 1 2 3 4 5 6 7 8 9 10 11 12 \
  --airports <AIRPORT_CODES> \
  --output-name turnarounds_2025.csv

python src/ctrg/build_turnarounds_for_period.py \
  --years 2024 \
  --months 1 2 3 4 5 6 7 8 9 10 11 12 \
  --airports <AIRPORT_CODES> \
  --output-name turnarounds_2024.csv
```

3. Run CTRG on the reconstructed turnaround tables.

```bash
python src/ctrg/resume_ctrg_from_turnarounds.py \
  --mode full \
  --turnarounds data/ctrg/processed/turnarounds_2025.csv \
  --years 2025 \
  --bootstrap-reps 300

python src/ctrg/resume_ctrg_from_turnarounds.py \
  --mode robust_2024 \
  --turnarounds data/ctrg/processed/turnarounds_2024.csv \
  --years 2024 \
  --bootstrap-reps 300
```

4. Build yearly robustness summaries and comparison-method diagnostics.

```bash
python src/ctrg/build_yearly_summary.py

python src/ctrg/run_comparison_methods_diagnostic.py \
  --turnarounds data/ctrg/processed/turnarounds_2025.csv
```

5. Build display tables and figures from saved results. This step does not rerun the full experiments.

```bash
python src/ctrg/build_display_items.py
```

Display outputs are written to `results/display`.

## Method Notes

CTRG evaluates stressed turnaround episodes using compatible donor continuations observed in the same public data source. A donor continuation must match the observable airport-carrier context, fall within a bounded scheduled-departure time window, come from a different aircraft tail, satisfy minimum available turn time, avoid cancellation and diversion, and remain close in distance group when that descriptor is available.

The code reports observed-path recoverability, feasible-rewire recoverability, recoverability gap, donor count, support status, and top-slice failed-exit enrichment. Comparison-method diagnostics use the same public-data constraints.

## License

Code is released under the MIT License.
