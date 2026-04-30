"""
End-to-end pipeline scheduler for the Data Mining project.

Key requirement from user:
- features.py must run once before regime_label.py
- features.py must run once after regime_label.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Step:
    name: str
    script: str
    expected_outputs: list[str]


def run_step(step: Step, project_dir: Path, python_exec: str, dry_run: bool = False) -> None:
    script_path = project_dir / step.script
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    cmd = [python_exec, str(script_path)]
    print("\n" + "=" * 90)
    print(f"[RUN] {step.name}")
    print(f"      Script: {step.script}")
    print("=" * 90)

    if dry_run:
        print(f"[DRY-RUN] {' '.join(cmd)}")
        return

    t0 = time.time()
    subprocess.run(cmd, cwd=project_dir, check=True)
    dt = time.time() - t0
    print(f"[DONE] {step.name} ({dt:.1f}s)")

    for rel in step.expected_outputs:
        out_path = project_dir / rel
        if not out_path.exists():
            raise FileNotFoundError(f"Expected output not found after step '{step.name}': {out_path}")


def build_steps(skip_visualization: bool, skip_best_analysis: bool) -> list[Step]:
    steps: list[Step] = []

    # Data ingestion
    steps.append(
        Step(
            name="Download data from Yahoo",
            script="download_data_yahoo.py",
            expected_outputs=["data_yahoo.csv"],
        )
    )
    steps.append(
        Step(
            name="Download data from IBKR",
            script="download_data_ibkr.py",
            expected_outputs=["data_ibkr.csv"],
        )
    )

    # Merge raw data
    steps.append(
        Step(
            name="Combine Yahoo/IBKR data",
            script="combine_data.py",
            expected_outputs=["combine_data.csv"],
        )
    )

    # REQUIRED ORDER: features before regime
    steps.append(
        Step(
            name="Feature engineering (pass 1, before regime_label)",
            script="features.py",
            expected_outputs=["features.csv"],
        )
    )

    # Regime labeling depends on features.csv from pass 1
    steps.append(
        Step(
            name="Generate regime labels",
            script="regime_label.py",
            expected_outputs=["regime_label.csv"],
        )
    )

    # REQUIRED ORDER: features after regime
    steps.append(
        Step(
            name="Feature engineering (pass 2, after regime_label)",
            script="features.py",
            expected_outputs=["features.csv"],
        )
    )

    # Supervised labels
    steps.append(
        Step(
            name="Generate target labels",
            script="target_label.py",
            expected_outputs=["target_label.csv"],
        )
    )

    if not skip_visualization:
        steps.append(
            Step(
                name="Generate exploratory visualizations",
                script="visualization.py",
                expected_outputs=["daily_close_timeseries_all_symbols.png"],
            )
        )

    # Feature selection and model training/evaluation
    steps.append(
        Step(
            name="Feature selection",
            script="feature_selection.py",
            expected_outputs=["feature_importance_sp500.csv"],
        )
    )
    steps.append(
        Step(
            name="Batch model comparison",
            script="model_comparison_batch.py",
            expected_outputs=["model_comparison_batch_results.xlsx"],
        )
    )

    if not skip_best_analysis:
        steps.append(
            Step(
                name="Best model analysis and dashboard plots",
                script="best_model_analysis.py",
                expected_outputs=["best_model_analysis_output"],
            )
        )

    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full project pipeline in dependency order.")
    parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="Project directory where scripts/data are located.",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip visualization.py step.",
    )
    parser.add_argument(
        "--skip-best-analysis",
        action="store_true",
        help="Skip best_model_analysis.py step.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print steps/commands only, do not execute.",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    python_exec = sys.executable

    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")

    print("=" * 90)
    print("PIPELINE SCHEDULER")
    print("=" * 90)
    print(f"Project dir : {project_dir}")
    print(f"Python      : {python_exec}")
    print("IBKR step   : REQUIRED")
    print(f"dry_run     : {args.dry_run}")

    steps = build_steps(
        skip_visualization=args.skip_visualization,
        skip_best_analysis=args.skip_best_analysis,
    )

    print("\nPlanned steps:")
    for i, s in enumerate(steps, 1):
        print(f"{i:2d}. {s.name} -> {s.script}")

    t0 = time.time()
    for step in steps:
        run_step(step, project_dir, python_exec, dry_run=args.dry_run)

    print("\n" + "=" * 90)
    print(f"PIPELINE COMPLETED in {time.time() - t0:.1f}s")
    print("=" * 90)


if __name__ == "__main__":
    main()
