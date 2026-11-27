from __future__ import annotations

import argparse
from pathlib import Path

from dispatch_env.sarsa_evaluator import (
    EVALUATION_SEEDS,
    evaluate_checkpoint,
    format_evaluation_report,
    save_evaluation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SARSA checkpoint on the common fixed seeds"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trained SARSA JSON checkpoint to evaluate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON report path (default: <checkpoint>.evaluation.json)",
    )
    args = parser.parse_args()

    report = evaluate_checkpoint(args.checkpoint, seeds=EVALUATION_SEEDS)
    output_path = args.output or args.checkpoint.with_name(
        "{}.evaluation.json".format(args.checkpoint.stem)
    )
    save_evaluation_report(report, output_path)
    print(format_evaluation_report(report))
    print("Saved per-seed evaluation report: {}".format(output_path.resolve()))


if __name__ == "__main__":
    main()
