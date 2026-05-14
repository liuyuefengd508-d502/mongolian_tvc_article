#!/usr/bin/env python3
"""Evaluate the small OCR case study after expert transcripts are filled.

Expected input:
  page_level_ocr/results/ocr_case_study_3pages/
    expert_transcription_template_oracle.csv
    detected_crop_review_template.csv

The script computes:
  * oracle-crop CER/WER if OCR predictions are available;
  * detected-crop CER/WER on matched GT columns when expert transcripts can be
    propagated from oracle rows to detected rows;
  * layout coverage of the selected case-study pages.

If OCR prediction fields are empty, it still validates transcript coverage and
reports that recognition metrics are not computable.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_end_to_end_ocr_experiment import cer, wer


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=Path("page_level_ocr/results/ocr_case_study_3pages"))
    args = parser.parse_args()

    oracle_path = args.case_dir / "expert_transcription_template_oracle.csv"
    detected_path = args.case_dir / "detected_crop_review_template.csv"
    oracle_rows = read_csv(oracle_path)
    detected_rows = read_csv(detected_path) if detected_path.exists() else []

    transcript_by_col = {
        row["column_id"]: row.get("expert_transcript", "").strip()
        for row in oracle_rows
        if row.get("column_id") and row.get("expert_transcript", "").strip()
    }
    oracle_cer, oracle_wer = [], []
    for row in oracle_rows:
        gt = row.get("expert_transcript", "").strip()
        pred = row.get("ocr_prediction_optional", "").strip()
        if gt and pred:
            oracle_cer.append(cer(pred, gt))
            oracle_wer.append(wer(pred, gt))

    detected_cer, detected_wer = [], []
    for row in detected_rows:
        gt = transcript_by_col.get(row.get("matched_gt_column_id", ""), "")
        pred = row.get("ocr_prediction_optional", "").strip()
        row["expert_transcript_for_matched_gt"] = gt
        if gt and pred:
            detected_cer.append(cer(pred, gt))
            detected_wer.append(wer(pred, gt))

    if detected_rows:
        with detected_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(detected_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detected_rows)

    summary = {
        "num_oracle_rows": len(oracle_rows),
        "num_filled_expert_transcripts": len(transcript_by_col),
        "oracle_ocr": {
            "num_evaluated": len(oracle_cer),
            "mean_cer": mean(oracle_cer),
            "mean_wer": mean(oracle_wer),
        },
        "detected_ocr": {
            "num_evaluated": len(detected_cer),
            "mean_cer": mean(detected_cer),
            "mean_wer": mean(detected_wer),
        },
        "status": "computed" if oracle_cer or detected_cer else "transcripts_or_predictions_missing",
    }
    (args.case_dir / "ocr_case_study_evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.case_dir / "ocr_case_study_evaluation_summary.md").write_text(
        "\n".join(
            [
                "# OCR Case Study Evaluation Summary",
                "",
                f"- filled expert transcripts: {summary['num_filled_expert_transcripts']} / {summary['num_oracle_rows']}",
                f"- oracle evaluated columns: {summary['oracle_ocr']['num_evaluated']}",
                f"- oracle mean CER: {summary['oracle_ocr']['mean_cer']}",
                f"- oracle mean WER: {summary['oracle_ocr']['mean_wer']}",
                f"- detected evaluated columns: {summary['detected_ocr']['num_evaluated']}",
                f"- detected mean CER: {summary['detected_ocr']['mean_cer']}",
                f"- detected mean WER: {summary['detected_ocr']['mean_wer']}",
                f"- status: {summary['status']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
