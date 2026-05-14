#!/usr/bin/env python3
"""Train RT-DETR-L and DocLayout-YOLO with YOLOv8n-comparable budget (50 epochs + patience).

RT-DETR uses ``ultralytics.RTDETR`` (not the ``yolo`` CLI shimmed to doclayout_yolo),
because the latter hits a validator ``postprocess(conf=...)`` incompatibility.

DocLayout-YOLO uses ``doclayout_yolo.YOLOv10`` with ``val=False`` if needed to avoid
the same validator issue; run ``evaluate_doclayout_yolo_baseline.py`` afterward for
project-level val-threshold selection.

Example:
  cd page_level_ocr
  .venv_yolo/bin/python train_fair_transfer_baselines.py --only rtdetr
  .venv_yolo/bin/python evaluate_rtdetr_baseline.py --annotations page_level_annotations.json \\
      --weights runs/detect/results/rtdetr_baseline/rtdetr_l_fair50ep/weights/best.pt \\
      --output-dir results/rtdetr_baseline/rtdetr_l_fair50ep_eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RTDETR_INIT = ROOT.parent / "rtdetr-l.pt"
DOC_INIT = ROOT / "results/doclayout_yolo_baseline/hf_weights/doclayout_yolo_docstructbench_imgsz1024.pt"
DATA = ROOT / "page_level_yolo_dataset_105/data.yaml"


def train_rtdetr(epochs: int, patience: int, name: str) -> Path:
    from ultralytics import RTDETR

    if not RTDETR_INIT.exists():
        raise FileNotFoundError(f"Missing RT-DETR weights: {RTDETR_INIT}")
    model = RTDETR(str(RTDETR_INIT))
    model.train(
        data=str(DATA),
        epochs=epochs,
        imgsz=640,
        batch=1,
        device="cpu",
        workers=0,
        project=str(ROOT / "runs/detect/results/rtdetr_baseline"),
        name=name,
        exist_ok=True,
        seed=20260513,
        patience=patience,
        verbose=True,
    )
    save_dir = ROOT / "runs/detect/results/rtdetr_baseline" / name
    return save_dir / "weights" / "best.pt"


def train_doclayout(epochs: int, patience: int, name: str, val: bool) -> Path:
    from doclayout_yolo import YOLOv10

    if not DOC_INIT.exists():
        raise FileNotFoundError(f"Missing DocLayout weights: {DOC_INIT}")
    model = YOLOv10(str(DOC_INIT))
    model.train(
        data=str(DATA),
        epochs=epochs,
        imgsz=1024,
        batch=1,
        device="cpu",
        workers=0,
        project=str(ROOT / "results/doclayout_yolo_baseline"),
        name=name,
        exist_ok=True,
        seed=20260513,
        patience=patience,
        val=val,
        verbose=True,
    )
    save_dir = ROOT / "results/doclayout_yolo_baseline" / name
    return save_dir / "weights" / "best.pt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", choices=("rtdetr", "doclayout", "both"), default="both")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=15, help="Early-stop patience on fitness (0=disabled).")
    p.add_argument("--doclayout-val", action="store_true", help="Enable built-in val for DocLayout (may crash on some stacks).")
    args = p.parse_args()

    epochs = args.epochs
    patience = args.patience if args.patience > 0 else 99999

    if args.only in ("rtdetr", "both"):
        out = train_rtdetr(epochs, patience, "rtdetr_l_fair50ep")
        print("RT-DETR best weights:", out)

    if args.only in ("doclayout", "both"):
        out = train_doclayout(epochs, patience, "doclayout_yolo_fair50ep", val=args.doclayout_val)
        print("DocLayout-YOLO best weights:", out)


if __name__ == "__main__":
    main()
