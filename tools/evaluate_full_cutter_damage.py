from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from full_cutter_detector import DEFAULT_CHECKPOINT, FullCutterDetector


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "training_full_cutter_damage_v1"
DEFAULT_OUTPUT = (
    ROOT
    / "models"
    / "full_cutter_damage_v1"
    / "full_image_multiscale_evaluation.json"
)


def box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    intersection = (
        max(0.0, min(ax2, bx2) - max(ax1, bx1))
        * max(0.0, min(ay2, by2) - max(ay1, by1))
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for view in ("all", "top", "isometric"):
        selected = rows if view == "all" else [row for row in rows if row["view"] == view]
        ious = [float(row["iou"]) for row in selected]
        matched = sum(bool(row["match_iou_50"]) for row in selected)
        summary[view] = {
            "images": len(selected),
            "matched_iou_50": matched,
            "recall_at_iou_50": matched / len(selected),
            "mean_iou": statistics.mean(ious),
            "median_iou": statistics.median(ious),
            "minimum_iou": min(ious),
            "no_detection": sum(not row["detected"] for row in selected),
        }
    return summary


def evaluate(
    checkpoint: Path,
    dataset_root: Path,
    *,
    split: str = "val",
    image_sizes: tuple[int, ...] = (640, 1024),
    device: str | int | None = None,
) -> dict:
    manifest_path = dataset_root / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = [item for item in manifest["images"] if item["split"] == split]
    if not items:
        raise ValueError(f"No images found for split {split!r}.")

    detector = FullCutterDetector(
        checkpoint, image_sizes=image_sizes, device=device
    )
    rows = []
    for item in items:
        prediction = detector.predict_primary(dataset_root / item["exported_image"])
        if prediction is None:
            rows.append({
                "source_key": item["source_key"],
                "view": item["view"],
                "detected": False,
                "confidence": None,
                "iou": 0.0,
                "match_iou_50": False,
            })
            continue
        iou = max(
            box_iou(prediction["box_xyxy_normalized"], ground_truth)
            for ground_truth in item["boxes_xyxy_normalized"]
        )
        rows.append({
            "source_key": item["source_key"],
            "view": item["view"],
            "detected": True,
            "confidence": prediction["confidence"],
            "box_xyxy_normalized": prediction["box_xyxy_normalized"],
            "iou": iou,
            "match_iou_50": iou >= 0.50,
            "candidate_count": prediction["candidate_count"],
            "fusion_applied": prediction["fusion_applied"],
            "fusion_source_count": prediction["fusion_source_count"],
            "fusion_members": prediction["fusion_members"],
            "speed_ms": prediction["speed_ms"],
        })
    return {
        "checkpoint": str(checkpoint.resolve()),
        "dataset": str(dataset_root.resolve()),
        "split": split,
        "inference_contract": {
            "input": "complete EXIF-normalized RGB image",
            "image_sizes": list(image_sizes),
            "confidence": detector.confidence,
            "candidate_nms_iou": detector.candidate_iou,
            "fusion_cluster_iou": detector.cluster_iou,
            "selection": "fused candidate nearest full-image center",
        },
        "summary": summarize(rows),
        "images": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate full-image multi-scale center-cutter inference."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--image-sizes", type=int, nargs="+", default=(640, 1024))
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = evaluate(
        args.checkpoint,
        args.dataset,
        split=args.split,
        image_sizes=tuple(args.image_sizes),
        device=args.device,
    )
    if args.output.exists():
        raise FileExistsError(
            f"Evaluation output already exists: {args.output}. Choose a new --output path."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Evaluation report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
