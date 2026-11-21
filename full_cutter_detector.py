from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image

from crop_pipeline import fuse_detection_candidates
from image_orientation import normalize_uploaded_image


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "models"
    / "full_cutter_damage_v1"
    / "weights"
    / "best.pt"
)
DEFAULT_IMAGE_SIZES = (640, 1024)
CANDIDATE_NMS_IOU = 0.99
FUSION_CLUSTER_IOU = 0.60
DEFAULT_CONFIDENCE = 0.25
CLASS_NAME = "full_cutter"


def select_center_cluster(
    candidates: Iterable[dict],
    image_size: tuple[int, int],
    *,
    cluster_iou: float = FUSION_CLUSTER_IOU,
) -> dict | None:
    """Fuse overlapping multi-scale candidates and select nearest image center.

    Candidate boxes must use full-image pixel coordinates. Confidence breaks
    exact center-distance ties; it does not override center priority.
    """
    candidates = list(candidates)
    if not candidates:
        return None
    fused = fuse_detection_candidates(candidates, iou_threshold=cluster_iou)
    width, height = map(float, image_size)
    center_x, center_y = width / 2.0, height / 2.0

    def selection_key(item: dict):
        x1, y1, x2, y2 = map(float, item["box"])
        box_center_x = (x1 + x2) / 2.0
        box_center_y = (y1 + y2) / 2.0
        distance_squared = (
            (box_center_x - center_x) ** 2
            + (box_center_y - center_y) ** 2
        )
        return distance_squared, -float(item["confidence"])

    return min(fused, key=selection_key)


class FullCutterDetector:
    """Reference inference wrapper for the damage-rich center-cutter model."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        *,
        image_sizes: tuple[int, ...] = DEFAULT_IMAGE_SIZES,
        confidence: float = DEFAULT_CONFIDENCE,
        candidate_iou: float = CANDIDATE_NMS_IOU,
        cluster_iou: float = FUSION_CLUSTER_IOU,
        device: str | int | None = None,
    ):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Full-cutter checkpoint not found: {checkpoint}")
        if not image_sizes or any(size <= 0 for size in image_sizes):
            raise ValueError("At least one positive inference image size is required.")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")
        if not 0.0 < candidate_iou <= 1.0:
            raise ValueError("Candidate IoU must be greater than 0 and at most 1.")
        if not 0.0 < cluster_iou <= 1.0:
            raise ValueError("Cluster IoU must be greater than 0 and at most 1.")

        from ultralytics import YOLO

        self.checkpoint = checkpoint
        self.image_sizes = tuple(dict.fromkeys(int(size) for size in image_sizes))
        self.confidence = float(confidence)
        self.candidate_iou = float(candidate_iou)
        self.cluster_iou = float(cluster_iou)
        self.device = device
        self.model = YOLO(str(checkpoint))
        names = getattr(self.model, "names", {})
        class_zero = names.get(0) if isinstance(names, dict) else names[0]
        if class_zero != CLASS_NAME:
            raise ValueError(
                f"Checkpoint class 0 must be {CLASS_NAME!r}; found {class_zero!r}."
            )

    @staticmethod
    def normalize_image(source: str | Path | Image.Image) -> Image.Image:
        if isinstance(source, Image.Image):
            return normalize_uploaded_image(source)
        with Image.open(source) as image:
            return normalize_uploaded_image(image)

    def predict_primary(self, source: str | Path | Image.Image) -> dict | None:
        """Return the selected primary center cutter in normalized-image pixels."""
        image = self.normalize_image(source)
        candidates = []
        speed = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
        for image_size in self.image_sizes:
            options = {
                "source": image,
                "imgsz": image_size,
                "conf": self.confidence,
                "iou": self.candidate_iou,
                "verbose": False,
            }
            if self.device is not None:
                options["device"] = self.device
            result = self.model.predict(**options)[0]
            for key in speed:
                speed[key] += float(result.speed.get(key, 0.0))
            for index, (raw_box, raw_confidence) in enumerate(
                zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()), 1
            ):
                x1, y1, x2, y2 = map(float, raw_box)
                if x2 <= x1 or y2 <= y1:
                    continue
                candidates.append({
                    "candidate": f"S{image_size}C{index}",
                    "confidence": float(raw_confidence),
                    "box": (x1, y1, x2, y2),
                })

        selected = select_center_cluster(
            candidates, image.size, cluster_iou=self.cluster_iou
        )
        if selected is None:
            return None
        x1, y1, x2, y2 = map(float, selected["box"])
        width, height = image.size
        return {
            "class_id": 0,
            "class_name": CLASS_NAME,
            "confidence": float(selected["confidence"]),
            "box_xyxy": (x1, y1, x2, y2),
            "box_xyxy_normalized": (
                x1 / width,
                y1 / height,
                x2 / width,
                y2 / height,
            ),
            "image_size": image.size,
            "candidate_count": len(candidates),
            "fusion_applied": int(selected["member_count"]) > 1,
            "fusion_source_count": int(selected["member_count"]),
            "fusion_members": tuple(selected["members"]),
            "inference_sizes": self.image_sizes,
            "speed_ms": speed,
        }
