from __future__ import annotations

import math
from pathlib import Path

from flask import Blueprint, abort, jsonify, render_template, request, send_file
from PIL import Image

ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "full_cutter"
DATA_ROOT = ROOT / "dataset" / "full_cutter_damage_v1"
MANIFEST_PATH = DATA_ROOT / "annotations.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

full_cutter = Blueprint("full_cutter", __name__)


def _load_manifest() -> dict:
    import json

    if not MANIFEST_PATH.is_file():
        return {"version": 1, "class_name": "full_cutter", "annotations": {}}
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read {MANIFEST_PATH}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("annotations"), dict):
        raise RuntimeError(f"Invalid annotation manifest: {MANIFEST_PATH}")
    return data


def _save_manifest(manifest: dict) -> None:
    import json

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(MANIFEST_PATH)


def _images() -> list[Path]:
    if not IMAGE_ROOT.is_dir():
        return []
    return sorted(
        (path for path in IMAGE_ROOT.rglob("*")
         if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.relative_to(IMAGE_ROOT).as_posix().casefold(),
    )


def _image_path(key: str) -> Path:
    if not isinstance(key, str) or not key:
        abort(400, "An image key is required.")
    root = IMAGE_ROOT.resolve()
    path = (root / key).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        abort(400, "Invalid image key.")
    if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
        abort(404, "Image not found.")
    return path


@full_cutter.errorhandler(400)
def bad_request(error):
    return jsonify({"error": getattr(error, "description", "Bad request.")}), 400


@full_cutter.errorhandler(404)
def not_found(error):
    return jsonify({"error": getattr(error, "description", "Not found.")}), 404


@full_cutter.get("/annotate/full-cutter")
def editor():
    return render_template("full_cutter.html")


@full_cutter.get("/api/full-cutter/images")
def image_index():
    images = _images()
    manifest = _load_manifest()
    results = []
    for path in images:
        key = path.relative_to(IMAGE_ROOT).as_posix()
        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception:
            continue
        record = manifest["annotations"].get(key, {})
        results.append({
            "key": key,
            "name": path.name,
            "width": width,
            "height": height,
            "view": record.get("view", ""),
            "boxes": record.get("boxes", []),
            "status": record.get("status", "pending"),
            "annotated": record.get("status") == "annotated" and bool(record.get("boxes")),
            "reviewed": record.get("status") in {"annotated", "skip"},
        })
    return jsonify({
        "images": results,
        "error": None if images else f"No images found in {IMAGE_ROOT}.",
        "manifest": str(MANIFEST_PATH),
        "class_name": "full_cutter",
    })


@full_cutter.get("/api/full-cutter/image")
def image_file():
    return send_file(_image_path(request.args.get("key", "")), max_age=0)


@full_cutter.post("/api/full-cutter/save")
def save_annotation():
    payload = request.get_json(silent=True) or {}
    key = payload.get("key")
    _image_path(key)
    view = payload.get("view", "")
    status = payload.get("status", "annotated")
    raw_boxes = payload.get("boxes", [])

    if view not in {"top", "isometric"}:
        abort(400, "Assign this image as Top or Isometric before saving.")
    if status not in {"annotated", "skip"}:
        abort(400, "Status must be annotated or skip.")
    if not isinstance(raw_boxes, list):
        abort(400, "Boxes must be a list.")

    boxes = []
    for raw in raw_boxes:
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            abort(400, "Each box must contain four normalized coordinates.")
        try:
            values = [float(value) for value in raw]
        except (TypeError, ValueError):
            abort(400, "Box coordinates must be numeric.")
        if not all(math.isfinite(value) for value in values):
            abort(400, "Box coordinates must be finite.")
        x1, y1, x2, y2 = (min(1.0, max(0.0, value)) for value in values)
        if x2 - x1 < 0.003 or y2 - y1 < 0.003:
            abort(400, "Each box must have a visible width and height.")
        boxes.append([x1, y1, x2, y2])

    if status == "annotated" and not boxes:
        abort(400, "Draw at least one full-cutter box, or mark the image Skip.")
    if status == "skip":
        boxes = []

    manifest = _load_manifest()
    manifest["version"] = 1
    manifest["class_name"] = "full_cutter"
    manifest["annotations"][key] = {
        "view": view,
        "status": status,
        "boxes": boxes,
        "target": "full_cutter",
    }
    _save_manifest(manifest)
    reviewed = sum(
        record.get("status") in {"annotated", "skip"}
        for record in manifest["annotations"].values()
    )
    return jsonify({
        "saved": True,
        "key": key,
        "status": status,
        "box_count": len(boxes),
        "reviewed_count": reviewed,
        "manifest": str(MANIFEST_PATH),
    })
