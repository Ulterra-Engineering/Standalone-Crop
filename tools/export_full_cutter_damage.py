from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from image_orientation import normalize_uploaded_image, read_exif_orientation


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES = ROOT / "full_cutter"
DEFAULT_ANNOTATIONS = ROOT / "dataset" / "full_cutter_damage_v1" / "annotations.json"
DEFAULT_OUTPUT = ROOT / "training_full_cutter_damage_v1"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_ID = 0
CLASS_NAME = "full_cutter"
VALID_VIEWS = {"top", "isometric"}


@dataclass(frozen=True)
class ExportRecord:
    key: str
    source_path: Path
    output_name: str
    view: str
    boxes: tuple[tuple[float, float, float, float], ...]
    source_width: int
    source_height: int
    width: int
    height: int
    exif_orientation: int
    source_sha256: str
    split: str = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xyxy_to_xywh(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1


def validate_box(raw_box, *, key: str, index: int) -> tuple[float, float, float, float]:
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        raise ValueError(f"{key}: box {index} must contain exactly four XYXY coordinates.")
    try:
        box = tuple(float(value) for value in raw_box)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key}: box {index} coordinates must be numeric.") from exc
    if not all(math.isfinite(value) for value in box):
        raise ValueError(f"{key}: box {index} coordinates must be finite.")
    if not all(0.0 <= value <= 1.0 for value in box):
        raise ValueError(f"{key}: box {index} coordinates must be normalized to [0, 1].")
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{key}: box {index} must have positive width and height.")
    return box


def _source_images(image_root: Path) -> dict[str, Path]:
    if not image_root.is_dir():
        raise FileNotFoundError(f"Source image directory not found: {image_root}")
    images: dict[str, Path] = {}
    casefolded: dict[str, str] = {}
    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.relative_to(image_root).as_posix()
        folded = key.casefold()
        if folded in casefolded:
            raise ValueError(f"Source image keys differ only by case: {casefolded[folded]!r} and {key!r}.")
        casefolded[folded] = key
        images[key] = path
    if not images:
        raise ValueError(f"No supported source images found in: {image_root}")
    return images


def _output_names(keys: list[str]) -> dict[str, str]:
    """Preserve basenames unless that would collide in the flat YOLO folders."""
    basenames = Counter(Path(key).name.casefold() for key in keys)
    stems = Counter(Path(key).stem.casefold() for key in keys)
    result = {}
    for key in keys:
        path = Path(key)
        if basenames[path.name.casefold()] == 1 and stems[path.stem.casefold()] == 1:
            result[key] = path.name
            continue
        readable = "_".join(part for part in path.with_suffix("").parts if part not in {".", ".."})
        safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in readable)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
        result[key] = f"{safe[:90] or 'image'}_{digest}{path.suffix.lower()}"
    return result


def load_and_validate_records(image_root: Path, annotations_path: Path) -> list[ExportRecord]:
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Annotation manifest not found: {annotations_path}")
    try:
        manifest = json.loads(annotations_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Annotation manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("annotations"), dict):
        raise ValueError("Annotation manifest must contain an annotations object.")
    if manifest.get("class_name") != CLASS_NAME:
        raise ValueError(f"Manifest class_name must be {CLASS_NAME!r}.")

    sources = _source_images(image_root)
    annotations = manifest["annotations"]
    source_keys = set(sources)
    annotation_keys = set(annotations)
    missing = sorted(annotation_keys - source_keys)
    unreviewed = sorted(source_keys - annotation_keys)
    if missing:
        raise FileNotFoundError("Annotated source images are missing: " + ", ".join(missing))
    if unreviewed:
        raise ValueError("Source images have no annotation record: " + ", ".join(unreviewed))

    export_keys = []
    validated: dict[str, tuple[str, tuple[tuple[float, float, float, float], ...]]] = {}
    for key in sorted(annotations, key=str.casefold):
        record = annotations[key]
        if not isinstance(record, dict):
            raise ValueError(f"{key}: annotation record must be an object.")
        view = record.get("view")
        if view not in VALID_VIEWS:
            raise ValueError(f"{key}: view must be 'top' or 'isometric'.")
        status = record.get("status")
        boxes = record.get("boxes")
        if status == "skip":
            if boxes:
                raise ValueError(f"{key}: skipped records cannot contain boxes.")
            continue
        if status != "annotated":
            raise ValueError(f"{key}: status must be 'annotated' or 'skip'.")
        if record.get("target") != CLASS_NAME:
            raise ValueError(f"{key}: target must be {CLASS_NAME!r}.")
        if not isinstance(boxes, list) or not boxes:
            raise ValueError(f"{key}: annotated records must contain at least one box.")
        normalized = tuple(
            validate_box(box, key=key, index=index)
            for index, box in enumerate(boxes, 1)
        )
        export_keys.append(key)
        validated[key] = view, normalized
    if not export_keys:
        raise ValueError("The manifest contains no annotated full-cutter images to export.")

    names = _output_names(export_keys)
    records = []
    for key in export_keys:
        path = sources[key]
        try:
            with Image.open(path) as source:
                source.load()
                source_width, source_height = source.size
                orientation = read_exif_orientation(source)
                normalized = normalize_uploaded_image(source)
                width, height = normalized.size
        except Exception as exc:
            raise ValueError(f"Could not decode source image {key}: {exc}") from exc
        if min(source_width, source_height, width, height) <= 0:
            raise ValueError(f"{key}: image dimensions must be positive.")
        view, boxes = validated[key]
        records.append(ExportRecord(
            key=key,
            source_path=path,
            output_name=names[key],
            view=view,
            boxes=boxes,
            source_width=source_width,
            source_height=source_height,
            width=width,
            height=height,
            exif_orientation=orientation,
            source_sha256=sha256_file(path),
        ))
    return records


def stratified_split(
    records: list[ExportRecord], val_fraction: float = 0.2, seed: int = 42
) -> list[ExportRecord]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("Validation fraction must be between 0 and 1.")
    assigned = []
    for view in sorted(VALID_VIEWS):
        group = [record for record in records if record.view == view]
        if len(group) < 2:
            raise ValueError(f"At least two {view} images are required for a train/validation split.")
        ordered = sorted(
            group,
            key=lambda record: hashlib.sha256(
                f"{seed}:{view}:{record.key}".encode("utf-8")
            ).digest(),
        )
        val_count = max(1, min(len(group) - 1, round(len(group) * val_fraction)))
        val_keys = {record.key for record in ordered[:val_count]}
        assigned.extend(
            ExportRecord(**{**record.__dict__, "split": "val" if record.key in val_keys else "train"})
            for record in group
        )
    if len({record.key for record in assigned}) != len(records):
        raise ValueError("Train/validation assignment contains missing or duplicate source keys.")
    return sorted(assigned, key=lambda record: record.key.casefold())


def _save_normalized_image(source_path: Path, destination: Path) -> None:
    with Image.open(source_path) as source:
        image = normalize_uploaded_image(source)
    suffix = destination.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(destination, format="JPEG", quality=95, subsampling=0)
    elif suffix == ".png":
        image.save(destination, format="PNG", optimize=True)
    elif suffix == ".webp":
        image.save(destination, format="WEBP", quality=95)
    elif suffix == ".bmp":
        image.save(destination, format="BMP")
    elif suffix in {".tif", ".tiff"}:
        image.save(destination, format="TIFF")
    else:
        raise ValueError(f"Unsupported output image extension: {destination.suffix}")


def _label_text(record: ExportRecord) -> str:
    rows = []
    for box in record.boxes:
        xc, yc, width, height = xyxy_to_xywh(box)
        rows.append(f"{CLASS_ID} {xc:.8f} {yc:.8f} {width:.8f} {height:.8f}")
    return "\n".join(rows) + "\n"


def _manifest_record(record: ExportRecord) -> dict:
    return {
        "source_key": record.key,
        "source_sha256": record.source_sha256,
        "exported_image": f"images/{record.split}/{record.output_name}",
        "exported_label": f"labels/{record.split}/{Path(record.output_name).stem}.txt",
        "view": record.view,
        "split": record.split,
        "source_dimensions": [record.source_width, record.source_height],
        "exported_dimensions": [record.width, record.height],
        "source_exif_orientation": record.exif_orientation,
        "boxes_xyxy_normalized": [list(box) for box in record.boxes],
        "boxes_xywh_normalized": [list(xyxy_to_xywh(box)) for box in record.boxes],
    }


def build_report(records: list[ExportRecord], annotations_path: Path, seed: int, val_fraction: float) -> dict:
    split_counts = Counter(record.split for record in records)
    view_counts = Counter(record.view for record in records)
    by_view_split = {
        view: {
            split: sum(record.view == view and record.split == split for record in records)
            for split in ("train", "val")
        }
        for view in sorted(VALID_VIEWS)
    }
    return {
        "dataset": "full_cutter_damage_v1",
        "source_annotations": str(annotations_path.resolve()),
        "class_mapping": {str(CLASS_ID): CLASS_NAME},
        "seed": seed,
        "validation_fraction": val_fraction,
        "split_algorithm": "per-view SHA-256 ordering of seed:view:source_key",
        "image_count": len(records),
        "box_count": sum(len(record.boxes) for record in records),
        "split_counts": dict(sorted(split_counts.items())),
        "view_counts": dict(sorted(view_counts.items())),
        "by_view_and_split": by_view_split,
        "source_files_modified": False,
    }


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def create_contact_sheet(records: list[ExportRecord], destination: Path) -> None:
    columns, tile_width, tile_height = 4, 360, 270
    rows = math.ceil(len(records) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "#11161b")
    font = _font(16)
    small_font = _font(13)
    for index, record in enumerate(records):
        with Image.open(record.source_path) as source:
            image = normalize_uploaded_image(source)
        draw = ImageDraw.Draw(image)
        color = "#33e6ff" if record.split == "train" else "#ffe14a"
        line_width = max(4, round(max(image.size) / 800))
        for box in record.boxes:
            x1, y1, x2, y2 = box
            draw.rectangle(
                (x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height),
                outline=color,
                width=line_width,
            )
        image.thumbnail((tile_width - 12, tile_height - 48), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_width, tile_height), "#1b2229")
        tile.paste(image, ((tile_width - image.width) // 2, 34 + (tile_height - 42 - image.height) // 2))
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.text((8, 5), record.output_name, fill="white", font=font)
        tile_draw.text(
            (8, 22), f"{record.view.upper()} | {record.split.upper()} | {len(record.boxes)} box(es)",
            fill=color,
            font=small_font,
        )
        sheet.paste(tile, ((index % columns) * tile_width, (index // columns) * tile_height))
    sheet.save(destination, format="JPEG", quality=92)


def export_dataset(
    image_root: Path,
    annotations_path: Path,
    output_root: Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
    validate_only: bool = False,
) -> dict:
    records = stratified_split(
        load_and_validate_records(image_root, annotations_path),
        val_fraction=val_fraction,
        seed=seed,
    )
    report = build_report(records, annotations_path, seed, val_fraction)
    if validate_only:
        return report
    if output_root.exists():
        raise FileExistsError(
            f"Output already exists: {output_root}. Choose a new output directory; existing exports are never overwritten."
        )

    temporary = output_root.with_name(f".{output_root.name}.building")
    if temporary.exists():
        raise FileExistsError(f"Temporary export directory already exists: {temporary}")
    try:
        for split in ("train", "val"):
            (temporary / "images" / split).mkdir(parents=True)
            (temporary / "labels" / split).mkdir(parents=True)
        for record in records:
            image_path = temporary / "images" / record.split / record.output_name
            label_path = temporary / "labels" / record.split / f"{Path(record.output_name).stem}.txt"
            _save_normalized_image(record.source_path, image_path)
            label_path.write_text(_label_text(record), encoding="utf-8")
        (temporary / "data.yaml").write_text(
            # With no explicit ``path``, Ultralytics resolves these portable
            # relative paths from the directory containing this YAML file.
            "train: images/train\nval: images/val\nnames:\n  0: full_cutter\n",
            encoding="utf-8",
        )
        split_manifest = {
            "version": 1,
            "dataset": report["dataset"],
            "class_mapping": report["class_mapping"],
            "seed": seed,
            "validation_fraction": val_fraction,
            "split_algorithm": report["split_algorithm"],
            "images": [_manifest_record(record) for record in records],
        }
        (temporary / "split_manifest.json").write_text(
            json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "export_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        create_contact_sheet(records, temporary / "contact_sheet.jpg")
        temporary.replace(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    # Prove that export did not alter any source file.
    for record in records:
        if sha256_file(record.source_path) != record.source_sha256:
            raise RuntimeError(f"Source image changed during export: {record.source_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and export the full-cutter damage annotations as one YOLO detection dataset."
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    report = export_dataset(
        args.images,
        args.annotations,
        args.output,
        val_fraction=args.val_fraction,
        seed=args.seed,
        validate_only=args.validate_only,
    )
    action = "Validation passed" if args.validate_only else "Dataset exported"
    print(f"{action}: {args.output.resolve()}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
