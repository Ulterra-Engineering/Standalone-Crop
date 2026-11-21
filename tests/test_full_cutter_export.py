import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.export_full_cutter_damage import (
    export_dataset,
    load_and_validate_records,
    stratified_split,
    validate_box,
    xyxy_to_xywh,
)


class FullCutterExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.annotations = self.root / "annotations.json"
        self.output = self.root / "output"

    def tearDown(self):
        self.temporary.cleanup()

    def make_image(self, name, color, orientation=1):
        path = self.images / name
        image = Image.new("RGB", (40, 20), color)
        exif = Image.Exif()
        exif[274] = orientation
        image.save(path, exif=exif)
        return path

    def write_manifest(self, records):
        self.annotations.write_text(
            json.dumps({"version": 1, "class_name": "full_cutter", "annotations": records}),
            encoding="utf-8",
        )

    @staticmethod
    def record(view, boxes):
        return {
            "view": view,
            "status": "annotated",
            "boxes": boxes,
            "target": "full_cutter",
        }

    def make_balanced_dataset(self):
        records = {}
        for view, prefix, base in (("top", "top", 20), ("isometric", "iso", 100)):
            for index in range(5):
                name = f"{prefix}_{index}.jpg"
                self.make_image(name, (base + index, 10, 20), orientation=6 if index == 0 else 1)
                records[name] = self.record(view, [[0.1, 0.2, 0.8, 0.9]])
        self.write_manifest(records)
        return records

    def test_xyxy_to_xywh_conversion(self):
        actual = xyxy_to_xywh((0.1, 0.2, 0.7, 0.8))
        for value, expected in zip(actual, (0.4, 0.5, 0.6, 0.6)):
            self.assertAlmostEqual(value, expected)

    def test_box_validation_rejects_bad_coordinates(self):
        invalid = (
            [0.1, 0.2, 0.3],
            [0.1, 0.2, "bad", 0.8],
            [0.1, 0.2, float("nan"), 0.8],
            [-0.1, 0.2, 0.7, 0.8],
            [0.7, 0.2, 0.1, 0.8],
        )
        for box in invalid:
            with self.subTest(box=box), self.assertRaises(ValueError):
                validate_box(box, key="sample.jpg", index=1)

    def test_split_is_deterministic_stratified_and_has_no_leakage(self):
        self.make_balanced_dataset()
        records = load_and_validate_records(self.images, self.annotations)
        first = stratified_split(records, seed=42)
        second = stratified_split(records, seed=42)
        self.assertEqual(
            [(record.key, record.split) for record in first],
            [(record.key, record.split) for record in second],
        )
        train = {record.key for record in first if record.split == "train"}
        val = {record.key for record in first if record.split == "val"}
        self.assertFalse(train & val)
        self.assertEqual(len(train | val), 10)
        self.assertEqual(
            {(view, split): sum(r.view == view and r.split == split for r in first)
             for view in ("top", "isometric") for split in ("train", "val")},
            {("top", "train"): 4, ("top", "val"): 1,
             ("isometric", "train"): 4, ("isometric", "val"): 1},
        )

    def test_missing_source_image_is_rejected(self):
        self.make_image("present.jpg", (10, 20, 30))
        self.write_manifest({"missing.jpg": self.record("top", [[0.1, 0.2, 0.8, 0.9]])})
        with self.assertRaises(FileNotFoundError):
            load_and_validate_records(self.images, self.annotations)

    def test_invalid_source_image_is_rejected(self):
        (self.images / "bad.jpg").write_bytes(b"not an image")
        self.write_manifest({"bad.jpg": self.record("top", [[0.1, 0.2, 0.8, 0.9]])})
        with self.assertRaisesRegex(ValueError, "Could not decode"):
            load_and_validate_records(self.images, self.annotations)

    def test_multiple_boxes_and_class_mapping_are_exported(self):
        records = self.make_balanced_dataset()
        records["top_0.jpg"]["boxes"] = [[0.1, 0.2, 0.5, 0.6], [0.55, 0.1, 0.9, 0.5]]
        self.write_manifest(records)
        report = export_dataset(self.images, self.annotations, self.output)
        manifest = json.loads((self.output / "split_manifest.json").read_text(encoding="utf-8"))
        item = next(item for item in manifest["images"] if item["source_key"] == "top_0.jpg")
        label = (self.output / item["exported_label"]).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(label), 2)
        self.assertTrue(all(row.startswith("0 ") for row in label))
        self.assertEqual(report["class_mapping"], {"0": "full_cutter"})
        data_yaml = (self.output / "data.yaml").read_text(encoding="utf-8")
        self.assertIn("0: full_cutter", data_yaml)
        self.assertNotIn("path:", data_yaml)

    def test_existing_output_is_never_overwritten(self):
        self.make_balanced_dataset()
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            export_dataset(self.images, self.annotations, self.output)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_validate_only_writes_nothing(self):
        self.make_balanced_dataset()
        report = export_dataset(
            self.images, self.annotations, self.output, validate_only=True
        )
        self.assertEqual(report["image_count"], 10)
        self.assertFalse(self.output.exists())

    def test_source_files_are_preserved_and_exif_pixels_are_normalized(self):
        self.make_balanced_dataset()
        source = self.images / "top_0.jpg"
        before = source.read_bytes()
        export_dataset(self.images, self.annotations, self.output)
        self.assertEqual(source.read_bytes(), before)
        manifest = json.loads((self.output / "split_manifest.json").read_text(encoding="utf-8"))
        item = next(item for item in manifest["images"] if item["source_key"] == "top_0.jpg")
        with Image.open(self.output / item["exported_image"]) as exported:
            self.assertEqual(exported.size, (20, 40))
            self.assertEqual(exported.getexif().get(274, 1), 1)


if __name__ == "__main__":
    unittest.main()
