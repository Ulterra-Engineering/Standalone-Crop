import tempfile
import unittest
from pathlib import Path

from tools.train_full_cutter_damage import build_parser, validate_training_paths


class FullCutterTrainingTests(unittest.TestCase):
    def test_defaults_match_reviewed_local_gpu_convention(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.model, "yolo11n.pt")
        self.assertEqual(args.imgsz, 1024)
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.batch, 4)
        self.assertEqual(args.patience, 25)
        self.assertEqual(args.device, "0")
        self.assertEqual(args.workers, 0)
        self.assertEqual(args.seed, 42)

    def test_missing_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                validate_training_paths(root / "missing.yaml", root / "models", "run")

    def test_existing_checkpoint_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data.yaml"
            data.write_text("names: {0: full_cutter}\n", encoding="utf-8")
            output = root / "models" / "run"
            output.mkdir(parents=True)
            sentinel = output / "best.pt"
            sentinel.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                validate_training_paths(data, root / "models", "run")
            self.assertEqual(sentinel.read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
