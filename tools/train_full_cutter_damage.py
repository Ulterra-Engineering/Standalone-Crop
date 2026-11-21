from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "training_full_cutter_damage_v1" / "data.yaml"
DEFAULT_PROJECT = ROOT / "models"
DEFAULT_NAME = "full_cutter_damage_v1"


def validate_training_paths(data_yaml: Path, project: Path, name: str) -> Path:
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset config not found: {data_yaml}. Export and review the dataset before training."
        )
    output = project / name
    if output.exists():
        raise FileExistsError(
            f"Training output already exists: {output}. Choose a new --name; checkpoints are never overwritten."
        )
    return output


def train(args) -> None:
    output = validate_training_paths(args.data, args.project, args.name)

    # Importing Ultralytics can initialize CUDA, so keep it out of validation
    # and command-help paths.
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    dataset = check_det_dataset(str(args.data), autodownload=False)
    names = dataset.get("names", {})
    if names != {0: "full_cutter"}:
        raise ValueError(f"Dataset must contain only class 0 'full_cutter'; found {names!r}.")

    print("Training unified full-cutter damage detector")
    print(f"  data: {args.data.resolve()}")
    print(f"  pretrained checkpoint: {args.model}")
    print(f"  output: {output.resolve()}")
    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the unified one-class full-cutter detector on the reviewed damage dataset."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained Ultralytics detection checkpoint.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--device", default="0", help="CUDA device index such as 0, or cpu.")
    parser.add_argument("--workers", type=int, default=0, help="Zero is the Windows-safe default.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
