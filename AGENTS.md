# Cropper StandAlone � Full-Cutter Damage Detection Handoff

You are continuing development of an existing Python/Flask computer-vision application for PDC cutter detection and cropping.

This is an ongoing engineering project. Do not treat it as a new repository or redesign the existing architecture without first inspecting it.

## 0. Source of truth and decision authority

- Direct user instructions in the current conversation take priority over older planning notes and chatbot-generated suggestions.
- The inspected repository code, current data, generated reports, and actual test results take priority over stale documentation.
- Use older plans as historical context, not as immutable design authority. Recommend a better approach when direct code inspection supports it, while preserving approved project boundaries and user data.
- The user personally created and reviewed all 106 full-cutter annotations and accepts their semantic content as ground truth. Continue to validate file integrity, schema, coordinates, and export consistency; do not second-guess the user's annotation judgment without concrete evidence of a technical mismatch.
- Record verified findings separately from assumptions. Update this file when a completed milestone materially changes the next task.

## 0.1 Final handoff boundary

- This repository is the development, training, and validation harness for the detector models. The primary goal is to produce demonstrably accurate, reproducible model artifacts and a clear inference contract.
- The eventual production consumer is a separate Plotly Dash application. The repository owner does not have access to that application and will not provide access to it.
- Do not attempt to implement, simulate, or redesign the private Plotly Dash application. Do not make unsupported assumptions about its internal callbacks, state management, deployment, or user interface.
- Handoff work should remain framework-neutral: provide the selected checkpoint, class mapping, required image-orientation/preprocessing behavior, tested inference example, dependency versions, recommended confidence/image-size settings, expected prediction coordinates, evaluation results, and known limitations.
- Keep proprietary training images and annotation data out of the handoff repository unless the user explicitly approves sharing them.
- Integration into the production application is the receiving team's responsibility. This project's responsibility ends with validated model artifacts, portable inference guidance, and any explicitly approved standalone reference code.

## 1. Repository and environment

- Development repository: `crico98/Standalone-Crop`
- Current working branch: `experiment/full-cutter-damage-annotation`
- Local Windows repository: `C:\Users\Chris\Standalone-Crop`
- Future destination: `Ulterra-Engineering/Standalone-Crop`, after development and validation.
- Personal development computer: Windows, Python 3.12, NVIDIA RTX 3070 Ti 8GB.
- Existing stack includes Flask, Ultralytics YOLO, PyTorch, OpenCV, NumPy, and Pillow.

Verify the actual repository state, installed dependencies, and current Git branch before making changes.

## 2. Current project status

We have completed the first annotation stage for a new full-cutter damage-detection experiment.

The new annotation editor is available at:

`/annotate/full-cutter`

Its implementation includes:

- `full_cutter_web.py`
- `templates/full_cutter.html`
- `static/full_cutter.js`
- Blueprint registration in `app.py`

The editor uses images from:

`full_cutter/`

And saves annotations to:

`dataset/full_cutter_damage_v1/annotations.json`

The dataset contains:

- 106 reviewed images
- 106 annotated images
- 106 bounding boxes
- 0 skipped images
- 56 Top images
- 50 Isometric images

These figures were confirmed using PowerShell against the local annotation manifest.

The view distribution differs slightly from the original source folders (54 Top / 52 Isometric). The user explicitly approved proceeding with the saved 56/50 assignments. Do not silently change them.

The standalone YOLO exporter has now been implemented and verified:

- Exporter: `tools/export_full_cutter_damage.py`
- Export tests: `tests/test_full_cutter_export.py`
- Training entry point: `tools/train_full_cutter_damage.py`
- Training guard tests: `tests/test_full_cutter_training.py`
- Generated local export: `training_full_cutter_damage_v1/`
- Split: 85 train / 21 validation
- View-stratified counts: Top 45/11 and Isometric 40/10 train/validation
- Fixed split seed: 42
- Class mapping: class 0 = `full_cutter`
- Ultralytics successfully resolves the generated `data.yaml` paths.
- The generated contact sheet was visually inspected for EXIF orientation and box alignment.
- Source images and the source annotation manifest remain unchanged.

The proprietary source images, annotation manifest, archive, and generated export are excluded in `.gitignore`. Do not force-add them.

## 3. Detection objective

Train a new YOLO object detector using one class:

`full_cutter`

The bounding box must enclose the entire physical PDC cutter, including:

- Diamond table
- Carbide substrate
- Visible damaged or worn regions

Top and Isometric images use the same detection class.

View labels are metadata for dataset splitting and performance evaluation, not separate detection classes.

This experiment is intended to improve detection of cutters that may have chipped, worn, spalled, or partially missing diamond material.

Do not replace the existing Face detector or the older Top/Isometric detection pipeline.

## 4. Completed dataset-export milestone

The standalone YOLO dataset exporter for the completed full-cutter annotations is implemented. Preserve its isolation and safety behavior.

Before modifying the exporter:

1. Inspect the current repository, branch, and working tree.
2. Read `full_cutter_web.py` and the annotation manifest schema.
3. Inspect existing dataset-export and YOLO-training utilities for reusable patterns.
4. Identify existing tests and project conventions.
5. Report any architectural conflicts before changing code.

### Export requirements retained as the contract

Create an isolated output directory, preferably:

`training_full_cutter_damage_v1/`

Use a deterministic 80/20 train/validation split stratified by the saved Top and Isometric view metadata.

Requirements:

- One class: `full_cutter`, class ID 0.
- Convert normalized XYXY bounding boxes to YOLO normalized XYWH.
- Preserve all valid bounding boxes per image.
- Validate coordinates, dimensions, image existence, and annotation consistency.
- Use a fixed random seed and record it in the export metadata.
- Prevent train/validation image leakage.
- Generate `data.yaml` with portable dataset paths.
- Generate a split manifest identifying source images, views, and train/validation membership.
- Preserve original image filenames where possible.
- Create a visual contact sheet showing exported bounding boxes.
- Report dataset counts by view and split.
- Support a validation-only/dry-run mode.
- Do not overwrite existing exports without explicit approval.
- Do not modify the source annotation JSON or original photographs.

The expected annotation count is 106, but validate actual files rather than hardcoding that count.

### Testing requirements

Add automated tests covering:

- XYXY-to-XYWH conversion
- Bounding-box validation
- Deterministic stratified splitting
- Missing or invalid source images
- Multiple bounding boxes per image
- Output directory collision protection
- Correct class mapping
- No train/validation leakage
- Preservation of source files

Run relevant tests and report their actual results.

## 5. Current task: training and evaluation

The separate training workflow is available at `tools/train_full_cutter_damage.py`.

Use existing project training conventions where appropriate.

The generated export passed automated integrity checks, Ultralytics dataset resolution, and contact-sheet review. Training completed with early stopping after 73 epochs; the best checkpoint is epoch 48 at `models/full_cutter_damage_v1/weights/best.pt`. Preserve this checkpoint and all existing model outputs.

Default reviewed training settings are:

- Pretrained checkpoint: `yolo11n.pt`
- Image size: 1024
- Epochs: 100
- Batch: 4
- Patience: 25
- Device: CUDA GPU 0
- Workers: 0 for Windows stability
- Seed: 42
- Output: `models/full_cutter_damage_v1/`

Default command:

`python -m tools.train_full_cutter_damage`

### Verified reference inference contract

Do not apply the legacy centered Top/Isometric detection regions to this new checkpoint. Those regions retain the complete annotated cutter in only 35 of 106 images and clip 71 targets. The checkpoint was trained on complete EXIF-normalized images.

Use the framework-neutral wrapper in `full_cutter_detector.py`:

- Input: complete EXIF-normalized RGB image.
- Inference sizes: 640 and 1024.
- Confidence floor: 0.25.
- Candidate NMS IoU: 0.99, retaining overlapping candidates.
- Combine candidates from both inference sizes in full-image pixel coordinates.
- Cluster candidates at IoU 0.60 and confidence-weight their coordinates.
- Select the fused cluster nearest the full-image center; confidence breaks only exact center-distance ties.
- Output class: class 0, `full_cutter`.

Reproducible evaluation command:

`python -m tools.evaluate_full_cutter_damage`

The saved report is `models/full_cutter_damage_v1/full_image_multiscale_evaluation.json`. On the fixed 21-image validation split, this contract selected 21/21 targets at IoU 0.50 or better: Top 11/11, Isometric 10/10, mean IoU 0.8921, median IoU 0.9032, and minimum IoU 0.7649. This is strong internal validation, not proof of performance on an independent production dataset.

Training should support the local NVIDIA RTX 3070 Ti and preserve existing checkpoints.

Model output directories must be unique to this experiment.

Future evaluation should include:

- Overall precision and recall
- mAP50 and mAP50-95
- Top-view performance
- Isometric-view performance
- Detection quality on visibly damaged cutters
- Visual inspection of false positives and false negatives

Do not claim damage-specific performance without appropriate evaluation labels or a documented review methodology.

## 6. Critical development rules

These rules take priority over convenience.

**Repository inspection**

- Inspect existing code before proposing modifications.
- Reuse established project architecture where practical.
- Do not assume earlier implementation details remain unchanged.
- Distinguish verified findings from assumptions.

**Data protection**

- Never delete, reset, clean, or overwrite uncommitted work.
- Never overwrite original annotation manifests.
- Never overwrite existing YOLO exports or trained checkpoints.
- Keep source images unchanged.
- Do not commit large image datasets, model weights, or generated outputs without explicit approval.
- Preserve the original `/annotate` editor and its associated dataset.

**Git workflow**

- Remain on `experiment/full-cutter-damage-annotation` unless explicitly instructed otherwise.
- Do not merge into `main`.
- Do not force-push.
- Do not commit unrelated changes.
- Review `git status` before and after changes.
- Obtain approval before destructive operations.

**Engineering quality**

- Favor root-cause solutions over temporary workarounds.
- Avoid unnecessary architectural complexity.
- Keep experimental components isolated.
- Write readable, maintainable Python.
- Add tests for new behavior.
- Do not claim code was tested unless tests actually ran.

**Communication**

- The user understands the engineering goals but prefers clear explanations over raw code dumps.
- Explain what was found, what is changing, and why.
- When giving PowerShell instructions, provide one command at a time and wait for the output.
- Ask questions only when a missing decision materially affects implementation.
- Do not repeatedly ask for information already available in the repository or this handoff.

## 7. Important project boundaries

The existing application includes other cropping and detection workflows.

Do not modify the Face detection model, original annotation exporter, or unrelated image-processing logic as part of this task.

The new full-cutter experiment must remain independently reproducible.

## 8. Deliverable status

Completed:

1. A standalone full-cutter YOLO exporter.
2. A validated dataset split.
3. A YOLO `data.yaml`.
4. A machine-readable split manifest.
5. A bounding-box visual review sheet.
6. Automated tests.
7. A concise export report.
8. Training instructions for the personal GPU computer.

The trained checkpoint, training metrics, per-view internal evaluation, framework-neutral inference wrapper, and machine-readable evaluation report now exist. Remaining handoff work is independent-image validation when representative unseen photos are available, concise receiving-team documentation, and an explicit decision about distributing the checkpoint through Git LFS or an approved external channel. Do not claim damage classification or damage grading performance; this remains a one-class center-cutter detector trained on damage-rich imagery.

## 9. Persistent project instructions

Check whether the repository already contains an `AGENTS.md` file.

If one exists, read it and preserve its existing instructions.

If none exists, propose a root-level `AGENTS.md` containing the durable development rules above.

Do not overwrite existing project instructions.Unless they are meant to make our code better

The goal is to make future Codex sessions consistent with this project's established architecture and data-protection requirements.

Implement one ticket only.
Do not implement future-ticket features.
Do not refactor unrelated systems.
Do not introduce new architecture unless required.
Avoid unnecessary dependencies.
Use strict TypeScript.
Run build/tests when possible.
Report files changed, commands run, build results, and manual verification steps.
