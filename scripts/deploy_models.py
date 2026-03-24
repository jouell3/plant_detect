"""
Deploy model files to Google Cloud Storage.

PyTorch models          → gs://<bucket>/models/
Sklearn models          → gs://<bucket>/models_sklearn/
TensorFlow models       → gs://<bucket>/models_tensorflow/
PyTorch large models    → gs://<bucket>/models_pytorch_large/

For sklearn, the most recent timestamped files are selected and uploaded
under the canonical names expected by model_sklearn.py:
  - config_sklearn.json
  - label_encoder_sklearn.pkl
  - efficientnet_b3__logistic_regression.pkl

Usage:
    python scripts/deploy_models.py [--bucket BUCKET] [--project PROJECT] [--dry-run]

Environment variables (used as defaults):
    GCS_BUCKET_NAME   bucket name            (default: plant-detect-models)
    GCS_PROJECT       GCP project ID         (default: bootcamparomatic)
"""

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve repo root (script lives in <root>/scripts/)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "backend" / "app" / "models"
MODELS_SKLEARN_DIR = REPO_ROOT / "backend" / "app" / "models_sklearn"
MODELS_TENSORFLOW_DIR = REPO_ROOT / "backend" / "app" / "models_tensorflow"
MODELS_PYTORCH_LARGE_DIR = REPO_ROOT / "backend" / "app" / "models_pytorch_large"

# ---------------------------------------------------------------------------
# Files to upload for PyTorch model (path → GCS blob name)
# ---------------------------------------------------------------------------
PYTORCH_FILES = {
    MODELS_DIR / "resnet18_plants.pt": "models/resnet18_plants.pt",
    MODELS_DIR / "label_encoder.pkl": "models/label_encoder.pkl",
}


def _pick_latest(directory: Path, pattern: str) -> Path:
    """Return the most recent file matching pattern (sorted by name)."""
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {directory}")
    return matches[-1]


def build_sklearn_files(sklearn_dir: Path) -> dict[Path, str]:
    """Resolve most-recent sklearn files → canonical GCS blob names."""
    config = _pick_latest(sklearn_dir, "config_sklearn__*.json")
    encoder = _pick_latest(sklearn_dir, "label_encoder_sklearn__*.pkl")
    pipeline = _pick_latest(sklearn_dir, "efficientnet_b3__logistic_regression__*.pkl")

    return {
        config:   f"models_sklearn/{config.name}",
        encoder:  f"models_sklearn/{encoder.name}",
        pipeline: f"models_sklearn/{pipeline.name}",
    }


def build_dir_files(directory: Path, gcs_folder: str) -> dict[Path, str]:
    """Return all files in directory mapped to their GCS blob name under gcs_folder."""
    return {
        f: f"{gcs_folder}/{f.name}"
        for f in sorted(directory.iterdir())
        if f.is_file()
    }


def upload(bucket, local_path: Path, blob_name: str, dry_run: bool) -> None:
    size_mb = local_path.stat().st_size / 1_048_576
    print(f"  {'[DRY-RUN] ' if dry_run else ''}uploading {local_path.name} ({size_mb:.1f} MB) → gs://{bucket.name}/{blob_name}")
    if not dry_run:
        bucket.blob(blob_name).upload_from_filename(str(local_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy model files to GCS")
    parser.add_argument("--bucket",  default=os.getenv("GCS_BUCKET_NAME", "plant-detect-models"))
    parser.add_argument("--project", default=os.getenv("GCS_PROJECT", "bootcamparomatic"))
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded without uploading")
    args = parser.parse_args()

    # Build upload manifest
    try:
        sklearn_files = build_sklearn_files(MODELS_SKLEARN_DIR)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    tensorflow_files = build_dir_files(MODELS_TENSORFLOW_DIR, "models_tensorflow")
    pytorch_large_files = build_dir_files(MODELS_PYTORCH_LARGE_DIR, "models_pytorch_large")

    all_files = {**PYTORCH_FILES, **sklearn_files, **tensorflow_files, **pytorch_large_files}

    # Validate local files exist
    missing = [p for p in all_files if not p.exists()]
    if missing:
        print("ERROR: missing local files:", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)

    print(f"\nBucket  : gs://{args.bucket}")
    print(f"Project : {args.project}")
    print(f"Files   : {len(all_files)}\n")

    if not args.dry_run:
        from google.cloud import storage
        client = storage.Client(project=args.project)
        bucket = client.bucket(args.bucket)
    else:
        bucket = type("Bucket", (), {"name": args.bucket})()

    print("PyTorch model:")
    for local_path, blob_name in PYTORCH_FILES.items():
        upload(bucket, local_path, blob_name, args.dry_run)

    print("\nSklearn model (most recent files):")
    for local_path, blob_name in sklearn_files.items():
        upload(bucket, local_path, blob_name, args.dry_run)

    print("\nTensorFlow model:")
    for local_path, blob_name in tensorflow_files.items():
        upload(bucket, local_path, blob_name, args.dry_run)

    print("\nPyTorch large model:")
    for local_path, blob_name in pytorch_large_files.items():
        upload(bucket, local_path, blob_name, args.dry_run)

    print("\nDone." if not args.dry_run else "\nDry run complete — nothing was uploaded.")


if __name__ == "__main__":
    main()
