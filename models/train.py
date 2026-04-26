"""
models/train.py
───────────────
Transfer-learning training script using Ultralytics YOLOv8.

Features:
  • Loads a pretrained YOLOv8n/s/m backbone
  • Fine-tunes on the 3-class BDD100K subset
  • Saves best checkpoint + training metrics
  • Optional resume from checkpoint

Usage:
    python models/train.py --config configs/config.yaml
    python models/train.py --config configs/config.yaml --resume
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ─── Config loader ───────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─── Main training function ──────────────────────────────────────────────────

def train(config_path: str, resume: bool = False) -> None:
    """
    Fine-tune YOLOv8 on the robotics navigation dataset.

    Args:
        config_path:  Path to configs/config.yaml
        resume:       Whether to resume from last checkpoint
    """
    cfg = load_config(config_path)
    model_cfg  = cfg["model"]
    train_cfg  = cfg["training"]
    data_cfg   = cfg["dataset"]

    # ── Auto-generate data.yaml if needed ───────────────────────────────────
    data_yaml = _ensure_data_yaml(cfg)
    logger.info(f"Using data YAML: {data_yaml}")

    # ── Device selection ─────────────────────────────────────────────────────
    device = _resolve_device(train_cfg["device"])
    logger.info(f"Training on device: {device}")

    # ── Model initialisation ─────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed.  Run: pip install ultralytics")
        sys.exit(1)

    arch = model_cfg["architecture"]          # e.g. "yolov8n"
    if resume and train_cfg.get("checkpoint_path"):
        logger.info(f"Resuming from {train_cfg['checkpoint_path']}")
        model = YOLO(train_cfg["checkpoint_path"])
    else:
        logger.info(f"Loading pretrained {arch}")
        model = YOLO(f"{arch}.pt")            # downloads weights automatically

    # ── Training ─────────────────────────────────────────────────────────────
    logger.info("Starting training …")
    results = model.train(
        data       = data_yaml,
        epochs     = train_cfg["epochs"],
        imgsz      = data_cfg["image_size"],
        batch      = train_cfg["batch_size"],
        lr0        = train_cfg["learning_rate"],
        weight_decay = train_cfg["weight_decay"],
        optimizer  = train_cfg["optimizer"],
        warmup_epochs = train_cfg["warmup_epochs"],
        device     = device,
        workers    = train_cfg["workers"],
        project    = train_cfg["save_dir"],
        name       = cfg["project"]["name"],
        exist_ok   = True,
        resume     = resume,
        seed       = cfg["project"]["seed"],
        # Augmentation
        flipud     = 0.0,
        fliplr     = 0.5,
        mosaic     = 1.0,
        mixup      = 0.1,
        # Validation
        val        = True,
        save_period= 10,
        plots      = True,
    )

    logger.info("Training complete.")
    logger.info(f"Best weights: {results.save_dir}/weights/best.pt")

    # ── Post-training summary ────────────────────────────────────────────────
    _print_metrics(results)


def _print_metrics(results) -> None:
    """Pretty-print final validation metrics."""
    try:
        metrics = results.results_dict
        print("\n" + "=" * 50)
        print("  TRAINING SUMMARY")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"  {k:<30s} {v:.4f}")
        print("=" * 50 + "\n")
    except Exception:
        pass   # metrics may not be available in all ultralytics versions


def _ensure_data_yaml(cfg: dict) -> str:
    """
    Generate configs/data.yaml from the project config if it doesn't exist.
    Falls back to looking for an existing configs/data.yaml.
    """
    yaml_path = Path("configs/data.yaml")
    if yaml_path.exists():
        return str(yaml_path)

    dataset_root = cfg["dataset"]["root"]
    train_images = f"{cfg['dataset']['images_dir']}/{cfg['dataset']['train_split']}"
    val_images   = f"{cfg['dataset']['images_dir']}/{cfg['dataset']['val_split']}"

    from dataset.bdd100k_dataset import create_yolo_data_yaml, CLASS_NAMES
    return create_yolo_data_yaml(
        dataset_root=dataset_root,
        train_images=train_images,
        val_images=val_images,
        output_yaml=str(yaml_path),
    )


def _resolve_device(device_str: str) -> str:
    """Return available device, falling back gracefully."""
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available – switching to CPU")
        return "cpu"
    if device_str == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available – switching to CPU")
        return "cpu"
    return device_str


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 on robotics nav dataset")
    p.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to project config YAML (default: configs/config.yaml)"
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint"
    )
    p.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs from config"
    )
    p.add_argument(
        "--batch", type=int, default=None,
        help="Override batch size from config"
    )
    p.add_argument(
        "--device", default=None,
        help="Override device (cuda / cpu / mps)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Apply CLI overrides to config before training
    cfg = load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch:
        cfg["training"]["batch_size"] = args.batch
    if args.device:
        cfg["training"]["device"] = args.device

    # Persist overridden config to a temp file
    tmp_cfg = "configs/_train_override.yaml"
    with open(tmp_cfg, "w") as f:
        yaml.dump(cfg, f)

    train(config_path=tmp_cfg, resume=args.resume)
