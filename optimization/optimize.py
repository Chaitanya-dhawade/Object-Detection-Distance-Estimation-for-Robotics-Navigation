"""
optimization/optimize.py
─────────────────────────
Model optimization pipeline for edge/robotics deployment.

Techniques applied:
  1. ONNX export       – framework-independent serialisation
  2. INT8 quantization – via ONNX Runtime dynamic quantization
  3. FP16 quantization – via PyTorch / Ultralytics built-in
  4. L1 unstructured pruning – removes low-magnitude weights
  5. Size & latency comparison report

Usage:
    python optimization/optimize.py --weights best.pt --config configs/config.yaml
    python optimization/optimize.py --weights best.pt --mode fp16
    python optimization/optimize.py --weights best.pt --mode prune --sparsity 0.3
"""

import argparse
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.utils.prune as torch_prune
import yaml

logger = logging.getLogger("optimization")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─── File size helper ────────────────────────────────────────────────────────

def file_size_mb(path: str) -> float:
    """Return file size in megabytes."""
    return Path(path).stat().st_size / (1024 ** 2)


# ─── ONNX export ─────────────────────────────────────────────────────────────

def export_to_onnx(
    weights: str,
    output_path: str,
    img_size: int = 640,
    opset: int = 17,
    dynamic: bool = True,
) -> str:
    """
    Export YOLOv8 .pt model to ONNX.

    Args:
        weights:     Path to .pt checkpoint
        output_path: Where to save the .onnx file
        img_size:    Model input size
        opset:       ONNX opset version
        dynamic:     Enable dynamic batch / spatial axes

    Returns:
        Path to the exported .onnx file
    """
    logger.info(f"Exporting {weights} → ONNX …")
    try:
        from ultralytics import YOLO
        model = YOLO(weights)
        exported = model.export(
            format  = "onnx",
            imgsz   = img_size,
            opset   = opset,
            dynamic = dynamic,
            simplify= True,
        )
        # Ultralytics puts the onnx next to the pt by default
        default_onnx = str(Path(weights).with_suffix(".onnx"))
        if output_path != default_onnx:
            shutil.copy(default_onnx, output_path)
        logger.info(f"ONNX exported → {output_path}  ({file_size_mb(output_path):.1f} MB)")
        return output_path
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        raise


# ─── INT8 Quantization (ONNX Runtime) ────────────────────────────────────────

def quantize_int8_onnx(onnx_path: str, output_path: str) -> str:
    """
    Apply dynamic INT8 quantization to an ONNX model using
    onnxruntime.quantization.

    Dynamic quantization does NOT require a calibration dataset –
    weights are quantized statically and activations dynamically.

    Args:
        onnx_path:   Float32 ONNX model path
        output_path: Destination for the INT8 model

    Returns:
        Path to the quantised model.
    """
    logger.info(f"Quantizing {onnx_path} → INT8 …")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            model_input  = onnx_path,
            model_output = output_path,
            weight_type  = QuantType.QInt8,
        )
        logger.info(
            f"INT8 model saved → {output_path}  "
            f"({file_size_mb(output_path):.1f} MB)"
        )
        return output_path
    except ImportError:
        logger.error("onnxruntime-tools not installed.  Run: pip install onnxruntime")
        raise


# ─── FP16 Export (Ultralytics) ────────────────────────────────────────────────

def export_fp16(weights: str, output_path: str, img_size: int = 640) -> str:
    """
    Export YOLOv8 to FP16 TorchScript / ONNX (half precision).

    Requires a CUDA-enabled machine for full benefit.

    Args:
        weights:     Path to .pt checkpoint
        output_path: Destination path
        img_size:    Model input size

    Returns:
        Path to FP16 model.
    """
    logger.info(f"Exporting {weights} → FP16 …")
    try:
        from ultralytics import YOLO
        model = YOLO(weights)
        exported = model.export(
            format = "onnx",
            imgsz  = img_size,
            half   = True,
            opset  = 17,
        )
        default_fp16 = str(Path(weights).with_suffix(".onnx"))
        if output_path != default_fp16:
            shutil.copy(default_fp16, output_path)
        logger.info(f"FP16 model → {output_path}  ({file_size_mb(output_path):.1f} MB)")
        return output_path
    except Exception as e:
        logger.error(f"FP16 export failed: {e}")
        raise


# ─── L1 Unstructured Pruning ─────────────────────────────────────────────────

def prune_model(
    weights:     str,
    output_path: str,
    sparsity:    float = 0.30,
) -> str:
    """
    Apply L1 unstructured pruning to Conv2d weight tensors.

    After pruning the zero-masks are made permanent (remove_reparametrization),
    and the pruned model is saved.

    Args:
        weights:     Path to .pt YOLOv8 checkpoint
        output_path: Where to save the pruned model
        sparsity:    Fraction of weights to zero out  (0.0 – 1.0)

    Returns:
        Path to pruned model.
    """
    logger.info(f"Pruning {weights} with sparsity={sparsity:.0%} …")
    try:
        from ultralytics import YOLO
        import torch.nn as nn

        model = YOLO(weights)
        pytorch_model = model.model.model     # underlying nn.Module

        pruned_count = 0
        total_count  = 0

        for name, module in pytorch_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                torch_prune.l1_unstructured(module, name="weight", amount=sparsity)
                torch_prune.remove(module, "weight")   # make permanent
                w = module.weight.data
                pruned_count += int((w == 0).sum().item())
                total_count  += w.numel()

        actual_sparsity = pruned_count / total_count if total_count else 0
        logger.info(f"Actual weight sparsity: {actual_sparsity:.2%}  "
                    f"({pruned_count:,}/{total_count:,} zeroed)")

        # Save pruned checkpoint
        model.save(output_path)
        logger.info(f"Pruned model → {output_path}  ({file_size_mb(output_path):.1f} MB)")
        return output_path

    except Exception as e:
        logger.error(f"Pruning failed: {e}")
        raise


# ─── Latency benchmark (quick) ───────────────────────────────────────────────

def quick_latency(model_path: str, device: str = "cpu", runs: int = 50) -> float:
    """
    Measure average inference latency for a .pt or .onnx model.

    Returns:
        Mean latency in milliseconds.
    """
    dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
    times = []

    suffix = Path(model_path).suffix.lower()

    if suffix in (".pt", ".pth", ""):
        from ultralytics import YOLO
        model = YOLO(model_path)
        # Warm-up
        for _ in range(5):
            model.predict(dummy, verbose=False, device=device)
        for _ in range(runs):
            t0 = time.perf_counter()
            model.predict(dummy, verbose=False, device=device)
            times.append(time.perf_counter() - t0)

    elif suffix == ".onnx":
        import onnxruntime as ort
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        sess = ort.InferenceSession(model_path, providers=providers)
        inp  = sess.get_inputs()[0].name
        for _ in range(5):
            sess.run(None, {inp: dummy})
        for _ in range(runs):
            t0 = time.perf_counter()
            sess.run(None, {inp: dummy})
            times.append(time.perf_counter() - t0)

    mean_ms = float(np.mean(times)) * 1000
    return mean_ms


# ─── Size & Latency comparison table ─────────────────────────────────────────

def comparison_report(models: Dict[str, str], device: str = "cpu") -> None:
    """
    Print a comparison table of model sizes and latencies.

    Args:
        models: {label: path} mapping
        device: "cpu" or "cuda"
    """
    print("\n" + "=" * 72)
    print(f"  MODEL COMPARISON  |  device={device}")
    print("=" * 72)
    print(f"  {'Label':<22s}  {'Size (MB)':>10s}  {'Latency (ms)':>14s}  {'FPS':>8s}")
    print("-" * 72)

    for label, path in models.items():
        if not Path(path).exists():
            print(f"  {label:<22s}  {'NOT FOUND':>10s}")
            continue
        size = file_size_mb(path)
        try:
            lat  = quick_latency(path, device=device, runs=30)
            fps  = 1000 / lat if lat > 0 else 0
            print(f"  {label:<22s}  {size:>10.1f}  {lat:>14.1f}  {fps:>8.1f}")
        except Exception as e:
            print(f"  {label:<22s}  {size:>10.1f}  {'ERROR':>14s}")
            logger.debug(f"Latency error for {label}: {e}")

    print("=" * 72 + "\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model optimization & export")
    p.add_argument("--weights",   required=True,
                   help="Path to YOLOv8 .pt checkpoint")
    p.add_argument("--config",    default="configs/config.yaml")
    p.add_argument("--mode",      default="all",
                   choices=["all", "onnx", "int8", "fp16", "prune", "compare"],
                   help="Optimization mode (default: all)")
    p.add_argument("--output-dir", default="optimization/outputs",
                   help="Where to save optimized models")
    p.add_argument("--sparsity",  type=float, default=None,
                   help="Pruning sparsity override (0.0–1.0)")
    p.add_argument("--device",    default="cpu",
                   help="Device for latency benchmarks")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    opt_cfg    = cfg["optimization"]
    img_size   = cfg["dataset"]["image_size"]
    sparsity   = args.sparsity or opt_cfg["pruning"]["sparsity"]
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name  = Path(args.weights).stem

    paths: Dict[str, str] = {"original (.pt)": args.weights}

    # ── ONNX export ───────────────────────────────────────────────────────────
    if args.mode in ("all", "onnx", "int8"):
        onnx_path = str(out_dir / f"{base_name}_fp32.onnx")
        export_to_onnx(args.weights, onnx_path, img_size=img_size,
                       opset=opt_cfg["export"]["opset_version"])
        paths["onnx fp32"] = onnx_path

    # ── INT8 quantization ─────────────────────────────────────────────────────
    if args.mode in ("all", "int8"):
        int8_path = str(out_dir / f"{base_name}_int8.onnx")
        quantize_int8_onnx(onnx_path, int8_path)
        paths["onnx int8"] = int8_path

    # ── FP16 export ───────────────────────────────────────────────────────────
    if args.mode in ("all", "fp16"):
        fp16_path = str(out_dir / f"{base_name}_fp16.onnx")
        export_fp16(args.weights, fp16_path, img_size=img_size)
        paths["onnx fp16"] = fp16_path

    # ── Pruning ───────────────────────────────────────────────────────────────
    if args.mode in ("all", "prune"):
        pruned_path = str(out_dir / f"{base_name}_pruned.pt")
        prune_model(args.weights, pruned_path, sparsity=sparsity)
        paths[f"pruned ({sparsity:.0%})"] = pruned_path

    # ── Comparison ────────────────────────────────────────────────────────────
    if args.mode in ("all", "compare"):
        comparison_report(paths, device=args.device)


if __name__ == "__main__":
    main()
