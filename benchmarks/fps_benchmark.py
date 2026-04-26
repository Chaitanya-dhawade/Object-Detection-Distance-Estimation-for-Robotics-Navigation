"""
benchmarks/fps_benchmark.py
─────────────────────────────
Comprehensive FPS & latency benchmarking across:
  - CPU vs GPU
  - Batch sizes (1, 4, 8)
  - Model variants (original, int8, fp16, pruned)

Output:
  - Console table
  - benchmarks/results/benchmark_results.csv
  - benchmarks/results/benchmark_plot.png  (matplotlib)

Usage:
    python benchmarks/fps_benchmark.py --weights best.pt
    python benchmarks/fps_benchmark.py --multi  # compare all optimized variants
"""

import argparse
import csv
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("benchmark")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Benchmark runner ─────────────────────────────────────────────────────────

class ModelBenchmark:
    """
    Measures throughput and latency for YOLO .pt and ONNX models.

    Args:
        model_path:   Path to .pt or .onnx file
        device:       "cpu" or "cuda"
        img_size:     Square input size (px)
        batch_size:   Number of images per forward pass
        warmup_runs:  Ignored warm-up iterations
        bench_runs:   Timed iterations
    """

    def __init__(
        self,
        model_path:  str,
        device:      str = "cpu",
        img_size:    int = 640,
        batch_size:  int = 1,
        warmup_runs: int = 20,
        bench_runs:  int = 200,
    ):
        self.model_path  = model_path
        self.device      = self._resolve_device(device)
        self.img_size    = img_size
        self.batch_size  = batch_size
        self.warmup_runs = warmup_runs
        self.bench_runs  = bench_runs

        self._suffix = Path(model_path).suffix.lower()
        self._load_model()

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._suffix in (".pt", ".pth", ""):
            self._load_pytorch()
        elif self._suffix == ".onnx":
            self._load_onnx()
        else:
            raise ValueError(f"Unsupported model format: {self._suffix}")

    def _load_pytorch(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(self.model_path)
        self._mode  = "pytorch"
        logger.info(f"Loaded PyTorch model: {self.model_path}")

    def _load_onnx(self) -> None:
        import onnxruntime as ort
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(self.model_path, providers=providers)
        self._inp_name = self._session.get_inputs()[0].name
        self._mode = "onnx"
        logger.info(f"Loaded ONNX model: {self.model_path}  "
                    f"providers={self._session.get_providers()}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def _forward(self, dummy: np.ndarray) -> None:
        """Single forward pass (no post-processing needed for timing)."""
        if self._mode == "pytorch":
            self._model.predict(
                dummy, verbose=False, device=self.device
            )
        else:
            self._session.run(None, {self._inp_name: dummy})

    def _make_dummy(self) -> np.ndarray:
        """Create a random float32 input batch."""
        return np.random.rand(
            self.batch_size, 3, self.img_size, self.img_size
        ).astype(np.float32)

    # ── Main benchmark ────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """
        Execute warm-up + timed benchmark.

        Returns:
            Dict with keys: mean_ms, std_ms, p50_ms, p95_ms, p99_ms, fps
        """
        dummy = self._make_dummy()

        # CUDA synchronisation wrapper
        def timed_forward():
            t0 = time.perf_counter()
            self._forward(dummy)
            if self.device == "cuda":
                torch.cuda.synchronize()
            return time.perf_counter() - t0

        # Warm-up
        logger.info(f"  Warm-up ({self.warmup_runs} runs) …")
        for _ in range(self.warmup_runs):
            timed_forward()

        # Timed runs
        logger.info(f"  Benchmarking ({self.bench_runs} runs) …")
        latencies = [timed_forward() for _ in range(self.bench_runs)]

        arr      = np.array(latencies) * 1000     # → milliseconds
        mean_ms  = float(np.mean(arr))
        std_ms   = float(np.std(arr))
        p50_ms   = float(np.percentile(arr, 50))
        p95_ms   = float(np.percentile(arr, 95))
        p99_ms   = float(np.percentile(arr, 99))
        fps      = (1000.0 / mean_ms) * self.batch_size

        return {
            "mean_ms": round(mean_ms, 2),
            "std_ms":  round(std_ms,  2),
            "p50_ms":  round(p50_ms,  2),
            "p95_ms":  round(p95_ms,  2),
            "p99_ms":  round(p99_ms,  2),
            "fps":     round(fps,     1),
        }

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA unavailable → CPU")
            return "cpu"
        return device


# ─── Multi-model suite ────────────────────────────────────────────────────────

def benchmark_suite(
    models:      Dict[str, str],
    devices:     List[str] = None,
    batch_sizes: List[int] = None,
    img_size:    int       = 640,
    warmup:      int       = 20,
    runs:        int       = 200,
) -> List[Dict]:
    """
    Run benchmarks for all model × device × batch_size combinations.

    Returns:
        List of result dicts, one per combination.
    """
    devices     = devices     or ["cpu"]
    batch_sizes = batch_sizes or [1]
    all_results = []

    for model_label, model_path in models.items():
        if not Path(model_path).exists():
            logger.warning(f"Skipping missing model: {model_path}")
            continue

        size_mb = Path(model_path).stat().st_size / (1024 ** 2)

        for device in devices:
            for bs in batch_sizes:
                logger.info(f"\n{'─'*50}")
                logger.info(f"  Model:  {model_label}")
                logger.info(f"  Device: {device}  |  batch={bs}")
                logger.info(f"{'─'*50}")

                try:
                    bench = ModelBenchmark(
                        model_path  = model_path,
                        device      = device,
                        img_size    = img_size,
                        batch_size  = bs,
                        warmup_runs = warmup,
                        bench_runs  = runs,
                    )
                    metrics = bench.run()
                    row = {
                        "model":   model_label,
                        "path":    model_path,
                        "size_mb": round(size_mb, 1),
                        "device":  device,
                        "batch":   bs,
                        **metrics,
                    }
                    all_results.append(row)
                    logger.info(
                        f"  → mean={metrics['mean_ms']:.1f}ms  "
                        f"fps={metrics['fps']:.1f}  "
                        f"p95={metrics['p95_ms']:.1f}ms"
                    )

                except Exception as e:
                    logger.error(f"  Benchmark failed: {e}")

    return all_results


# ─── Reporters ───────────────────────────────────────────────────────────────

def print_table(results: List[Dict]) -> None:
    """Print a pretty ASCII table of benchmark results."""
    if not results:
        print("No results to display.")
        return

    cols = ["model", "device", "batch", "size_mb", "fps", "mean_ms", "p95_ms", "p99_ms"]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in results)) for c in cols}

    sep = "─" * (sum(widths.values()) + 3 * len(cols) + 1)
    header = " │ ".join(c.upper().ljust(widths[c]) for c in cols)

    print("\n" + sep)
    print(f" {header}")
    print(sep)
    for r in results:
        row = " │ ".join(str(r.get(c, "─")).ljust(widths[c]) for c in cols)
        print(f" {row}")
    print(sep + "\n")


def save_csv(results: List[Dict], output: str = "benchmarks/results/benchmark_results.csv") -> None:
    """Write results to CSV."""
    if not results:
        return
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Results saved → {output}")


def save_plot(results: List[Dict], output: str = "benchmarks/results/benchmark_plot.png") -> None:
    """Generate FPS comparison bar chart."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Model Optimization Benchmark", fontsize=14, fontweight="bold")

        devices = sorted(set(r["device"] for r in results))
        colors  = {"cpu": "#4C72B0", "cuda": "#DD8452"}

        for ax, device in zip(axes, devices):
            subset = [r for r in results if r["device"] == device and r["batch"] == 1]
            if not subset:
                ax.set_visible(False)
                continue
            labels = [r["model"] for r in subset]
            fps    = [r["fps"]   for r in subset]

            bars = ax.bar(labels, fps, color=colors.get(device, "#888"), edgecolor="white")
            ax.set_title(f"Device: {device.upper()}  (batch=1)")
            ax.set_ylabel("FPS")
            ax.set_xlabel("Model variant")
            ax.tick_params(axis="x", rotation=30)

            for bar, val in zip(bars, fps):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.0f}",
                    ha="center", va="bottom", fontsize=9,
                )

        plt.tight_layout()
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150)
        logger.info(f"Plot saved → {output}")
        plt.close()

    except ImportError:
        logger.warning("matplotlib not installed – skipping plot generation")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FPS benchmarking for model variants")
    p.add_argument("--weights",    default="runs/train/robotics_nav_detection/weights/best.pt",
                   help="Path to primary .pt model")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--multi",      action="store_true",
                   help="Benchmark all optimized variants in optimization/outputs/")
    p.add_argument("--devices",    nargs="+", default=["cpu"],
                   help="Devices to benchmark on (cpu cuda)")
    p.add_argument("--batches",    nargs="+", type=int, default=[1, 4, 8])
    p.add_argument("--runs",       type=int, default=200)
    p.add_argument("--warmup",     type=int, default=20)
    p.add_argument("--output-dir", default="benchmarks/results")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    img_size = cfg["dataset"]["image_size"]

    # ── Build model registry ─────────────────────────────────────────────────
    models: Dict[str, str] = {"original": args.weights}

    if args.multi:
        opt_dir = Path("optimization/outputs")
        for p in sorted(opt_dir.glob("*.pt")):
            models[p.stem] = str(p)
        for p in sorted(opt_dir.glob("*.onnx")):
            models[p.stem] = str(p)

    # ── Run suite ────────────────────────────────────────────────────────────
    results = benchmark_suite(
        models      = models,
        devices     = args.devices,
        batch_sizes = args.batches,
        img_size    = img_size,
        warmup      = args.warmup,
        runs        = args.runs,
    )

    # ── Report ────────────────────────────────────────────────────────────────
    print_table(results)
    save_csv(results,  output=f"{args.output_dir}/benchmark_results.csv")
    save_plot(results, output=f"{args.output_dir}/benchmark_plot.png")


if __name__ == "__main__":
    main()
