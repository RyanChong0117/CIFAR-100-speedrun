"""Frozen-recipe CUDA attribution and paired, uninstrumented performance studies.

Run via modal_runner.py::performance. Every variant is a fresh subprocess on
the same allocated GPU. The production submission is never monkey-patched.
"""

import argparse
import ast
import hashlib
import inspect
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from benchmark.api import BuildContext, TrainingData
from benchmark.data import load_split
from benchmark.evaluate import accuracy, predict
from benchmark.hardware import gpu_telemetry
from benchmark.worker import seed_everything
from dev import airbench_reference as recipe
from dev.airbench_optimizations import (
    BatchedMuon,
    batch_crop_vectorized,
    crop_with_shifts,
    zeropower_batched,
)


def sync():
    torch.cuda.synchronize()


def timed(function, *args):
    sync()
    start = time.perf_counter()
    result = function(*args)
    sync()
    return result, time.perf_counter() - start


class CudaRegions:
    """Record events asynchronously; synchronize only at trial boundaries.

    Events report stream elapsed time, including host launch gaps while the
    stream is idle. They do not claim summed busy-kernel time from CUPTI.
    Newton-Schulz intervals are nested inside Muon, not an additive category.
    """

    def __init__(self):
        self.pool = {}
        self.used = defaultdict(int)
        for name, count in {
            "prepare": 1,
            "train": 1,
            "augmentation": 12,
            "forward_backward": 300,
            "sgd": 300,
            "muon": 300,
            "newton_schulz": 2700,
        }.items():
            pairs = [
                (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for _ in range(count)
            ]
            for start, end in pairs:
                start.record()
                end.record()
            self.pool[name] = pairs
        sync()

    def start(self, name):
        self.pool[name][self.used[name]][0].record()

    def end(self, name):
        self.pool[name][self.used[name]][1].record()
        self.used[name] += 1

    def result(self):
        sync()
        values = {
            name: sum(a.elapsed_time(b) for a, b in pairs[: self.used[name]]) / 1000
            for name, pairs in self.pool.items()
        }
        values["muon_excluding_newton_schulz"] = values["muon"] - values["newton_schulz"]
        values["other_train"] = values["train"] - sum(
            values[name] for name in ("augmentation", "forward_backward", "sgd", "muon")
        )
        return {"cuda_elapsed_seconds": values, "region_counts": dict(self.used)}


def instrument_train(regions):
    """Insert only event calls; retain the original training statements verbatim."""
    tree = ast.parse(inspect.getsource(recipe.train))

    def event(method, name):
        return ast.Expr(
            ast.Call(
                ast.Attribute(ast.Name("_regions", ast.Load()), method, ast.Load()),
                [ast.Constant(name)],
                [],
            )
        )

    class Instrument(ast.NodeTransformer):
        def generic_visit(self, node):
            node = super().generic_visit(node)
            if isinstance(node, ast.FunctionDef | ast.While | ast.For):
                body, i = [], 0
                while i < len(node.body):
                    item = node.body[i]
                    region, width = None, 1
                    if isinstance(item, ast.Assign) and isinstance(item.value, ast.Call):
                        function = item.value.func
                        if isinstance(function, ast.Name) and function.id == "epoch_images":
                            region = "augmentation"
                        elif isinstance(function, ast.Name) and function.id == "model":
                            region, width = "forward_backward", 2
                    if isinstance(item, ast.Expr) and isinstance(item.value, ast.Call):
                        function = item.value.func
                        if (
                            isinstance(function, ast.Attribute)
                            and function.attr == "step"
                            and isinstance(function.value, ast.Name)
                            and function.value.id in ("sgd", "muon")
                        ):
                            region = function.value.id
                    if region:
                        body += [
                            event("start", region),
                            *node.body[i : i + width],
                            event("end", region),
                        ]
                    else:
                        body.append(item)
                    i += width
                node.body = body
            return node

    tree = ast.fix_missing_locations(Instrument().visit(tree))
    namespace = {**recipe.__dict__, "_regions": regions}
    exec(compile(tree, "<event-instrumented original train>", "exec"), namespace)
    return namespace["train"]


def crop_validation():
    results = []
    # Cover the reference's two algorithms, zero translation, and every offset.
    for radius in (0, 1, 2, 3):
        size, n = 8, (2 * radius + 1) ** 2
        images = torch.arange(
            n * 3 * (size + 2 * radius) ** 2, device="cuda", dtype=torch.float32
        ).reshape(n, 3, size + 2 * radius, size + 2 * radius)
        images = images.contiguous(memory_format=torch.channels_last)
        shifts = torch.cartesian_prod(
            torch.arange(-radius, radius + 1, device="cuda"),
            torch.arange(-radius, radius + 1, device="cuda"),
        )
        actual = crop_with_shifts(images, size, shifts)
        expected = torch.stack(
            [
                images[
                    i,
                    :,
                    radius + int(dy) : radius + int(dy) + size,
                    radius + int(dx) : radius + int(dx) + size,
                ]
                for i, (dy, dx) in enumerate(shifts.cpu().tolist())
            ]
        )
        assert torch.equal(actual, expected)
        seed_everything(123)
        expected = recipe.batch_crop(images, size)
        rng_expected = torch.cuda.get_rng_state()
        seed_everything(123)
        actual = batch_crop_vectorized(images, size)
        rng_actual = torch.cuda.get_rng_state()
        assert torch.equal(actual, expected) and actual.stride() == expected.stride()
        assert torch.equal(rng_actual, rng_expected)
        results.append(
            {
                "radius": radius,
                "all_offsets_exact": True,
                "same_seed_output_exact": True,
                "rng_state_exact": True,
                "output_strides_equal": True,
            }
        )
    return results


def ns_validation(state, compiled_batched):
    buckets = defaultdict(list)
    for p in state.model.parameters():
        if p.requires_grad and p.ndim == 4:
            buckets[(len(p), p.numel() // len(p))].append(p)
    seed_everything(123)
    results = []
    for shape, parameters in buckets.items():
        count = len(parameters)
        if count < 2:
            continue
        matrices = torch.randn(count, *shape, device="cuda", dtype=torch.float16)
        serial = torch.stack([state.zeropower(g) for g in matrices])
        batched = compiled_batched(matrices)
        difference = batched.float() - serial.float()
        relative_l2 = float(difference.norm() / serial.float().norm())
        assert torch.isfinite(batched).all()
        # BF16 batched GEMM/reduction orders can round differently. Report that
        # explicitly; never claim bitwise Newton-Schulz identity.
        if relative_l2 > 0.05:
            raise AssertionError(f"Excessive batched NS numerical deviation: {relative_l2}")
        results.append(
            {
                "matrix_shape": list(shape),
                "count": count,
                "bitwise_equal": torch.equal(serial, batched),
                "relative_l2_error": relative_l2,
                "max_absolute_error": float(difference.abs().max()),
                "iterations": 3,
                "normalization": "independent per matrix",
            }
        )
    return results


def worker(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    train_data = load_split(Path("data"), train=True)
    test_data = load_split(Path("data"), train=False)
    variant = json.loads(args.variant)
    mode = variant["mode"]
    crop = variant.get("crop", False)
    compiled_crop = variant.get("compiled_crop", False)
    batched = variant.get("batched", False)
    profiling = variant.get("profile", False)
    params = {"compile_mode": mode}
    validation = {}
    # Validate before patching the frozen module.
    original_crop = recipe.batch_crop
    if crop or compiled_crop:
        validation["crop"] = crop_validation()
        recipe.batch_crop = batch_crop_vectorized
    if compiled_crop:
        # randint stays eager to preserve the exact reference RNG stream.
        crop_kernel = torch.compile(crop_with_shifts, fullgraph=True)

        def compiled_crop_wrapper(images, crop_size):
            radius = (images.size(-1) - crop_size) // 2
            shifts = torch.randint(-radius, radius + 1, size=(len(images), 2), device=images.device)
            return crop_kernel(images, crop_size, shifts).contiguous()

        recipe.batch_crop = compiled_crop_wrapper
    batched_function = torch.compile(zeropower_batched) if batched else None
    if batched:

        def make_muon(*a, **kw):
            return BatchedMuon(*a, **kw, batched_zeropower=batched_function)

        recipe.Muon = make_muon
    context = BuildContext(torch.device("cuda"), params)
    state, build_time = timed(recipe.build, context)
    if compiled_crop:

        def warm_and_validate_crop():
            synthetic = TrainingData(
                torch.randint(
                    0,
                    256,
                    (50_000, 3, 32, 32),
                    dtype=torch.uint8,
                    generator=torch.Generator().manual_seed(123),
                ),
                torch.zeros(50_000, dtype=torch.int64),
            )
            # Run the actual preparation path on synthetic data. In particular,
            # reflection padding's output strides can differ from a manually
            # constructed channels-last tensor. Real-data work stays timed.
            recipe.prepare(state, synthetic, 123)
            images = state.padded_images
            seed_everything(123)
            expected = original_crop(images, 32)
            expected_rng = torch.cuda.get_rng_state()
            seed_everything(123)
            actual = recipe.batch_crop(images, 32)
            actual_rng = torch.cuda.get_rng_state()
            assert torch.equal(actual, expected)
            assert actual.stride() == expected.stride()
            assert torch.equal(actual_rng, expected_rng)
            return {
                "count": 50_000,
                "dtype": str(state.dtype),
                "pixels_exact": True,
                "strides_exact": True,
                "rng_exact": True,
                "production_shape_warmed_in_build": True,
                "input_strides": list(images.stride()),
                "synthetic_prepare_path": True,
            }

        production_validation, crop_build_time = timed(warm_and_validate_crop)
        build_time += crop_build_time
        validation["compiled_crop"] = production_validation
    if batched:
        validation["newton_schulz"] = ns_validation(state, batched_function)
    # First real-data trials keep every operation/data transfer inside timing.
    metadata = {
        "variant": variant,
        "resolved_config": state.cfg,
        "seeds": list(range(args.n)),
        "gpu": torch.cuda.get_device_name(),
        "telemetry": gpu_telemetry(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "build_time_seconds": build_time,
        "validation": validation,
        "reference_sha256": hashlib.sha256(Path(recipe.__file__).read_bytes()).hexdigest(),
        "timing_method": "synchronized perf_counter, same boundaries as benchmark.worker",
        "profile_method": "async CUDA events; elapsed stream intervals include launch gaps",
        "ns_compile_mode": "default (unchanged from reference)",
        "compile_cache": str(Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).resolve()),
        "official": False,
    }
    (out / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for source in ("airbench_reference.py", "airbench_optimizations.py", "airbench_perf.py"):
        (out / source).write_bytes((Path("dev") / source).read_bytes())
    rows = []
    for seed in range(args.n):
        regions = CudaRegions() if profiling else None
        train_function = instrument_train(regions) if profiling else recipe.train
        seed_everything(seed)
        sync()
        begin = time.perf_counter()
        if profiling:
            regions.start("prepare")
        recipe.prepare(state, train_data, seed)
        if profiling:
            regions.end("prepare")
        sync()
        prepared = time.perf_counter()
        if profiling:
            original_zeropower = state.muon.zeropower

            def measured_ns(matrix):
                regions.start("newton_schulz")
                value = original_zeropower(matrix)
                regions.end("newton_schulz")
                return value

            state.muon.zeropower = measured_ns
            regions.start("train")
        model = train_function(state)
        if profiling:
            regions.end("train")
        sync()
        trained = time.perf_counter()
        profile = regions.result() if profiling else None
        predictions, evaluation_time = predict(model, test_data.images, state.device, 1024)
        row = {
            "seed": seed,
            "accuracy": accuracy(predictions, test_data.labels),
            "prepare_time": prepared - begin,
            "train_time": trained - prepared,
            "total_timed_time": trained - begin,
            "evaluation_time": evaluation_time,
            "steps": state.steps,
            "status": "ok",
        }
        if profile:
            row["profile"] = profile
        rows.append(row)
        with (out / "trials.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"{variant['name']} trial {seed + 1}/{args.n}: "
            f"accuracy={row['accuracy']:.2%} prepare+train={row['total_timed_time']:.4f}s",
            flush=True,
        )
    summary = {
        "variant": variant,
        "number_of_trials": len(rows),
        "complete": len(rows) == args.n,
        "mean_accuracy": statistics.mean(r["accuracy"] for r in rows),
        "accuracy_std": statistics.stdev(r["accuracy"] for r in rows) if len(rows) > 1 else 0,
        "mean_training_time": statistics.mean(r["total_timed_time"] for r in rows),
        "training_time_std": statistics.stdev(r["total_timed_time"] for r in rows)
        if len(rows) > 1
        else 0,
        "mean_prepare_time": statistics.mean(r["prepare_time"] for r in rows),
        "mean_train_time": statistics.mean(r["train_time"] for r in rows),
        "mean_evaluation_time": statistics.mean(r["evaluation_time"] for r in rows),
        "build_time_seconds": build_time,
        "gpu": metadata["gpu"],
        "qualified": statistics.mean(r["accuracy"] for r in rows) >= 0.75,
        "official": False,
    }
    if profiling:
        summary["mean_cuda_elapsed_seconds"] = {
            key: statistics.mean(r["profile"]["cuda_elapsed_seconds"][key] for r in rows)
            for key in rows[0]["profile"]["cuda_elapsed_seconds"]
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def study(args):
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = Path("results/performance/airbench_muon") / (run_id + "-" + args.stage)
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "profile":
        variants = [{"name": "baseline_cuda_profile", "mode": "default", "profile": True}]
    elif args.stage in ("modes", "compare"):
        variants = [
            {"name": mode, "mode": mode} for mode in ("default", "reduce-overhead", "max-autotune")
        ]
    elif args.stage == "crop-compiled":
        variants = [
            {"name": "baseline_control", "mode": args.mode},
            {"name": "compiled_vectorized_crop_only", "mode": args.mode, "compiled_crop": True},
            {"name": "baseline_control_after", "mode": args.mode},
        ]
    else:
        variants = [
            {"name": "baseline_control", "mode": args.mode},
            {"name": "vectorized_crop_only", "mode": args.mode, "crop": True},
            {"name": "batched_muon_only", "mode": args.mode, "batched": True},
            {
                "name": "vectorized_crop_and_batched_muon",
                "mode": args.mode,
                "crop": True,
                "batched": True,
            },
            {"name": "baseline_control_after", "mode": args.mode},
        ]
    summaries = []
    for variant in variants:
        out = root / variant["name"]
        command = [
            sys.executable,
            "-m",
            "dev.airbench_perf",
            "--stage",
            "worker",
            "--n",
            str(args.n),
            "--variant",
            json.dumps(variant),
            "--out",
            str(out),
        ]
        # Reuse precisely the same model autotuning decisions for the isolated
        # optimizations. Only the three initial mode comparisons use cold caches.
        cache_name = variant["name"]
        if args.stage in ("optimizations", "compare", "crop-compiled") and cache_name not in (
            "default",
            "reduce-overhead",
            "max-autotune",
        ):
            cache_name = "baseline_control"
        env = {
            **os.environ,
            "TORCHINDUCTOR_CACHE_DIR": str(
                Path("/tmp") / ("airbench-" + root.name + "-" + cache_name)
            ),
        }
        print(f'Starting {variant["name"]} on {torch.cuda.get_device_name()}', flush=True)
        result = subprocess.run(command, env=env, check=False)
        if result.returncode:
            failure = {"variant": variant, "complete": False, "returncode": result.returncode}
            out.mkdir(parents=True, exist_ok=True)
            (out / "failure.json").write_text(json.dumps(failure, indent=2))
            summaries.append(failure)
        else:
            summaries.append(json.loads((out / "summary.json").read_text()))
        (root / "study.json").write_text(json.dumps(summaries, indent=2) + "\n")
        if args.stage == "compare" and len(summaries) == 3:
            eligible = [s for s in summaries if s.get("complete") and s.get("qualified")]
            if not eligible:
                eligible = [s for s in summaries if s.get("complete")]
            if not eligible:
                raise RuntimeError("No compile mode completed; cannot compare optimizations")
            best = min(eligible, key=lambda s: s["mean_training_time"])["variant"]["mode"]
            print(f"Using measured compile mode {best} for isolated optimizations", flush=True)
            variants.extend(
                [
                    {"name": "baseline_control", "mode": best},
                    {"name": "vectorized_crop_only", "mode": best, "crop": True},
                    {"name": "batched_muon_only", "mode": best, "batched": True},
                    {
                        "name": "vectorized_crop_and_batched_muon",
                        "mode": best,
                        "crop": True,
                        "batched": True,
                    },
                    {"name": "baseline_control_after", "mode": best},
                ]
            )
    print(f"Results: {root.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("profile", "modes", "compare", "crop-compiled", "optimizations", "worker"),
        required=True,
    )
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--mode", default="default")
    parser.add_argument("--variant")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.stage == "worker":
        try:
            worker(args)
        except Exception:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / "error.txt").write_text(traceback.format_exc())
            raise
    else:
        study(args)


if __name__ == "__main__":
    main()
