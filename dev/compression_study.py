"""Current-recipe compression study: 50 harness trials on one GPU, plus profiling.

Run with modal run modal_runner.py::compression. Baseline seeds 0-4 run before
the variants and 5-9 afterwards. Every variant uses the same seeds 0-9. The
shared compiler cache fixes model autotuning choices where cache keys match;
each harness invocation still builds and resets its own fresh process.
"""

import argparse
import ast
import hashlib
import inspect
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

VARIANTS = {
    "baseline": {"epochs": 7},
    "epochs_6_75": {"epochs": 6.75},
    "epochs_6_5": {"epochs": 6.5},
    "batched_muon": {"epochs": 7},
    "fused_prepare": {"epochs": 7},
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def freeze(root):
    source = Path("submissions/airbench_muon/submission.py").read_bytes()
    helper = Path("dev/compression_variants.py").read_bytes()
    for name in VARIANTS:
        directory = root / "recipes" / name
        directory.mkdir(parents=True, exist_ok=True)
        text = source.decode("utf-8")
        if name in ("batched_muon", "fused_prepare"):
            text += ("\nfrom .compression_variants import install as _install_compression\n"
                     f"_install_compression(globals(), {name!r})\n")
            (directory / "compression_variants.py").write_bytes(helper)
        (directory / "submission.py").write_text(text, encoding="utf-8")
    return hashlib.sha256(source).hexdigest()


class CudaRegions:
    """Async stream intervals, including host launch gaps; not busy-kernel totals."""

    def __init__(self, steps, epochs):
        import torch

        self.used = defaultdict(int)
        sizes = dict.fromkeys(("reset", "optimizers", "transfer_cast", "normalize",
                               "labels", "whitening", "preflip", "padding"), 1)
        sizes.update(augmentation=epochs, forward_backward=steps, sgd=steps, muon=steps)
        self.pool = {}
        for name, count in sizes.items():
            pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                     for _ in range(count)]
            for start, end in pairs:
                start.record()
                end.record()
            self.pool[name] = pairs
        torch.cuda.synchronize()

    def start(self, name):
        self.pool[name][self.used[name]][0].record()

    def end(self, name):
        self.pool[name][self.used[name]][1].record()
        self.used[name] += 1

    def result(self):
        return {name: sum(a.elapsed_time(b) for a, b in pairs[:self.used[name]]) / 1000
                for name, pairs in self.pool.items()}


def instrument(function, regions):
    """Insert event calls only; verify stripping them restores the exact original AST."""
    original = ast.parse(inspect.getsource(function))

    def event(method, name):
        return ast.Expr(ast.Call(ast.Attribute(ast.Name("_regions", ast.Load()), method,
                                              ast.Load()), [ast.Constant(name)], []))

    class Insert(ast.NodeTransformer):
        def generic_visit(self, node):
            node = super().generic_visit(node)
            if not isinstance(node, ast.FunctionDef | ast.For | ast.If):
                return node
            body, index = [], 0
            while index < len(node.body):
                item = node.body[index]
                text = ast.unparse(item)
                region, width = None, 1
                for prefix, candidate in (
                    ("model.reset()", "reset"),
                    ("state.sgd, state.muon =", "optimizers"),
                    ("images = data.images.to", "transfer_cast"),
                    ("images = model.normalize", "normalize"),
                    ("state.labels = data.labels.to", "labels"),
                    ("model.init_whiten", "whitening"),
                    ("images = batch_flip_lr", "preflip"),
                    ("state.padded_images = F.pad", "padding"),
                    ("images = epoch_images", "augmentation"),
                    ("sgd.step()", "sgd"),
                    ("muon.step()", "muon"),
                ):
                    if text.startswith(prefix):
                        region = candidate
                        break
                if text.startswith("outputs = model("):
                    region, width = "forward_backward", 2
                    assert "cross_entropy" in ast.unparse(node.body[index + 1])
                if region:
                    body += [event("start", region), *node.body[index:index + width],
                             event("end", region)]
                else:
                    body.append(item)
                index += width
            node.body = body
            return node

    class Strip(ast.NodeTransformer):
        def visit_Expr(self, node):
            return None if ast.unparse(node).startswith("_regions.") else node

    tree = ast.fix_missing_locations(Insert().visit(ast.parse(inspect.getsource(function))))
    stripped = Strip().visit(ast.parse(ast.unparse(tree)))
    assert ast.dump(stripped) == ast.dump(original)
    namespace = {**function.__globals__, "_regions": regions}
    exec(compile(tree, "<async event instrumentation>", "exec"), namespace)
    return namespace[function.__name__]


def profile_baseline(root):
    import torch

    from benchmark.api import BuildContext
    from benchmark.data import load_split
    from benchmark.evaluate import accuracy, predict
    from benchmark.worker import load_submission, seed_everything

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    recipe = load_submission(root / "recipes" / "baseline")
    data = load_split(Path("data"), train=True)
    test = load_split(Path("data"), train=False)
    state = recipe.build(BuildContext(torch.device("cuda"), VARIANTS["baseline"]))
    steps = math.ceil(state.cfg["epochs"] * (50_000 // state.cfg["batch_size"]))
    rows = []
    for seed in range(3):
        regions = CudaRegions(steps, math.ceil(state.cfg["epochs"]))
        prepare, train = instrument(recipe.prepare, regions), instrument(recipe.train, regions)
        seed_everything(seed)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        prepare(state, data, seed)
        torch.cuda.synchronize()
        prepared = time.perf_counter()
        model = train(state)
        torch.cuda.synchronize()
        trained = time.perf_counter()
        intervals = regions.result()
        predictions, evaluation_time = predict(model, test.images, state.device, 1024)
        rows.append(dict(seed=seed, prepare_time=prepared - begin, train_time=trained - prepared,
                         total_timed_time=trained - begin, cuda_intervals=intervals,
                         accuracy=accuracy(predictions, test.labels), steps=state.steps,
                         evaluation_time=evaluation_time, region_counts=dict(regions.used)))
        print(f"Profile {seed + 1}/3: {trained - begin:.4f}s", flush=True)
        write_json(root / "profile.json", rows)


def run_logged(command, log, environment, timeout=1800):
    """Retain complete diagnostics without flooding the client with compile logs."""
    print("Running: " + " ".join(command[:5]), flush=True)
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run(command, env=environment, stdout=output,
                                stderr=subprocess.STDOUT, timeout=timeout, check=False)
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-24:]
    print("\n".join(tail), flush=True)
    return result.returncode


def aggregate(root):
    groups = defaultdict(list)
    builds = defaultdict(list)
    failures = []
    for manifest in sorted((root / "runs").glob("*/*/*/config.json")):
        config = json.loads(manifest.read_text())
        name = config["parameters"]["experiment_name"]
        builds[name].append(config.get("build_time"))
        summary = json.loads(manifest.with_name("summary.json").read_text())
        if not summary["complete"]:
            failures.append(dict(name=name, summary=summary))
        for line in manifest.with_name("trials.jsonl").read_text().splitlines():
            row = json.loads(line)
            row["result_dir"] = str(manifest.parent.relative_to(root))
            groups[name].append(row)
    summary = {}
    for name, rows in groups.items():
        rows.sort(key=lambda row: row["seed"])
        successful = [row for row in rows if row["status"] == "ok"]
        complete = len(successful) == 10 and [row["seed"] for row in successful] == list(range(10))
        entry = dict(complete=complete, trials=len(rows), successful_trials=len(successful),
                     builds=builds[name], official=False)
        for key in ("accuracy", "prepare_time", "train_time", "total_timed_time"):
            values = [row[key] for row in successful]
            entry[f"mean_{key}"] = statistics.mean(values) if values else None
            entry[f"sd_{key}"] = statistics.stdev(values) if len(values) > 1 else 0
        entry["clears_development_target"] = complete and entry["mean_accuracy"] >= 0.75
        summary[name] = entry
    baseline = {row["seed"]: row for row in groups["baseline"] if row["status"] == "ok"}
    for name, entry in summary.items():
        if not entry["complete"] or len(baseline) != 10:
            continue
        rows = groups[name]
        for key in ("accuracy", "total_timed_time"):
            deltas = [row[key] - baseline[row["seed"]][key] for row in rows]
            mean = statistics.mean(deltas)
            half_width = 2.262 * statistics.stdev(deltas) / math.sqrt(10)
            entry[f"paired_delta_{key}"] = mean
            entry[f"paired_95ci_{key}"] = [mean - half_width, mean + half_width]
    result = dict(variants=summary, failures=failures,
                  note="Paired intervals exclude systematic hardware/time drift; development only.")
    write_json(root / "comparison.json", result)
    write_json(root / "trials.json", dict(groups))
    return result


def study():
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = Path("results/compression/airbench_muon") / run_id
    root.mkdir(parents=True)
    source_hash = freeze(root)
    environment = {**os.environ,
                   "TORCHINDUCTOR_CACHE_DIR": str(Path("/tmp") / f"compression-{run_id}"),
                   "TORCH_LOGS": "recompiles", "PYTHONDONTWRITEBYTECODE": "1"}
    for name in ("compression_study.py", "compression_variants.py", "compression_checks.py"):
        (root / name).write_bytes((Path("dev") / name).read_bytes())
    write_json(root / "study.json", dict(source_sha256=source_hash, variants=VARIANTS,
                                        seeds=list(range(10)), benchmark_trials=50,
                                        profile_trials=3, same_gpu=True,
                                        shared_compiler_cache=environment["TORCHINDUCTOR_CACHE_DIR"]))
    try:
        status = run_logged([sys.executable, "-m", "dev.compression_checks", "--root", str(root)],
                            root / "validation.log", environment)
        if status:
            raise RuntimeError("Synthetic validation failed; benchmark not started")
        status = run_logged([sys.executable, "-m", "dev.compression_study", "--profile",
                             "--root", str(root)], root / "profile.log", environment)
        if status:
            raise RuntimeError("Baseline profile failed; benchmark not started")
        # Split the ten baseline seeds around the experiments without adding trials.
        blocks = [("baseline_before", "baseline", 0, 5),
                  ("epochs_6_75", "epochs_6_75", 0, 10),
                  ("batched_muon", "batched_muon", 0, 10),
                  ("epochs_6_5", "epochs_6_5", 0, 10),
                  ("fused_prepare", "fused_prepare", 0, 10),
                  ("baseline_after", "baseline", 5, 5)]
        for block, name, seed, count in blocks:
            print(f"Starting {block}: {count} trials", flush=True)
            command = [sys.executable, "-m", "benchmark.run", "--submission-path",
                       str(root / "recipes" / name), "--n", str(count), "--seed", str(seed),
                       "--params", json.dumps({**VARIANTS[name], "experiment_name": name}),
                       "--results-root", str(root / "runs" / block)]
            status = run_logged(command, root / f"{block}.log", environment)
            # Exit 1 can mean a complete run below 75%; retain it and continue.
            if status not in (0, 1):
                raise RuntimeError(f"{block} failed with exit {status}")
            records = list((root / "runs" / block).glob("*/*/summary.json"))
            if len(records) != 1:
                raise RuntimeError(f"{block} did not produce exactly one harness summary")
            comparison = aggregate(root)
            if comparison["failures"]:
                raise RuntimeError(f"{block} did not finish; see saved failure")
        comparison = aggregate(root)
        if set(comparison["variants"]) != set(VARIANTS) or not all(
            entry["complete"] for entry in comparison["variants"].values()
        ):
            raise RuntimeError("Comparison incomplete; see retained trials")
        print(json.dumps(comparison, indent=2), flush=True)
    finally:
        print(f"Results: {root.resolve()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.profile:
        profile_baseline(args.root)
    else:
        study()
