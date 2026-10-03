"""Development-only, train-data-only CUDA-event profile of a frozen RC3 recipe.

    python -m dev.rc3_profile --submission-path path/to/frozen/recipe \
        --output results/profile.json --params '{}' --seed 1000 --n 3

Events are preallocated and initialized outside timing. Synchronization occurs at
prepare/train boundaries, never between individual training regions. Intervals
include stream idle time while the host launches work, rather than only busy
kernel time. Instrumented times are diagnostic, not submission benchmark scores.
Newton-Schulz is nested within Muon; batch_crop is nested within augmentation.
No evaluation data or labels are loaded, and accuracy is intentionally absent.
"""

import argparse
import ast
import copy
import hashlib
import inspect
import json
import math
import operator
import statistics
import subprocess
import textwrap
import time
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

PREP_REGIONS = (
    "reset", "optimizers", "transfer_cast", "normalize", "labels",
    "whitening", "initial_flip", "padding",
)
TRAIN_REGIONS = (
    "augmentation", "batch_gather", "forward", "loss", "backward", "sgd", "muon",
)
NESTED_REGIONS = ("batch_crop", "newton_schulz")


class CudaRegions:
    """Record asynchronous stream intervals without synchronizing the loop."""

    def __init__(self, steps, epochs, muon_parameters):
        import torch

        self.used = defaultdict(int)
        self.active = {}
        sizes = dict.fromkeys(PREP_REGIONS, 1)
        sizes.update(dict.fromkeys(TRAIN_REGIONS, steps))
        sizes.update(prepare=1, train=1, augmentation=epochs, batch_crop=epochs,
                     batch_gather=2 * steps, newton_schulz=steps * muon_parameters)
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
        if name in self.active:
            raise RuntimeError(f"Profile region is recursively active: {name}")
        index = self.used[name]
        if index >= len(self.pool[name]):
            raise RuntimeError(f"Profile capacity exceeded: {name}")
        self.active[name] = index
        self.pool[name][index][0].record()

    def end(self, name):
        index = self.active.pop(name)
        self.pool[name][index][1].record()
        self.used[name] += 1

    def call(self, name, function, /, *args, **kwargs):
        self.start(name)
        try:
            return function(*args, **kwargs)
        finally:
            self.end(name)

    def result(self):
        if self.active:
            raise RuntimeError(f"Unclosed profile regions: {self.active}")
        return {name: sum(a.elapsed_time(b) for a, b in pairs[:self.used[name]]) / 1000
                for name, pairs in self.pool.items()}


def instrument_tree(source):
    """Wrap original expressions; stripping wrappers must recover the exact AST.

    Nested call argument evaluation remains in its original order. In particular,
    image indexing finishes before forward timing starts, and loss calculation
    finishes before backward timing starts. No loss/backward math is rewritten.
    """
    original = ast.parse(textwrap.dedent(source))

    def event(method, name):
        return ast.Expr(ast.Call(ast.Attribute(ast.Name("_rc3_regions", ast.Load()), method,
                                              ast.Load()), [ast.Constant(name)], []))

    def wrap(name, function, args, keywords=()):
        return ast.Call(ast.Attribute(ast.Name("_rc3_regions", ast.Load()), "call", ast.Load()),
                        [ast.Constant(name), function, *args], list(keywords))

    class Insert(ast.NodeTransformer):
        def visit_Subscript(self, node):
            node = self.generic_visit(node)
            if ast.unparse(node) in ("images[idxs]", "state.labels[idxs]"):
                return wrap("batch_gather", ast.Name("_rc3_getitem", ast.Load()),
                            [node.value, node.slice])
            return node

        def visit_Call(self, node):
            node = self.generic_visit(node)
            target = ast.unparse(node.func)
            region = {
                "model": "forward", "F.cross_entropy": "loss",
                "epoch_images": "augmentation", "batch_crop": "batch_crop",
                "sgd.step": "sgd", "muon.step": "muon",
            }.get(target)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "backward":
                region = "backward"
            return wrap(region, node.func, node.args, node.keywords) if region else node

        def generic_visit(self, node):
            node = super().generic_visit(node)
            if not isinstance(node, ast.FunctionDef | ast.For | ast.If):
                return node
            body = []
            for item in node.body:
                statement = ast.unparse(item)
                region = None
                for prefix, candidate in (
                    ("model.reset()", "reset"),
                    ("state.sgd, state.muon =", "optimizers"),
                    ("images = data.images.to", "transfer_cast"),
                    ("images = model.normalize", "normalize"),
                    ("state.labels = data.labels.to", "labels"),
                    ("model.init_whiten", "whitening"),
                    ("images = batch_flip_lr", "initial_flip"),
                    ("state.padded_images = F.pad", "padding"),
                ):
                    if statement.startswith(prefix):
                        region = candidate
                        break
                if region:
                    body.extend([event("start", region), item, event("end", region)])
                else:
                    body.append(item)
            node.body = body
            return node

    class Strip(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id == "_rc3_regions"
                    and node.value.func.attr in ("start", "end")):
                return None
            return self.generic_visit(node)

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if (isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "_rc3_regions" and node.func.attr == "call"):
                function, args = node.args[1], node.args[2:]
                if isinstance(function, ast.Name) and function.id == "_rc3_getitem":
                    return ast.Subscript(args[0], args[1], ast.Load())
                return ast.Call(function, args, node.keywords)
            return node

    tree = ast.fix_missing_locations(Insert().visit(copy.deepcopy(original)))
    stripped = ast.fix_missing_locations(Strip().visit(copy.deepcopy(tree)))
    if ast.dump(stripped) != ast.dump(original):
        raise AssertionError("Profiling changed the original learning statements")
    return tree


def instrument(function, regions, overrides=None):
    tree = instrument_tree(inspect.getsource(function))
    namespace = {**function.__globals__, "_rc3_regions": regions, "_rc3_getitem": operator.getitem,
                 **(overrides or {})}
    exec(compile(tree, "<RC3 development-only CUDA-event profile>", "exec"), namespace)
    return namespace[function.__name__]


def summarize(rows):
    valid = [row for row in rows if row["status"] == "ok"]
    summary = {"trials": len(rows), "successful_trials": len(valid)}
    for metric in ("prepare_time", "train_time", "total_timed_time"):
        values = [row[metric] for row in valid]
        summary[metric] = stats(values)
    summary["cuda_seconds"] = {
        key: stats([row["cuda_seconds"][key] for row in valid])
        for key in valid[0]["cuda_seconds"]
    } if valid else {}
    return summary


def stats(values):
    return {
        "mean": statistics.mean(values) if values else None,
        "sd": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def write_json(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args):
    import torch

    from benchmark.api import BuildContext
    from benchmark.data import load_split
    from benchmark.worker import load_submission, seed_everything

    recipe_path = args.submission_path.resolve()
    if recipe_path.is_file():
        if recipe_path.name != "submission.py":
            raise ValueError("A submission file must be named submission.py")
        recipe_path = recipe_path.parent
    source = (recipe_path / "submission.py").read_bytes()
    params = json.loads(args.params)
    if not isinstance(params, dict):
        raise ValueError("--params must be a JSON object")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "started_utc": datetime.now(UTC).isoformat(),
        "submission_path": str(recipe_path),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "profiler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameter_overrides": params,
        "requested_trials": args.n,
        "seeds": list(range(args.seed, args.seed + args.n)),
        "official": False,
        "train_data_only": True,
        "notes": [
            "Asynchronous CUDA intervals; synchronized prepare/train boundaries.",
            "Instrumented diagnostic intervals include host launch gaps and event overhead.",
            "newton_schulz is included in muon; batch_crop is included in augmentation.",
            "Optimizer construction is preparation; first-step state allocation is training.",
            "No evaluation split or evaluation labels are loaded; accuracy is not measured.",
        ],
        "rows": [], "status": "starting",
    }
    # Refuse to replace any existing profile. Subsequent writes checkpoint this run only.
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True)
        result["gpu_model"] = gpu.stdout.strip()
        result["nvidia_smi_command"] = "nvidia-smi --query-gpu=name --format=csv,noheader"
        result["torch_version"] = torch.__version__
        result["cuda_version"] = torch.version.cuda
        module = load_submission(recipe_path)
        # Only the training split is accessed, exactly as in prepare/train worker use.
        data = load_split(args.data_root, train=True)
        device = torch.device("cuda")
        torch.cuda.synchronize()
        build_start = time.perf_counter()
        state = module.build(BuildContext(device, params))
        torch.cuda.synchronize()
        result["build_time"] = time.perf_counter() - build_start
        result["parameters"] = dict(state.cfg)
        steps = math.ceil(state.cfg["epochs"] * (len(data.labels) // state.cfg["batch_size"]))
        epochs = math.ceil(state.cfg["epochs"])
        muon_parameters = sum(len(p.shape) == 4 and p.requires_grad
                              for p in state.model.parameters())
        result["expected_optimizer_steps"] = steps
        result["convolution_parameters"] = muon_parameters
        original_zeropower = state.zeropower
        for seed in result["seeds"]:
            regions = CudaRegions(steps, epochs, muon_parameters)
            epoch_images = instrument(module.epoch_images, regions)
            prepare = instrument(module.prepare, regions)
            train = instrument(module.train, regions, {"epoch_images": epoch_images})

            def measured_zeropower(*a, _regions=regions, **kw):
                return _regions.call("newton_schulz", original_zeropower, *a, **kw)

            state.zeropower = measured_zeropower
            seed_everything(seed)
            row = {"seed": seed, "status": "running", "gpu_model": result["gpu_model"]}
            result["rows"].append(row)
            try:
                torch.cuda.synchronize()
                begin = time.perf_counter()
                regions.call("prepare", prepare, state, data, seed)
                torch.cuda.synchronize()
                prepared = time.perf_counter()
                regions.call("train", train, state)
                torch.cuda.synchronize()
                trained = time.perf_counter()
                intervals = regions.result()
                intervals["prepare_misc"] = intervals["prepare"] - sum(
                    intervals[name] for name in PREP_REGIONS)
                intervals["train_misc"] = intervals["train"] - sum(
                    intervals[name] for name in TRAIN_REGIONS)
                intervals["muon_excluding_ns"] = intervals["muon"] - intervals["newton_schulz"]
                intervals["augmentation_excluding_crop"] = (
                    intervals["augmentation"] - intervals["batch_crop"])
                intervals["transfer_normalize_combined"] = (
                    intervals["transfer_cast"] + intervals["normalize"])
                row.update(status="ok", prepare_time=prepared - begin,
                           train_time=trained - prepared, total_timed_time=trained - begin,
                           cuda_seconds=intervals, region_counts=dict(regions.used),
                           optimizer_steps=state.steps)
                if state.steps != steps:
                    raise AssertionError(f"Expected {steps} steps, observed {state.steps}")
                expected_counts = {"forward": steps, "backward": steps, "loss": steps,
                                   "sgd": steps, "muon": steps, "batch_gather": 2 * steps,
                                   "newton_schulz": steps * muon_parameters,
                                   "augmentation": epochs, "prepare": 1, "train": 1}
                for name, expected in expected_counts.items():
                    if regions.used[name] != expected:
                        raise AssertionError(
                            f"{name}: expected {expected} calls, got {regions.used[name]}")
                print(f"Profile seed {seed}: prepare={prepared - begin:.6f}s "
                      f"train={trained - prepared:.6f}s steps={state.steps}", flush=True)
            except BaseException as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                           traceback=traceback.format_exc())
                raise
            finally:
                state.zeropower = original_zeropower
                result["summary"] = summarize(result["rows"])
                write_json(args.output, result)
        result["status"] = "complete"
        print(json.dumps(result["summary"], indent=2), flush=True)
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        raise
    finally:
        result["finished_utc"] = datetime.now(UTC).isoformat()
        result["summary"] = summarize(result["rows"])
        write_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--params", default="{}")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    run(args)


if __name__ == "__main__":
    main()
