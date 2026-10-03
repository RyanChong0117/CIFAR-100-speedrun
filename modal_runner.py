"""Run the CIFAR-100 harness and dev tools on Modal A100s.

    modal run modal_runner.py --download                         # once: fetch CIFAR-100
    modal run modal_runner.py --submission baseline_resnet9 --n 2
    modal run modal_runner.py::curve --submission baseline_resnet9 --params '{"epochs": 20}'
    modal run modal_runner.py::sweep --file dev/sweeps/example.json   # parallel, one GPU each

Results are copied back into ./results/ (and kept in the "cifar100-results" volume),
then `python -m dev.collect_results` builds results/results.csv.

Note: Modal's A100-80GB is likely the SXM variant, not the official PCIe card, so
absolute times differ from official ones; compare recipes against each other.
"""

import json
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent

# The Dockerfile's `COPY . .` would otherwise rebuild the image on every code edit.
# Code we iterate on (submissions/, dev/) is excluded here and mounted at startup instead.
image = (
    modal.Image.from_dockerfile(
        ROOT / "Dockerfile",
        context_dir=ROOT,
        ignore=[
            ".git", ".venv", ".local", ".env*", "seeds.json", "**/__pycache__",
            ".pytest_cache", ".ruff_cache", "data", "results", "tests",
            "submissions", "dev", "modal_runner.py",
        ],
    )
    .add_local_dir(ROOT / "submissions", "/app/submissions", ignore=["**/__pycache__"])
    .add_local_dir(ROOT / "dev", "/app/dev", ignore=["**/__pycache__"])
)

app = modal.App("cifar100-speedrun")

# Persistent storage for CIFAR-100
data_volume = modal.Volume.from_name("cifar100-data", create_if_missing=True)

# Persistent storage for benchmark results (summary.json, trials.jsonl, curves, ...)
results_volume = modal.Volume.from_name("cifar100-results", create_if_missing=True)

# Account limit: at most 10 GPUs at once. sweep() launches in waves of this size, and each
# GPU function is also capped as a backstop.
MAX_GPUS = 10

GPU_FUNCTION = dict(
    image=image,
    gpu="A100-80GB",
    max_containers=MAX_GPUS,
    cpu=4,
    volumes={"/app/data": data_volume, "/app/results": results_volume},
    timeout=3600,
)


@app.function(image=image, volumes={"/app/data": data_volume}, cpu=2, timeout=600)
def download_data():
    subprocess.run(
        ["uv", "run", "python", "-m", "benchmark.data", "--root", "data"],
        cwd="/app",
        check=True,
    )
    data_volume.commit()
    print("CIFAR-100 download complete.")


def run_and_collect(command: list[str]) -> dict:
    """Stream a command's output, then return the files of the result it reports."""
    process = subprocess.Popen(
        ["uv", "run", "python", "-m", *command],
        cwd="/app",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    result_path = None
    for line in process.stdout:
        print(line, end="")
        if line.startswith("Results: "):
            result_path = Path(line.removeprefix("Results: ").strip())
    returncode = process.wait()
    results_volume.commit()  # persist even if the run failed or did not qualify

    files = {}
    if result_path is not None:
        root = Path("/app/results").resolve()  # the harness reports resolved volume paths
        paths = [result_path] if result_path.is_file() else result_path.rglob("*")
        for path in paths:
            if path.is_file() and "__pycache__" not in path.parts:
                files[str(path.relative_to(root))] = path.read_text()
    return {"returncode": returncode, "files": files}


@app.function(**GPU_FUNCTION)
def benchmark(
    submission: str, n: int, params: dict, accuracy_target: bool = True, seed: int = 0
) -> dict:
    command = ["benchmark.run", "--submission", submission, "--n", str(n),
               "--params", json.dumps(params), "--seed", str(seed)]
    if not accuracy_target:
        command.append("--no-accuracy-target")
    return run_and_collect(command)


@app.function(**GPU_FUNCTION)
def training_curve(submission: str, seed: int, params: dict, eval_every: int = 1) -> dict:
    return run_and_collect(
        ["dev.curve", "--submission", submission, "--seed", str(seed),
         "--params", json.dumps(params), "--eval-every", str(eval_every)]
    )


@app.function(**GPU_FUNCTION)
def profile_run(submission: str, params: dict, variants: str = "") -> dict:
    command = ["dev.profile", "--submission", submission, "--params", json.dumps(params)]
    if variants:
        command += ["--variants", variants]
    return run_and_collect(command)


@app.function(**GPU_FUNCTION)
def finite_check_run(submission: str, variants: str) -> dict:
    return run_and_collect(
        ["dev.check_finite", "--submission", submission, "--variants", variants]
    )


@app.function(**GPU_FUNCTION)
def compression_run() -> dict:
    """Five isolated variants, 10 seeds each, sequentially on one allocated A100."""
    return run_and_collect(["dev.compression_study"])


def save_locally(output: dict) -> None:
    for relpath, text in output["files"].items():
        path = ROOT / "results" / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    summary = next((k for k in output["files"] if k.endswith("summary.json")), None)
    curve = next((k for k in output["files"] if k.startswith(("curves/", "profiles/"))), None)
    saved = summary or curve
    print(f"exit {output['returncode']}; saved results/{Path(saved).parent if summary else saved}"
          if saved else f"exit {output['returncode']}; no result files")


@app.local_entrypoint()
def main(
    submission: str = "baseline_resnet9",
    n: int = 1,
    no_accuracy_target: bool = False,
    params: str = "{}",
    download: bool = False,
    seed: int = 0,
):
    if download:
        download_data.remote()
    save_locally(benchmark.remote(submission, n, json.loads(params), not no_accuracy_target, seed))


@app.local_entrypoint()
def curve(
    submission: str = "baseline_resnet9", seed: int = 0, params: str = "{}", eval_every: int = 1
):
    save_locally(training_curve.remote(submission, seed, json.loads(params), eval_every))


@app.local_entrypoint()
def profile(submission: str = "airbench_muon", params: str = "{}", variants_file: str = ""):
    variants = Path(variants_file).read_text() if variants_file else ""
    save_locally(profile_run.remote(submission, json.loads(params), variants))


@app.local_entrypoint()
def check_finite(variants_file: str, submission: str = "airbench_muon"):
    save_locally(finite_check_run.remote(submission, Path(variants_file).read_text()))


@app.local_entrypoint()
def compression():
    """Profile current defaults and run the approved 50-trial compression comparison."""
    save_locally(compression_run.remote())


@app.local_entrypoint()
def sweep(file: str):
    """Run every experiment in a JSON list in parallel, one A100 per experiment.

    Each entry: {"mode": "harness" | "curve", "submission": ..., "params": {...},
                 "n": 1, "seed": 0, "eval_every": 1, "no_accuracy_target": false}
    """
    experiments = json.loads(Path(file).read_text())
    for start in range(0, len(experiments), MAX_GPUS):
        wave = experiments[start : start + MAX_GPUS]
        if len(experiments) > MAX_GPUS:
            print(f"Wave {start // MAX_GPUS + 1}: experiments {start + 1}-{start + len(wave)}")
        run_wave(wave)


def run_wave(experiments: list[dict]) -> None:
    calls = []
    for e in experiments:
        if e.get("mode", "harness") == "curve":
            call = training_curve.spawn(
                e["submission"], e.get("seed", 0), e.get("params", {}), e.get("eval_every", 1)
            )
        else:
            call = benchmark.spawn(
                e["submission"], e.get("n", 1), e.get("params", {}),
                not e.get("no_accuracy_target", False), e.get("seed", 0),
            )
        calls.append((e, call))
    print(f"Launched {len(calls)} experiments in parallel")
    for e, call in calls:
        name = e.get("params", {}).get("experiment_name", e["submission"])
        try:
            output = call.get()
        except Exception as exc:
            print(f"{name}: failed: {exc!r}")
            continue
        print(f"{name}: ", end="")
        save_locally(output)
