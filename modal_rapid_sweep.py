"""Parallel 30-epoch ResNet11 recipe experiments and gated paired width comparison."""

import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

import modal

from modal_runner import data_volume, image
from rapid_convergence_experiments import (
    rapid_parameters,
    resolve_rapid_parameters,
    run_rapid_stages,
)
from sweep_utils import print_ranking, require_sweep_gpu, run_sweep, save_summaries, write_json

app = modal.App("cifar100-rapid-convergence")
results_volume = modal.Volume.from_name("cifar100-experiments", create_if_missing=True)
check_image = image.add_local_file(
    Path(__file__).parent / "tests" / "rapid_recipe_check.py",
    "/app/rapid_recipe_check.py",
)


@app.function(image=check_image, cpu=4, memory=8192, retries=0, timeout=600)
def check_recipes():
    subprocess.run(["uv", "run", "python", "rapid_recipe_check.py"], cwd="/app", check=True)


def run_job(job):
    names = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        )
        .strip()
        .splitlines()
    )
    print(f"Job {job['configuration']} provider GPU: {names}", flush=True)
    require_sweep_gpu(names)
    parameters = job["parameters"]
    return run_sweep(
        parameters if isinstance(parameters, list) else [parameters],
        Path(job["directory"]),
        n=job["n"],
        seed=job["seed"],
        checkpoint=results_volume.commit,
    )


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    volumes={"/app/data": data_volume, "/sweep-results": results_volume},
    max_containers=12,
    retries=0,
    timeout=3600,
)
def benchmark_recipe(job: dict):
    return run_job(job)[0]


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    volumes={"/app/data": data_volume, "/sweep-results": results_volume},
    max_containers=1,
    retries=0,
    timeout=7200,
)
def compare_widths(job: dict):
    # Both three-seed benchmarks are sequential on this same GPU allocation.
    return run_job(job)


@app.function(
    image=image,
    cpu=1,
    volumes={"/sweep-results": results_volume},
    retries=0,
    timeout=28800,
)
def sweep(parameters: list[dict], screen_seed: int = 0, validation_seed: int = 100):
    check_recipes.remote()
    run_id = "rapid-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = Path("/sweep-results") / run_id
    print(f"Persistent results: cifar100-experiments/{run_id}", flush=True)

    def execute_stage(jobs):
        for result in benchmark_recipe.map(jobs, return_exceptions=True):
            results_volume.reload()
            yield result

    def execute_comparison(job):
        try:
            result = compare_widths.remote(job)
        except Exception as exc:
            result = exc
        results_volume.reload()
        return result

    rows = run_rapid_stages(
        parameters,
        directory,
        execute_stage=execute_stage,
        execute_comparison=execute_comparison,
        checkpoint=results_volume.commit,
        screen_seed=screen_seed,
        validation_seed=validation_seed,
    )
    return {"run_id": run_id, "volume": "cifar100-experiments", "rows": rows}


@app.local_entrypoint()
def main(params_file: str = "", screen_seed: int = 0, validation_seed: int = 100):
    parameters = (
        resolve_rapid_parameters(json.loads(Path(params_file).read_text(encoding="utf-8")))
        if params_file
        else rapid_parameters()
    )
    print(json.dumps(parameters, indent=2, allow_nan=False), flush=True)
    result = sweep.remote(parameters, screen_seed=screen_seed, validation_seed=validation_seed)
    directory = Path(__file__).parent / "results" / "sweeps" / result["run_id"]
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "modal_result.json", result)
    save_summaries(directory, result["rows"])
    for stage in sorted({row["stage"] for row in result["rows"]}):
        print(f"\n{stage.upper()} RESULTS")
        print_ranking([row for row in result["rows"] if row["stage"] == stage])
    print(f"Local summaries: {directory}")
    print(f"Full artifacts: Modal Volume {result['volume']}/{result['run_id']}")
