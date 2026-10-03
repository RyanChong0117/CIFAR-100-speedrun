"""Parallel single-GPU architecture screening, followed by automatic validation."""

import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

import modal

from architecture_experiments import (
    architecture_parameters,
    resolve_architecture_parameters,
    run_architecture_stages,
)
from modal_runner import data_volume, image
from sweep_utils import print_ranking, require_sweep_gpu, run_sweep, save_summaries, write_json

app = modal.App("cifar100-architecture-sweep")
results_volume = modal.Volume.from_name("cifar100-experiments", create_if_missing=True)
architecture_image = image.add_local_file(
    Path(__file__).parent / "tests" / "architecture_model_check.py",
    "/app/architecture_model_check.py",
)


@app.function(
    image=architecture_image,
    cpu=4,
    memory=8192,
    timeout=600,
    retries=0,
)
def check_models():
    """Synthetic CPU interface/reset checks, before spending GPU time."""
    subprocess.run(["uv", "run", "python", "architecture_model_check.py"], cwd="/app", check=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    volumes={"/app/data": data_volume, "/sweep-results": results_volume},
    max_containers=8,
    retries=0,
    timeout=3600,
)
def benchmark_architecture(job: dict):
    # Modal's default input concurrency is one: independent jobs never share a
    # GPU concurrently. Eight containers allow up to eight independent GPUs.
    names = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        )
        .strip()
        .splitlines()
    )
    print(f"{job['stage']} {job['configuration']} provider GPU: {names}", flush=True)
    require_sweep_gpu(names)
    return run_sweep(
        [job["parameters"]],
        Path(job["directory"]),
        n=job["n"],
        seed=job["seed"],
        checkpoint=results_volume.commit,
    )[0]


@app.function(
    image=image,
    cpu=1,
    volumes={"/sweep-results": results_volume},
    timeout=14400,
    retries=0,
)
def sweep(parameters: list[dict], screen_seed: int = 0, validation_seed: int = 100):
    # The remote CPU coordinator survives a disconnected local terminal and
    # performs promotion without changing a recipe inside any benchmark trial.
    check_models.remote()
    run_id = "architectures-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id += "-" + uuid.uuid4().hex[:8]
    directory = Path("/sweep-results") / run_id
    print(f"Persistent results: cifar100-experiments/{run_id}", flush=True)

    def execute_stage(jobs):
        for result in benchmark_architecture.map(jobs, return_exceptions=True):
            # Child containers commit disjoint directories. Refresh the CPU
            # coordinator before writing its consolidated summaries.
            results_volume.reload()
            yield result

    rows = run_architecture_stages(
        parameters,
        directory,
        execute_stage=execute_stage,
        checkpoint=results_volume.commit,
        screen_seed=screen_seed,
        validation_seed=validation_seed,
    )
    return {"run_id": run_id, "volume": "cifar100-experiments", "rows": rows}


@app.local_entrypoint()
def main(params_file: str = "", screen_seed: int = 0, validation_seed: int = 100):
    parameters = (
        resolve_architecture_parameters(json.loads(Path(params_file).read_text(encoding="utf-8")))
        if params_file
        else architecture_parameters()
    )
    print(json.dumps(parameters, indent=2, allow_nan=False), flush=True)
    result = sweep.remote(parameters, screen_seed=screen_seed, validation_seed=validation_seed)
    directory = Path(__file__).parent / "results" / "sweeps" / result["run_id"]
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "modal_result.json", result)
    save_summaries(directory, result["rows"])
    for stage in ("screen", "validation"):
        rows = [row for row in result["rows"] if row["stage"] == stage]
        if rows:
            print(f"\n{stage.upper()} RESULTS")
            print_ranking(rows)
    print(f"Local summaries: {directory}")
    print(f"Full artifacts: Modal Volume {result['volume']}/{result['run_id']}")
