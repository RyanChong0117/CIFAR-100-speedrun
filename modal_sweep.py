"""Sequential, persistent CIFAR-100 development sweeps on one Modal A100."""

import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

import modal

from modal_runner import data_volume, image
from sweep_utils import (
    phase1_parameters,
    print_ranking,
    require_sweep_gpu,
    resolve_parameters,
    run_sweep,
    save_summaries,
    write_json,
)

app = modal.App("cifar100-recipe-sweep")
results_volume = modal.Volume.from_name("cifar100-experiments", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    volumes={"/app/data": data_volume, "/sweep-results": results_volume},
    max_containers=1,
    retries=0,
    timeout=86400,
)
def sweep(parameters: list[dict], n: int = 3, seed: int = 0):
    # A single remote call holds one GPU for the entire sequential list. Each
    # configuration gets a fresh harness process; all seeds use normal timers.
    names = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        )
        .strip()
        .splitlines()
    )
    print(f"Provider GPU: {names}", flush=True)
    require_sweep_gpu(names)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = Path("/sweep-results") / run_id
    print(f"Persistent results: cifar100-experiments/{run_id}", flush=True)
    rows = run_sweep(parameters, directory, n=n, seed=seed, checkpoint=results_volume.commit)
    return {"run_id": run_id, "volume": "cifar100-experiments", "rows": rows}


@app.local_entrypoint()
def main(params_file: str = "", n: int = 3, seed: int = 0, dry_run: bool = False):
    """Default: 15 phase-1 configurations. Custom file: JSON list of overrides."""
    parameters = (
        json.loads(Path(params_file).read_text(encoding="utf-8"))
        if params_file
        else [params | {"architecture": "resnet9"} for params in phase1_parameters()]
    )
    parameters = resolve_parameters(parameters)
    if n < 1 or not 0 <= seed or seed + n > 2**32:
        raise ValueError("Use n >= 1 and an unsigned 32-bit seed range")
    # Keep worst-case harness deadlines plus startup/I/O headroom below Modal's
    # 24-hour function limit. Long validation lists must be split, not cut short.
    if len(parameters) * (900 + n * 610) >= 86400:
        raise ValueError("Sweep too large for one Modal call; split the configuration list")
    print(f"{len(parameters)} configurations; shared seeds {list(range(seed, seed + n))}")
    print(json.dumps(parameters, indent=2, allow_nan=False), flush=True)
    if dry_run:
        print("Dry run: no benchmark function invoked.")
        return
    result = sweep.remote(parameters, n=n, seed=seed)
    local_dir = Path(__file__).parent / "results" / "sweeps" / result["run_id"]
    local_dir.mkdir(parents=True, exist_ok=False)
    write_json(local_dir / "modal_result.json", result)
    save_summaries(local_dir, result["rows"])
    print_ranking(result["rows"])
    print(f"Local summaries: {local_dir}")
    print(f"Full artifacts remain in Modal Volume {result['volume']}/{result['run_id']}")
