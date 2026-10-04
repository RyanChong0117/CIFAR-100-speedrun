"""Launch bounded, independent RC3 experiment batches on at most four A100s."""

import json
import subprocess
import time
from pathlib import Path

import modal

data_volume = modal.Volume.from_name("cifar100-data", create_if_missing=True)
results_volume = modal.Volume.from_name("cifar100-results", create_if_missing=True)
image = None
if modal.is_local():
    # Only the local launcher needs the Docker context and local result writer.
    # Remote hydration mounts this entrypoint, not its sibling runner module.
    from modal_runner import image, save_locally

app = modal.App("cifar100-rc3-overnight")


@app.function(image=image, cpu=1, timeout=60)
def watchdog_probe() -> dict:
    from dev.overnight_watchdog_check import run

    return run()


@app.local_entrypoint()
def check_watchdog():
    result = watchdog_probe.remote()
    path = Path("results/overnight/rc3-20261004/watchdog-validation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


@app.function(image=image, gpu="A100-80GB", cpu=4, max_containers=4,
              volumes={"/app/data": data_volume, "/app/results": results_volume}, timeout=7200)
def batch(request: dict) -> dict:
    path = Path("/tmp") / f"{request['batch_id']}.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    started = time.time()
    process = subprocess.Popen(
        ["uv", "run", "python", "-m", "dev.overnight_worker", "--request", str(path)],
        cwd="/app", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    result_path = None
    for line in process.stdout:
        print(line, end="", flush=True)
        if line.startswith(("Checkpoint: ", "Results: ")):
            result_path = Path(line.split(": ", 1)[1].strip())
            results_volume.commit()
    returncode = process.wait()
    results_volume.commit()
    files = {}
    if result_path is not None:
        root = Path("/app/results").resolve()
        for file in result_path.rglob("*"):
            if file.is_file() and "__pycache__" not in file.parts:
                files[str(file.relative_to(root))] = file.read_text(encoding="utf-8")
    return dict(returncode=returncode, files=files, wall_seconds=time.time() - started)


@app.local_entrypoint()
def wave(file: str):
    requests = json.loads(Path(file).read_text(encoding="utf-8"))
    if not 1 <= len(requests) <= 4:
        raise ValueError("A wave must contain one to four independent GPU batches")
    if any(time.time() >= request["deadline_unix"] for request in requests):
        raise ValueError("The session deadline has elapsed")
    calls = [(request, batch.spawn(request)) for request in requests]
    print(f"Launched {len(calls)} batches, max four GPUs", flush=True)
    failures = []
    for request, call in calls:
        try:
            output = call.get()
            save_locally(output)
            receipt = Path("results/overnight") / request["session_id"] / "receipts"
            receipt.mkdir(parents=True, exist_ok=True)
            (receipt / f"{request['batch_id']}.json").write_text(
                json.dumps({k: v for k, v in output.items() if k != "files"}, indent=2) + "\n",
                encoding="utf-8")
            if output["returncode"]:
                failures.append(request["batch_id"])
        except Exception as exc:
            print(f"Batch {request['batch_id']} infrastructure error: {exc!r}", flush=True)
            failures.append(request["batch_id"])
    if failures:
        raise RuntimeError(f"Failed batches retained for review: {failures}")
