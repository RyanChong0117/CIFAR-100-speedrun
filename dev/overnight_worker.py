"""Bounded experiment batches; only the trusted harness evaluates test data."""

import argparse
import ast
import hashlib
import json
import os
import pprint
import shutil
import subprocess
import sys
import time
from pathlib import Path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_defaults(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "DEFAULTS"
                              for t in node.targets))
    return ast.literal_eval(assignment.value)


def freeze(directory, experimental, default_parameters=None):
    directory.mkdir(parents=True, exist_ok=False)
    source = Path("submissions/airbench_muon/submission.py").read_bytes()
    (directory / "submission.py").write_bytes(source)
    if default_parameters:
        with (directory / "submission.py").open("a", encoding="utf-8") as file:
            file.write("\nDEFAULTS.update(" + pprint.pformat(default_parameters,
                                                             sort_dicts=True) + ")\n")
    if experimental:
        for name in ("overnight_variants.py",):
            shutil.copyfile(Path("dev") / name, directory / name)
        with (directory / "submission.py").open("a", encoding="utf-8") as file:
            file.write("\nfrom .overnight_variants import install as _install_overnight\n"
                       "_install_overnight(globals())\n")
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob("*.py")}


def freeze_verified(directory, specification):
    """Copy an archived recipe verbatim, checking its benchmark-recorded hashes."""
    results = Path('results').resolve()
    source = (results / specification['path']).resolve()
    if not source.is_relative_to(results):
        raise ValueError('Archived recipe must remain inside the results volume')
    expected = specification['sha256']
    if 'submission.py' not in expected:
        raise ValueError('Archived recipe must include submission.py')
    files = {}
    for name, checksum in expected.items():
        if Path(name).name != name or not name.endswith('.py'):
            raise ValueError('Archived recipe manifest must contain Python basenames')
        path = (source / name).resolve()
        if not path.is_relative_to(source):
            raise ValueError('Archived source file escapes its recipe directory')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError(f'Archived source hash mismatch: {name}')
        files[name] = data
    directory.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (directory / name).write_bytes(data)
    return dict(expected)


def stop_command_tree(process):
    """Stop this command's children, including the harness worker's own session.

    The harness deliberately calls setsid() in its worker. Killing just the
    supervisor's process group would leave that worker and compiler children
    alive. Freeze discovered descendants before the next scan so no child can
    continue spawning work while the tree is being terminated.
    """
    import signal

    if process.poll() is not None:
        return
    frozen = {process.pid}
    try:
        os.kill(process.pid, signal.SIGSTOP)
    except ProcessLookupError:
        return
    while True:
        discovered = set()
        for status in Path("/proc").glob("[0-9]*/status"):
            try:
                pid = int(status.parent.name)
                parent = next(int(line.split()[1]) for line in status.read_text().splitlines()
                              if line.startswith("PPid:"))
                if parent in frozen and pid not in frozen:
                    discovered.add(pid)
            except (OSError, StopIteration):
                continue
        if not discovered:
            break
        for pid in discovered:
            try:
                os.kill(pid, signal.SIGSTOP)
                frozen.add(pid)
            except ProcessLookupError:
                continue
    for pid in frozen - {process.pid}:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.kill(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def command_run(command, path, environment, timeout):
    """Keep all compiler diagnostics, stream only progress and benchmark output."""
    with path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, env=environment, start_new_session=True)
        started = time.monotonic()
        # Harness deadlines bound normal runs. A watchdog also bounds compilation
        # and validation commands that do not use the harness.
        import threading

        expired = threading.Event()

        def terminate():
            expired.set()
            stop_command_tree(process)

        timer = threading.Timer(timeout, terminate)
        timer.start()
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith(("trial ", "Results: ", "Profile", "error:", "Traceback")):
                    print(f"[{path.stem}] {line}", end="", flush=True)
            status = process.wait()
        finally:
            timer.cancel()
        return dict(returncode=status, elapsed_seconds=time.monotonic() - started,
                    timeout=expired.is_set())


def run(request):
    root = Path("results/overnight") / request["session_id"] / "batches" / request["batch_id"]
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "request.json", request)
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip()
    write_json(root / "hardware.json", {"nvidia_smi_gpu_name": gpu})
    print(f"Batch {request['batch_id']} GPU: {gpu}", flush=True)
    print(f"Checkpoint: {root.resolve()}", flush=True)
    base_path = Path("submissions/airbench_muon/submission.py")
    base_hash = hashlib.sha256(base_path.read_bytes()).hexdigest()
    if base_hash != request["protected_source_sha256"]:
        raise RuntimeError("Protected RC3 source changed before launch")
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                   "TORCHINDUCTOR_CACHE_DIR": f"/tmp/{request['session_id']}-{request['batch_id']}",
                   "TORCH_LOGS": "recompiles"}
    records = []
    try:
        if request.get("require_pcie") and gpu != "NVIDIA A100 80GB PCIe":
            write_json(root / "hardware_mismatch.json", dict(
                status="hardware_mismatch", actual=gpu, required="NVIDIA A100 80GB PCIe",
                conclusion="No finalist trials launched on incompatible hardware."))
            return
        if request.get("validate") or request.get("profile"):
            control = root / "recipes" / "rc3"
            freeze(control, False)
            for stage in ("validate", "profile"):
                if not request.get(stage):
                    continue
                remaining = request["deadline_unix"] - time.time()
                if remaining < 900:
                    raise RuntimeError("Session deadline too near to start profiling/validation")
                if stage == "validate":
                    command = [sys.executable, "-m", "dev.overnight_checks", "--submission",
                               str(control), "--output", str(root / "validation.json"), "--cuda"]
                else:
                    command = [sys.executable, "-m", "dev.rc3_profile", "--submission-path",
                               str(control), "--output", str(root / "profile.json"),
                               "--params", "{}", "--seed", "50", "--n", "3"]
                outcome = command_run(command, root / f"{stage}.log", environment,
                                      min(1200, remaining))
                write_json(root / f"{stage}_status.json", outcome)
                print(f"Checkpoint: {root.resolve()}", flush=True)
                if outcome["returncode"]:
                    raise RuntimeError(f"{stage} failed; see retained diagnostics")
        for candidate in request.get("experiments", []):
            remaining = request["deadline_unix"] - time.time()
            if remaining < 600 + 15 * candidate["n"]:
                records.append(dict(experiment_id=candidate["id"], status="not_started_deadline"))
                write_json(root / "batch_status.json", records)
                break
            name = candidate["id"]
            recipe = root / "recipes" / name
            experimental = any(key in candidate.get("params", {}) for key in (
                "ns_steps", "fast_reset", "crop_impl", "batched_muon", "stage_depths",
                "stage_residuals", "compiled_muon"))
            if candidate.get('verified_source'):
                if candidate.get('materialize_defaults'):
                    raise ValueError('Archived source must remain byte-exact')
                hashes = freeze_verified(recipe, candidate['verified_source'])
            else:
                hashes = freeze(recipe, experimental,
                                candidate.get('params') if candidate.get('materialize_defaults')
                                else None)
            overrides = candidate.get('params', {})
            params = {**({} if candidate.get('materialize_defaults') else overrides),
                      "experiment_name": name,
                      "hypothesis": candidate["hypothesis"]}
            effective = overrides | params
            resolved = {**source_defaults(base_path), **effective,
                        "ns_steps": effective.get("ns_steps", 3),
                        "fast_reset": effective.get("fast_reset", False),
                        "crop_impl": effective.get("crop_impl", "reference"),
                        "batched_muon": effective.get("batched_muon", False),
                        "compiled_muon": effective.get("compiled_muon", False),
                        "stage_depths": effective.get("stage_depths", [effective.get(
                            "depth", source_defaults(base_path)["depth"])] * 3)}
            resolved["stage_residuals"] = effective.get("stage_residuals", [
                depth >= 3 for depth in resolved["stage_depths"]])
            metadata = {**candidate, "experiment_id": name, "parameters": resolved,
                        "complete_parameters": resolved,
                        "parameters_complete": True, "campaign": request["session_id"],
                        "source_commit": request["source_commit"], "source_sha256": hashes,
                        "protected_source_sha256": base_hash, "gpu": gpu,
                        "nvidia_smi_gpu_name": gpu, "batch_id": request["batch_id"],
                        "session_id": request["session_id"]}
            write_json(root / f"{name}.json", metadata)
            command = [sys.executable, "-m", "benchmark.run", "--submission-path", str(recipe),
                       "--n", str(candidate["n"]), "--seed", str(candidate["seed"]),
                       "--params", json.dumps(params), "--results-root", str(root / "runs")]
            print(f"Starting {name}: n={candidate['n']} seed={candidate['seed']} params={params}",
                  flush=True)
            outcome = command_run(command, root / f"{name}.log", environment,
                                  min(remaining, 900 + 40 * candidate["n"]))
            directories = list((root / "runs" / name).glob("*"))
            complete = False
            if len(directories) == 1:
                write_json(directories[0] / "experiment.json", {**metadata, **outcome})
                summary = directories[0] / "summary.json"
                if summary.exists():
                    complete = json.loads(summary.read_text())["complete"]
            records.append({**metadata, **outcome})
            write_json(root / "batch_status.json", records)
            print(f"Checkpoint: {root.resolve()}", flush=True)
            if outcome["returncode"] not in (0, 1) or outcome["timeout"] or not complete:
                raise RuntimeError(f"Infrastructure failure in {name}; stopping batch")
    finally:
        print(f"Results: {root.resolve()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.request.read_text(encoding="utf-8")))
