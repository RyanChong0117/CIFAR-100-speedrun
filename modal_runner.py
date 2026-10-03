from pathlib import Path

import modal

ROOT = Path(__file__).parent

# Use the hackathon's existing Dockerfile.
image = modal.Image.from_dockerfile(
    ROOT / "Dockerfile",
    context_dir=ROOT,
)

app = modal.App("cifar100-speedrun")


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    timeout=600,
)
def check_environment():
    import os
    import torch

    print("Repo mounted:", os.path.exists("/app/benchmark/run.py"))
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU memory:",
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
            "GB",
        )

@app.function(
    image=image,
    cpu=4,
    timeout=600,
)
def smoke_test():
    import subprocess

    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "benchmark.run",
            "--submission-path",
            "submission_template",
            "--device",
            "cpu",
            "--synthetic",
            "--n",
            "2",
        ],
        cwd="/app",
        check=True,
    )


@app.local_entrypoint()
def main():
    check_environment.remote()