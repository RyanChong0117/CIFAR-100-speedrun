from pathlib import Path
import subprocess
import modal

ROOT = Path(__file__).parent

image = modal.Image.from_dockerfile(
    ROOT / "Dockerfile",
    context_dir=ROOT,
)

app = modal.App("cifar100-speedrun")

# Persistent storage for CIFAR-100
data_volume = modal.Volume.from_name(
    "cifar100-data",
    create_if_missing=True,
)


@app.function(
    image=image,
    volumes={"/app/data": data_volume},
    cpu=2,
    timeout=600,
)
def download_data():
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "benchmark.data",
            "--root",
            "data",
        ],
        cwd="/app",
        check=True,
    )

    # Persist the downloaded dataset
    data_volume.commit()

    print("CIFAR-100 download complete.")

@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    volumes={"/app/data": data_volume},
    timeout=1200,
)
def benchmark():
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "benchmark.run",
            "--submission",
            "it_compiles",
            "--n",
            "1",
            "--no-accuracy-target",
        ],
        cwd="/app",
        check=True,
    )