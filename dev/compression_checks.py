"""Synthetic correctness checks run before spending GPU time on the comparison."""

import argparse
import math
from pathlib import Path

import torch

from benchmark.api import BuildContext, TrainingData
from benchmark.worker import load_submission, seed_everything
from dev.compression_study import VARIANTS, write_json
from dev.compression_variants import normalize_uint8, zeropower_batched


def check_resets(root):
    data = TrainingData(torch.randint(0, 256, (40, 3, 32, 32), dtype=torch.uint8),
                        torch.arange(40) % 100)
    original = data.images.clone()
    results = {}
    baseline = None
    for name in VARIANTS:
        recipe = load_submission(root / "recipes" / name)
        epochs = VARIANTS[name].get("epochs", 1.25)
        parameters = dict(compile=False, widths=[8, 16, 16], batch_size=8,
                          epochs=epochs, whiten_bias_epochs=0.6)
        state = recipe.build(BuildContext(torch.device("cpu"), parameters))

        def run(seed):
            seed_everything(seed)
            recipe.prepare(state, data, seed)
            assert not state.sgd.state and not state.muon.state
            model = recipe.train(state)
            assert state.steps == math.ceil(epochs * 5)
            return {key: value.clone() for key, value in model.state_dict().items()}

        first, different, repeated = run(42), run(43), run(42)
        assert all(torch.equal(first[key], repeated[key]) for key in first), name
        assert any(not torch.equal(first[key], different[key]) for key in first), name
        assert torch.equal(data.images, original), name
        if name == "baseline":
            baseline = first
        if name == "fused_prepare":
            assert all(torch.equal(first[key], baseline[key]) for key in first)
        results[name] = dict(reset_after_training_exact=True, input_unchanged=True,
                             steps=state.steps, epochs=epochs)
    return results


def check_gpu(root):
    recipe = load_submission(root / "recipes" / "baseline")
    mean = torch.tensor(recipe.CIFAR100_MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(recipe.CIFAR100_STD, device="cuda").view(1, 3, 1, 1)
    # Cycle all 256 uint8 values through every channel at production shape.
    raw = (torch.arange(50_000 * 3 * 32 * 32, device="cuda", dtype=torch.int32) % 256)
    raw = raw.to(torch.uint8).reshape(50_000, 3, 32, 32)
    compiled = torch.compile(normalize_uint8, fullgraph=True, dynamic=False)
    actual = compiled(raw, mean, std, torch.float16)
    expected = normalize_uint8(raw, mean, std, torch.float16)
    rng = torch.cuda.get_rng_state().clone()
    actual = compiled(raw, mean, std, torch.float16)
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    assert actual.stride() == expected.stride()
    assert torch.isfinite(actual).all()
    difference = (actual.float() - expected.float()).abs()
    maximum = difference.max().item()
    assert maximum <= 0.002, maximum
    normalization = dict(shape=list(raw.shape), strides=list(actual.stride()),
                         exact=torch.equal(actual, expected), max_absolute_error=maximum,
                         different_fraction=(actual != expected).float().mean().item(),
                         rng_unchanged=True, all_uint8_values_checked=True)
    del raw, actual, expected, difference

    batched = torch.compile(zeropower_batched)
    serial = torch.compile(recipe.zeropower_via_newtonschulz5)
    muon = []
    for count, rows, columns in ((2, 128, 1152), (5, 512, 4608)):
        seed_everything(123)
        gradients = torch.randn(count, rows, columns, device="cuda", dtype=torch.float16)
        expected = torch.stack([serial(gradient) for gradient in gradients])
        actual = batched(gradients)
        difference = actual.float() - expected.float()
        relative = (difference.norm() / expected.float().norm()).item()
        assert torch.isfinite(actual).all() and relative < 0.05
        muon.append(dict(shape=[count, rows, columns], relative_l2_error=relative,
                         max_absolute_error=difference.abs().max().item(),
                         iterations=3, independent_matrix_normalization=True))
    return dict(normalization=normalization, batched_muon=muon)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    results = dict(cpu=check_resets(args.root), gpu=check_gpu(args.root))
    write_json(args.root / "validation.json", results)
    print(results, flush=True)
