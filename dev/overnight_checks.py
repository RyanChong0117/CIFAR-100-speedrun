"""Synthetic validation for opt-in overnight variants; never reads CIFAR data."""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmark.api import BuildContext, TrainingData
from benchmark.worker import load_submission, seed_everything
from dev.overnight_variants import (
    batch_crop_vectorized,
    crop_with_shifts,
    fast_conv_reset,
    install,
    zeropower_batched,
)


def rng_state(device):
    values = [torch.get_rng_state().clone()]
    if device.type == "cuda":
        values.append(torch.cuda.get_rng_state(device).clone())
    return values


def equal_states(left, right):
    return left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)


def check_fast_reset(recipe, device):
    original = recipe.Conv.reset_parameters
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    parameters = dict(widths=[128, 512, 512], num_classes=100,
                      bn_momentum=0.6, dtype=dtype, depth=3)
    # Constructor draws run on CPU; per-trial reset runs on the target device.
    seed_everything(501)
    reference = recipe.CifarNet(**parameters).to(device, memory_format=torch.channels_last)
    constructor_rng = rng_state(device)
    recipe.Conv.reset_parameters = fast_conv_reset
    seed_everything(501)
    actual = recipe.CifarNet(**parameters).to(device, memory_format=torch.channels_last)
    assert equal_states(reference.state_dict(), actual.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(constructor_rng, rng_state(device), strict=True))
    for seed in (502, 503, 502):
        recipe.Conv.reset_parameters = original
        seed_everything(seed)
        reference.reset()
        expected_rng = rng_state(device)
        recipe.Conv.reset_parameters = fast_conv_reset
        seed_everything(seed)
        actual.reset()
        assert equal_states(reference.state_dict(), actual.state_dict())
        assert all(torch.equal(a, b) for a, b in zip(expected_rng, rng_state(device), strict=True))
        for ref, fast in zip(reference.parameters(), actual.parameters(), strict=True):
            assert ref.stride() == fast.stride()
    recipe.Conv.reset_parameters = original
    return dict(exact_weights=True, exact_rng=True, exact_strides=True,
                constructor_checked=True, reset_seeds=[502, 503, 502],
                widths=parameters["widths"], dtype=str(dtype))


def check_crop(recipe, device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    rows = []
    for radius in (0, 1, 2, 3):
        shifts = torch.tensor([(y, x) for y in range(-radius, radius + 1)
                               for x in range(-radius, radius + 1)], device=device)
        size = 32 + radius * 2
        images = torch.randint(0, 1024, (len(shifts), 3, size, size), device=device)
        images = images.to(dtype).contiguous(memory_format=torch.channels_last)
        expected = torch.stack([images[i, :, radius + y:radius + y + 32,
                                       radius + x:radius + x + 32]
                                for i, (y, x) in enumerate(shifts.cpu().tolist())])
        actual = crop_with_shifts(images, shifts, 32)
        assert torch.equal(actual, expected)
        assert actual.is_contiguous()
        seed_everything(700 + radius)
        expected = recipe.batch_crop(images, 32)
        expected_rng = rng_state(device)
        seed_everything(700 + radius)
        actual = batch_crop_vectorized(images, 32)
        assert torch.equal(actual, expected) and actual.stride() == expected.stride()
        assert all(torch.equal(a, b) for a, b in zip(expected_rng, rng_state(device), strict=True))
        rows.append(dict(radius=radius, exhaustive_offsets=len(shifts), exact_pixels=True,
                         exact_strides=True, exact_rng=True))
    if device.type == "cuda":
        # Full-size synthetic prepare obtains the true post-flip/pad layout.
        # A small model is sufficient because data layout is architecture independent.
        model = recipe.CifarNet([8, 16, 16], 100, 0.6, dtype, 3)
        model = model.to(device, memory_format=torch.channels_last)
        state = SimpleNamespace(model=model, cfg=dict(recipe.DEFAULTS), device=device,
                                dtype=dtype, zeropower=recipe.zeropower_via_newtonschulz5)
        generator = torch.Generator().manual_seed(709)
        data = TrainingData(
            torch.randint(0, 256, (50_000, 3, 32, 32), dtype=torch.uint8,
                          generator=generator),
            torch.randint(0, 100, (50_000,), generator=generator),
        )
        seed_everything(710)
        recipe.prepare(state, data, 710)
        images = state.padded_images
        compiled = torch.compile(crop_with_shifts, fullgraph=True, dynamic=False)
        alternate_format = (torch.channels_last if images.is_contiguous()
                            else torch.contiguous_format)
        for label, source in (("actual_prepare", images),
                              ("alternate_layout", images.contiguous(
                                  memory_format=alternate_format))):
            seed_everything(711)
            expected = recipe.batch_crop(source, 32)
            expected_rng = rng_state(device)
            seed_everything(711)
            actual = batch_crop_vectorized(source, 32, kernel=compiled)
            assert torch.equal(actual, expected) and actual.stride() == expected.stride()
            assert all(torch.equal(a, b) for a, b in zip(
                expected_rng, rng_state(device), strict=True))
            rows.append(dict(compiled=True, layout=label, shape=list(source.shape),
                             input_strides=list(source.stride()), exact_pixels=True,
                             output_strides=list(actual.stride()), exact_rng=True))
    return rows


def check_newton_schulz(recipe, device):
    original = recipe.zeropower_via_newtonschulz5
    serial = torch.compile(original) if device.type == "cuda" else original
    batched = torch.compile(zeropower_batched) if device.type == "cuda" else zeropower_batched
    shapes = ((2, 128, 1152), (5, 512, 4608)) if device.type == "cuda" else (
        (2, 8, 72), (3, 16, 144), (2, 24, 8))
    rows = []
    for shape in shapes:
        for steps in (3, 2, 1):
            seed_everything(801)
            matrices = torch.randn(shape, device=device, dtype=torch.float32)
            expected = torch.stack([serial(matrix, steps=steps) for matrix in matrices])
            actual = batched(matrices, steps=steps)
            difference = actual.float() - expected.float()
            relative = (difference.norm() / expected.float().norm()).item()
            assert torch.isfinite(actual).all() and relative < 0.05, (shape, steps, relative)
            rows.append(dict(shape=list(shape), steps=steps, relative_l2_error=relative,
                             max_absolute_error=difference.abs().max().item(),
                             independent_normalization=True))
    return rows


def check_install_and_resets(submission):
    # A full miniature learning loop checks optimizers, BatchNorm, whitening and
    # every optional implementation independently, including post-training reset.
    data = TrainingData(torch.randint(0, 256, (40, 3, 32, 32), dtype=torch.uint8),
                        torch.arange(40) % 100)
    data_copy = data.images.clone()
    rows = []
    baseline = None
    configurations = [dict(), dict(ns_steps=3), dict(stage_depths=[3, 3, 3]),
                      dict(stage_depths=[3, 2, 3]), dict(stage_depths=[2, 3, 3]),
                      dict(stage_depths=[3, 3, 2]), dict(fast_reset=True),
                      dict(crop_impl="vectorized"), dict(ns_steps=2), dict(ns_steps=1),
                      dict(batched_muon=True), dict(fast_reset=True, crop_impl="vectorized",
                                                   batched_muon=True, ns_steps=2)]
    for extras in configurations:
        recipe = load_submission(submission)
        original_ns = recipe.zeropower_via_newtonschulz5
        install(vars(recipe))
        parameters = dict(compile=False, widths=[8, 16, 16], batch_size=8,
                          epochs=1.2, whiten_bias_epochs=0.6, **extras)
        state = recipe.build(BuildContext(torch.device("cpu"), parameters))
        actual_depths = [1 + len(state.model.layers[i].convs) for i in (1, 2, 3)]
        assert actual_depths == extras.get("stage_depths", [3, 3, 3])
        gradient = torch.randn(8, 72)
        assert torch.equal(state.zeropower(gradient),
                           original_ns(gradient, steps=extras.get("ns_steps", 3)))

        def trial(seed):
            seed_everything(seed)
            recipe.prepare(state, data, seed)
            assert not state.sgd.state and not state.muon.state
            recipe.train(state)
            assert state.steps == math.ceil(1.2 * 5)
            assert torch.equal(data.images, data_copy)
            return {key: value.clone() for key, value in state.model.state_dict().items()}

        first, different, repeated = trial(901), trial(902), trial(901)
        assert equal_states(first, repeated), extras
        assert not equal_states(first, different), extras
        if baseline is None:
            baseline = first
        exact_control = not extras or extras in (
            dict(ns_steps=3), dict(stage_depths=[3, 3, 3]),
            dict(fast_reset=True), dict(crop_impl="vectorized"))
        if exact_control:
            assert equal_states(first, baseline), extras
        rows.append(dict(parameters=extras, reset_exact=True, input_unchanged=True,
                         steps=state.steps, stage_depths=actual_depths,
                         identical_to_control=equal_states(first, baseline)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda" if args.cuda else "cpu")
    recipe = load_submission(args.submission)
    result = dict(device=str(device), fast_reset=check_fast_reset(recipe, device),
                  crop=check_crop(recipe, device),
                  newton_schulz=check_newton_schulz(recipe, device),
                  install_and_reset=check_install_and_resets(args.submission))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
