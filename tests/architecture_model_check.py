"""Standalone synthetic model checks for the pinned Modal CPU environment.

No accuracy or performance claims are made from these generated inputs.
"""

from pathlib import Path

import torch
from torch import nn

from architecture_experiments import architecture_parameters
from benchmark.api import BuildContext, TrainingData
from benchmark.worker import load_submission, seed_everything


def legacy_model(module):
    """The pre-refactor control, for exact seeded numerical equivalence on CPU."""

    class LegacyResNet9(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
            self.register_buffer("std", torch.ones(1, 3, 1, 1))
            block = module._conv_block
            self.stem = block(3, 64)
            self.stage1 = block(64, 128, pool=True)
            self.residual1 = nn.Sequential(block(128, 128), block(128, 128))
            self.stage2 = block(128, 256, pool=True)
            self.stage3 = block(256, 512, pool=True)
            self.residual3 = nn.Sequential(block(512, 512), block(512, 512))
            self.pool = nn.AdaptiveMaxPool2d(1)
            self.head = nn.Linear(512, 100)

        def forward(self, images):
            x = ((images - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
            x = self.stage1(self.stem(x))
            x = x + self.residual1(x)
            x = self.stage3(self.stage2(x))
            x = x + self.residual3(x)
            return self.head(self.pool(x).flatten(1))

    return LegacyResNet9().to(memory_format=torch.channels_last)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    module = load_submission(Path("submissions/it_compiles"))
    device = torch.device("cpu")
    torch.manual_seed(73)
    legacy = legacy_model(module).eval()
    torch.manual_seed(73)
    default = module.build(BuildContext(device, {"architecture": "resnet9"}))
    assert default.epochs == 30
    default.model.eval()
    for old, new in zip(
        legacy.state_dict().values(), default.model.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(old, new, rtol=0, atol=0)
    inputs = torch.rand(3, 3, 32, 32)
    with torch.inference_mode():
        torch.testing.assert_close(legacy(inputs), default.model(inputs), rtol=0, atol=0)
    del legacy, default
    print(
        "PASS: explicit ResNet9 is numerically equivalent; default budget is 30 epochs.", flush=True
    )

    data = TrainingData(torch.randint(256, (5, 3, 32, 32), dtype=torch.uint8), torch.arange(5))
    original_images, original_labels = data.images.clone(), data.labels.clone()
    for index, params in enumerate(architecture_parameters(), 1):
        # Short synthetic checks exercise the unchanged training API, including a
        # remainder batch. These overrides are never used in the real sweep.
        state = module.build(BuildContext(device, params | {"epochs": 1, "batch_size": 4}))
        seed_everything(17)
        module.prepare(state, data, 17)
        initial = {key: value.clone() for key, value in state.model.state_dict().items()}
        initial_rng = state.rng.get_state().clone()
        optimizer = state.optimizer
        model = module.train(state)
        assert state.step == 2
        assert not model.training
        after_training = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.inference_mode():
            for batch_size in (1, 3):
                logits = model(inputs[:batch_size])
                assert logits.shape == (batch_size, 100)
                assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
            # Exercise the largest contract batch for the control; the real GPU
            # benchmarks exercise 1024 for every architecture.
            if index == 1:
                logits = model(torch.zeros(1024, 3, 32, 32))
                assert logits.shape == (1024, 100) and torch.isfinite(logits).all()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, after_training[key], rtol=0, atol=0)
        seed_everything(17)
        module.prepare(state, data, 17)
        assert state.optimizer is not optimizer and not state.optimizer.state
        assert state.step == 0
        torch.testing.assert_close(state.rng.get_state(), initial_rng, rtol=0, atol=0)
        for key, value in state.model.state_dict().items():
            torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
        assert torch.equal(data.images, original_images) and torch.equal(
            data.labels, original_labels
        )
        del initial, after_training, optimizer, model, state
        print(
            f"PASS: architecture {index}, training, finite/read-only inference, complete reset.",
            flush=True,
        )


if __name__ == "__main__":
    main()
