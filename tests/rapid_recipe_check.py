"""Pinned-environment synthetic training and native OneCycleLR equivalence checks."""

from pathlib import Path

import torch

from benchmark.api import BuildContext, TrainingData
from benchmark.worker import load_submission, seed_everything
from rapid_convergence_experiments import rapid_parameters


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    module = load_submission(Path("submissions/it_compiles"))
    context = BuildContext(torch.device("cpu"), {})
    state = module.build(context)
    assert state.epochs == 30 and state.architecture["architecture"] == "resnet11"
    assert state.architecture["stage_widths"] == [64, 128, 256, 512]
    del state
    for warmup in (98, 196):
        total, peak = 2940, 0.28
        expected = module._learning_rates(total, warmup, peak, {"lr_schedule": "one_cycle"})
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=peak, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=peak,
            total_steps=total,
            pct_start=warmup / total,
            anneal_strategy="cos",
            cycle_momentum=False,
            div_factor=25,
            final_div_factor=10000,
        )
        for rate in expected:
            assert abs(scheduler.get_last_lr()[0] - rate) < 1e-12
            assert optimizer.param_groups[0]["momentum"] == 0.9
            optimizer.step()
            scheduler.step()
        print(
            f"PASS: native PyTorch 2.4 OneCycleLR equivalence, warmup={warmup} updates.", flush=True
        )

    data = TrainingData(torch.randint(256, (5, 3, 32, 32), dtype=torch.uint8), torch.arange(5))
    before_images, before_labels = data.images.clone(), data.labels.clone()
    for index, params in enumerate(rapid_parameters(), 1):
        # Synthetic shortened checks only; the real sweep always uses 30 epochs.
        state = module.build(BuildContext(context.device, params | {"epochs": 3, "batch_size": 4}))
        seed_everything(17)
        module.prepare(state, data, 17)
        initial = {key: value.clone() for key, value in state.model.state_dict().items()}
        schedule = list(state.learning_rates)
        old_optimizer = state.optimizer
        model = module.train(state)
        assert state.step == 6 and not model.training
        after = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.inference_mode():
            outputs = model(torch.rand(1, 3, 32, 32))
            assert outputs.shape == (1, 100) and torch.isfinite(outputs).all()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, after[key], rtol=0, atol=0)
        seed_everything(17)
        module.prepare(state, data, 17)
        assert state.step == 0 and state.learning_rates == schedule
        assert state.optimizer is not old_optimizer and not state.optimizer.state
        for key, value in state.model.state_dict().items():
            torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
        assert torch.equal(data.images, before_images) and torch.equal(data.labels, before_labels)
        del state, model, old_optimizer, initial, after
        print(
            f"PASS: recipe {index}, training and remainder, immutable inference/data, full reset.",
            flush=True,
        )


if __name__ == "__main__":
    main()
