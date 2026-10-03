"""ResNet9-style CIFAR-100 baseline: the reference point every later change is measured against.

Deliberately conventional: SGD + Nesterov, one-cycle LR, label smoothing, random
crop + flip, bf16 autocast, channels_last, GPU-resident data. No torch.compile.

Every knob is a JSON parameter (see DEFAULTS); unknown keys such as
"experiment_name" or "hypothesis" are ignored so they can ride along into config.json.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from benchmark.api import BuildContext, TrainingData

DEFAULTS = {
    "epochs": 40,
    "batch_size": 512,
    "width": 64,
    "lr": 0.2,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "label_smoothing": 0.1,
    "warmup_frac": 0.2,
    "crop_pad": 4,
    "flip": True,
}

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def conv_bn(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class Residual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(conv_bn(channels, channels), conv_bn(channels, channels))

    def forward(self, x):
        return x + self.body(x)


class ResNet9(nn.Module):
    """Global pooling before the classifier, so any input resolution works."""

    def __init__(self, width: int, num_classes: int):
        super().__init__()
        w = width
        self.register_buffer("mean", torch.tensor(CIFAR100_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR100_STD).view(1, 3, 1, 1))
        self.net = nn.Sequential(
            conv_bn(3, w),
            conv_bn(w, 2 * w),
            nn.MaxPool2d(2),
            Residual(2 * w),
            conv_bn(2 * w, 4 * w),
            nn.MaxPool2d(2),
            conv_bn(4 * w, 8 * w),
            nn.MaxPool2d(2),
            Residual(8 * w),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(8 * w, num_classes, bias=False),
        )

    def forward(self, x):
        # Input is float in [0, 1]; normalisation lives in the model for train and eval alike.
        return self.net((x - self.mean) / self.std) * 0.125


def build(context: BuildContext):
    cfg = {**DEFAULTS, **context.parameters}
    model = ResNet9(cfg["width"], context.num_classes)
    model = model.to(context.device, memory_format=torch.channels_last)
    return SimpleNamespace(model=model, context=context, cfg=cfg, epoch_callback=None)


def prepare(state, data: TrainingData, seed: int) -> None:
    cfg, device = state.cfg, state.context.device
    for module in state.model.modules():
        if isinstance(module, nn.Conv2d | nn.Linear | nn.BatchNorm2d):
            module.reset_parameters()  # BatchNorm also resets its running statistics
    state.model.train()

    decay = [p for p in state.model.parameters() if p.ndim > 1]
    no_decay = [p for p in state.model.parameters() if p.ndim <= 1]
    state.optimizer = torch.optim.SGD(
        [
            {"params": decay, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        nesterov=True,
    )

    # GPU-resident data, padded once so each epoch's random crops are a cheap gather.
    pad = cfg["crop_pad"]
    images = data.images.to(device, non_blocking=True)
    if pad:
        images = F.pad(images, (pad, pad, pad, pad))  # zero padding
    state.images = images
    state.labels = data.labels.to(device, non_blocking=True)
    state.generator = torch.Generator(device=device).manual_seed(seed)


def augmented_batch(state, index: torch.Tensor) -> torch.Tensor:
    """Random crop (translation by up to crop_pad pixels) + horizontal flip, on GPU."""
    cfg = state.cfg
    pad, n = cfg["crop_pad"], len(index)
    batch = state.images[index]
    if pad:
        dy = torch.randint(0, 2 * pad + 1, (n,), device=batch.device, generator=state.generator)
        dx = torch.randint(0, 2 * pad + 1, (n,), device=batch.device, generator=state.generator)
        ar = torch.arange(32, device=batch.device)
        rows = (dy[:, None] + ar)[:, None, :, None]
        cols = (dx[:, None] + ar)[:, None, None, :]
        batch = batch[
            torch.arange(n, device=batch.device)[:, None, None, None],
            torch.arange(3, device=batch.device)[None, :, None, None],
            rows,
            cols,
        ]
    if cfg["flip"]:
        mask = torch.rand(n, device=batch.device, generator=state.generator) < 0.5
        batch = torch.where(mask[:, None, None, None], batch.flip(3), batch)
    batch = batch.float().div_(255)
    return batch.contiguous(memory_format=torch.channels_last)


def train(state) -> nn.Module:
    cfg, model, optimizer = state.cfg, state.model, state.optimizer
    n, bs = len(state.labels), cfg["batch_size"]
    steps_per_epoch = n // bs
    total_steps = cfg["epochs"] * steps_per_epoch
    warmup = max(1, int(cfg["warmup_frac"] * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return cfg["lr"] * step / warmup
        return cfg["lr"] * (total_steps - step) / (total_steps - warmup)

    step = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        order = torch.randperm(n, device=state.labels.device, generator=state.generator)
        for i in range(steps_per_epoch):
            index = order[i * bs : (i + 1) * bs]
            inputs = augmented_batch(state, index)
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=inputs.is_cuda):
                logits = model(inputs)
            loss = F.cross_entropy(
                logits.float(), state.labels[index], label_smoothing=cfg["label_smoothing"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
        # Dev-only hook (dev/curve.py); always None under the harness.
        if state.epoch_callback is not None:
            state.epoch_callback(epoch + 1, step)
    state.steps = step
    return model
