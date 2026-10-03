"""Configurable CIFAR residual networks; fresh SGD, GPU augmentation, BF16."""

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from benchmark.api import BuildContext, TrainingData

from .architecture_config import resolve_architecture
from .lr_schedule import learning_rates as _learning_rates
from .lr_schedule import resolve_schedule


def _conv_block(in_channels, out_channels, pool=False, downsampling="maxpool"):
    layers = [
        nn.Conv2d(
            in_channels, out_channels, 3, padding=1,
            stride=2 if pool and downsampling == "stride" else 1, bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool and downsampling != "stride":
        layers.append(nn.MaxPool2d(2) if downsampling == "maxpool" else nn.AvgPool2d(2))
    return nn.Sequential(*layers)


class _ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            _conv_block(channels, channels), _conv_block(channels, channels)
        )

    def forward(self, inputs):
        # Preserve the baseline's two Conv-BN-ReLU blocks and unscaled skip.
        return inputs + self.layers(inputs)


def _residual_stack(channels, count):
    return nn.Sequential(*(_ResidualBlock(channels) for _ in range(count)))


class CifarResNet(nn.Module):
    """CIFAR stem, three downsampled stages, configurable depth and widths."""

    def __init__(self, specification, num_classes):
        super().__init__()
        # These are fitted only on the current trial's training data in prepare.
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("std", torch.ones(1, 3, 1, 1))
        w0, w1, w2, w3 = specification["stage_widths"]
        d1, d2, d3 = specification["residual_blocks"]
        downsampling = specification["downsampling"]
        self.stem = _conv_block(3, w0)
        self.stage1 = _conv_block(w0, w1, pool=True, downsampling=downsampling)
        self.residual1 = _residual_stack(w1, d1)
        self.stage2 = _conv_block(w1, w2, pool=True, downsampling=downsampling)
        self.residual2 = _residual_stack(w2, d2)
        self.stage3 = _conv_block(w2, w3, pool=True, downsampling=downsampling)
        self.residual3 = _residual_stack(w3, d3)
        self.pool = (
            nn.AdaptiveMaxPool2d(1)
            if specification["global_pool"] == "max" else nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Linear(w3, num_classes)

    def forward(self, inputs, *, normalized=False):
        # The harness supplies float32 RGB in [0, 1]. Training uses the same
        # normalization, cached in prepare to avoid repeating it every epoch.
        if not normalized:
            inputs = (inputs - self.mean) / self.std
        inputs = inputs.contiguous(memory_format=torch.channels_last)
        # Keep parameters and optimizer state FP32. BF16 has FP32's exponent
        # range and does not require a gradient scaler. CPU smoke tests use FP32.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=inputs.is_cuda):
            x = self.stage1(self.stem(inputs))
            x = self.residual1(x)
            x = self.residual2(self.stage2(x))
            x = self.residual3(self.stage3(x))
            logits = self.head(self.pool(x).flatten(1))
        return logits.float()


class ResNet9(CifarResNet):
    """The original widths, two residual blocks, max pooling, and classifier."""

    def __init__(self, width, num_classes):
        super().__init__(
            resolve_architecture({"architecture": "resnet9", "width": width}), num_classes
        )


def build(context: BuildContext):
    """Allocate architecture/resources; never access real data or trial seeds."""
    params = context.parameters
    epochs = int(params.get("epochs", 30))
    width = int(params.get("width", 64))
    batch_size = int(params.get("batch_size", 512))
    warmup_epochs = int(params.get("warmup_epochs", 5))
    cutout = int(params.get("cutout", 8))
    smoothing = float(params.get("label_smoothing", 0.05))
    lr = float(params.get("lr", 0.2 * batch_size / 512))
    weight_decay = float(params.get("weight_decay", 5e-4))
    momentum = float(params.get("momentum", 0.9))
    if min(epochs, width, batch_size) < 1 or warmup_epochs < 0:
        raise ValueError("epochs, width, batch_size must be positive; warmup_epochs >= 0")
    if not 0 <= cutout <= 32 or not 0 <= smoothing < 1:
        raise ValueError("cutout must be in [0, 32]; label_smoothing in [0, 1)")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not math.isfinite(momentum) or not 0 <= momentum < 1:
        raise ValueError("momentum must be finite and in [0, 1)")

    specification = resolve_architecture(params)
    schedule = resolve_schedule(params)
    model = CifarResNet(specification, context.num_classes).to(
        device=context.device, memory_format=torch.channels_last
    )
    state = SimpleNamespace(
        context=context,
        model=model,
        architecture=specification,
        schedule=schedule,
        epochs=epochs,
        batch_size=batch_size,
        warmup_epochs=warmup_epochs,
        cutout=cutout,
        smoothing=smoothing,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        rows=torch.arange(32, device=context.device).view(1, 32, 1),
        cols=torch.arange(32, device=context.device).view(1, 1, 32),
    )
    if context.device.type == "cuda":
        # Ordinary cuDNN autotuning, not compilation or custom kernels.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        # Synthetic warmup covers full/remainder training shapes and inference
        # shapes. Real data-dependent autotuning remains inside the trial timer.
        # Every parameter and BatchNorm buffer touched here is reset in prepare.
        sizes = {batch_size, 50000 % batch_size}
        model.train()
        for size in sorted(sizes - {0}):
            inputs = torch.zeros(size, 3, 32, 32, device=context.device)
            model(inputs).square().mean().backward()
            model.zero_grad(set_to_none=True)
        model.eval()
        with torch.inference_mode():
            sizes = {context.eval_batch_size, 10000 % context.eval_batch_size, 1}
            for size in sorted(sizes - {0}):
                model(torch.zeros(size, 3, 32, 32, device=context.device))
    return state


@torch.no_grad()
def prepare(state, train_data: TrainingData, seed: int) -> None:
    """Reset all learned state and transfer/preprocess every image, inside timing."""
    model = state.model
    # The harness seeds the global RNG before this call. Reset every trainable
    # layer, gradients, and BatchNorm running statistics, including after warmup.
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(layer, nn.BatchNorm2d):
            layer.reset_parameters()
        elif isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)
    model.zero_grad(set_to_none=True)
    model.train()
    model.mean.zero_()
    model.std.fill_(1)
    state.rng = torch.Generator(device=state.context.device).manual_seed(seed)

    # Decay convolution/head weights, but not BatchNorm affine terms or biases.
    weights = [p for p in model.parameters() if p.ndim > 1]
    other = [p for p in model.parameters() if p.ndim <= 1]
    state.optimizer = torch.optim.SGD(
        [
            {"params": weights, "weight_decay": state.weight_decay},
            {"params": other, "weight_decay": 0.0},
        ],
        lr=state.lr,
        momentum=state.momentum,
        nesterov=state.momentum > 0,
        foreach=state.context.device.type == "cuda",
    )

    # A fresh float copy guarantees the harness's immutable uint8 inputs remain
    # untouched even in CPU smoke tests. Statistics come only from train_data.
    images = train_data.images.to(device=state.context.device, dtype=torch.float32).div_(255)
    std, mean = torch.std_mean(images, dim=(0, 2, 3), correction=0, keepdim=True)
    model.mean.copy_(mean)
    model.std.copy_(std.clamp_min(1e-6))
    images.sub_(model.mean).div_(model.std)
    # NHWC storage allows one gather to perform per-image crop and flip, yielding
    # a channels-last batch. Reflect padding is cached; random crops stay fresh.
    dtype = torch.bfloat16 if state.context.device.type == "cuda" else torch.float32
    state.images = (
        F.pad(images, (4, 4, 4, 4), mode="reflect").to(dtype).permute(0, 2, 3, 1).contiguous()
    )
    state.labels = train_data.labels.to(device=state.context.device, copy=True)
    state.num_images = len(state.labels)
    if state.num_images < 1:
        raise ValueError("Training data must be nonempty")
    steps_per_epoch = math.ceil(state.num_images / state.batch_size)
    total_steps = state.epochs * steps_per_epoch
    warmup_steps = min(state.warmup_epochs * steps_per_epoch, total_steps - 1)
    state.learning_rates = _learning_rates(total_steps, warmup_steps, state.lr, state.schedule)
    state.step = 0


def _augment(state, indices):
    """Independent integer crop, horizontal flip, and mean-filled 8px cutout."""
    device = state.context.device
    shape = (len(indices), 1, 1)
    top = torch.randint(9, shape, device=device, generator=state.rng)
    left = torch.randint(9, shape, device=device, generator=state.rng)
    flip = torch.rand(shape, device=device, generator=state.rng) < 0.5
    rows = state.rows + top
    cols = torch.where(flip, 31 - state.cols, state.cols) + left
    batch = state.images[indices[:, None, None], rows, cols]
    if state.cutout:
        cy = torch.randint(32, shape, device=device, generator=state.rng)
        cx = torch.randint(32, shape, device=device, generator=state.rng)
        lower = state.cutout // 2
        upper = state.cutout - lower
        erased = (
            (state.rows >= cy - lower)
            & (state.rows < cy + upper)
            & (state.cols >= cx - lower)
            & (state.cols < cx + upper)
        )
        # Zero in normalized space equals the current trial's training mean.
        batch.mul_((~erased).unsqueeze(-1))
    return batch.permute(0, 3, 1, 2)


def train(state) -> nn.Module:
    """All fitting is timed; return a plain, single-view, read-only classifier."""
    model = state.model
    model.train()
    for _ in range(state.epochs):
        order = torch.randperm(state.num_images, device=state.context.device, generator=state.rng)
        # Include the remainder: every image is used in each epoch.
        for indices in order.split(state.batch_size):
            for group in state.optimizer.param_groups:
                group["lr"] = state.learning_rates[state.step]
            state.optimizer.zero_grad(set_to_none=True)
            logits = model(_augment(state, indices), normalized=True)
            loss = F.cross_entropy(logits, state.labels[indices], label_smoothing=state.smoothing)
            loss.backward()
            state.optimizer.step()
            state.step += 1
    state.optimizer.zero_grad(set_to_none=True)
    model.eval()
    return model
