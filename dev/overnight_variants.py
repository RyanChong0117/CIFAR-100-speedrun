"""Opt-in experiment patches for a private frozen RC3 submission.

Append ``from .overnight_variants import install; install(globals())`` to a
frozen submission and copy this file alongside it. Nothing in benchmark/ or
the production recipe is changed. Context parameters select independent changes.
All real-data and seeded initialization work remains in prepare/train.
"""

import torch
from torch import nn

from benchmark.api import TrainingData


def fast_conv_reset(module):
    """Exactly preserve Conv's RNG draws and Dirac values without scalar launches."""
    nn.Conv2d.reset_parameters(module)
    weight = module.weight.data
    # RC3 calls dirac_(weight[:in_channels]) with groups=1. Its remaining output
    # channels retain the random initialization from Conv2d.reset_parameters.
    prefix = weight[:weight.size(1)]
    prefix.zero_()
    prefix.diagonal(dim1=0, dim2=1)[weight.size(2) // 2, weight.size(3) // 2].fill_(1)


def crop_with_shifts(images, shifts, crop_size):
    """Deterministic gather for RC3's original per-image (y, x) shift draws."""
    radius = (images.size(-1) - crop_size) // 2
    samples = torch.arange(len(images), device=images.device)[:, None, None, None]
    channels = torch.arange(images.size(1), device=images.device)[None, :, None, None]
    offsets = torch.arange(crop_size, device=images.device)
    ys = (shifts[:, 0, None] + radius + offsets)[..., None]
    xs = (shifts[:, 1, None] + radius + offsets)[:, None, :]
    return images[samples, channels, ys[:, None], xs[:, None]].contiguous()


def batch_crop_vectorized(images, crop_size, *, kernel=crop_with_shifts):
    radius = (images.size(-1) - crop_size) // 2
    # Keep this outside torch.compile: changing the RNG implementation changes
    # subsequent permutations and therefore the training trajectory.
    shifts = torch.randint(-radius, radius + 1, size=(len(images), 2), device=images.device)
    return kernel(images, shifts, crop_size)


def zeropower_batched(gradients, steps=3, eps=1e-7):
    """Independent NS updates; BF16 batched GEMM rounding can differ from serial."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = gradients.bfloat16()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    transpose = gradients.size(-2) > gradients.size(-1)
    if transpose:
        x = x.transpose(-2, -1)
    for _ in range(steps):
        gram = x @ x.transpose(-2, -1)
        polynomial = b * gram + c * gram @ gram
        x = a * x + polynomial @ x
    return x.transpose(-2, -1) if transpose else x


class BatchedMuon(torch.optim.Optimizer):
    """Group only equal matrix shapes; retain RC3 momentum and weight scaling."""

    def __init__(self, params, lr, momentum, nesterov=True, *, zeropower,
                 batched_zeropower=zeropower_batched):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov))
        self.zeropower = zeropower
        self.batched_zeropower = batched_zeropower
        self.buckets = []
        for group in self.param_groups:
            shapes = {}
            for parameter in group["params"]:
                shape = (len(parameter), parameter.numel() // len(parameter))
                shapes.setdefault(shape, []).append(parameter)
            self.buckets.append((group, list(shapes.values())))

    @torch.no_grad()
    def step(self):
        for group, buckets in self.buckets:
            lr, momentum = group["lr"], group["momentum"]
            for bucket in buckets:
                active, matrices = [], []
                for parameter in bucket:
                    gradient = parameter.grad
                    if gradient is None:
                        continue
                    state = self.state[parameter]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(gradient)
                    buffer = state["momentum_buffer"]
                    buffer.mul_(momentum).add_(gradient)
                    gradient = (
                        gradient.add(buffer, alpha=momentum) if group["nesterov"] else buffer
                    )
                    parameter.mul_(len(parameter) ** 0.5 / parameter.norm())
                    active.append(parameter)
                    matrices.append(gradient.reshape(len(gradient), -1))
                if len(active) == 1:
                    updates = [self.zeropower(matrices[0])]
                elif active:
                    updates = self.batched_zeropower(torch.stack(matrices)).unbind()
                else:
                    continue
                for parameter, update in zip(active, updates, strict=True):
                    parameter.add_(update.reshape(parameter.shape), alpha=-lr)


def install(namespace):
    """Select ns_steps, fast_reset, crop_impl, batched_muon via BuildContext.

    With defaults this delegates to the untouched RC3 implementation. Existing
    batch_size, epoch, width, and learning-rate parameters work without changes.
    The reported step count is ceil(epochs * (50000 // batch_size)).
    """
    original_build = namespace["build"]
    original_ns = namespace["zeropower_via_newtonschulz5"]
    original_muon = namespace["Muon"]
    original_crop = namespace["batch_crop"]
    original_reset = namespace["Conv"].reset_parameters

    def build(context):
        parameters = context.parameters
        ns_steps = parameters.get("ns_steps", 3)
        if type(ns_steps) is not int or ns_steps not in (1, 2, 3):
            raise ValueError("ns_steps must be one of 1, 2, 3")
        crop_impl = parameters.get("crop_impl", "reference")
        if crop_impl not in ("reference", "vectorized", "compiled"):
            raise ValueError("crop_impl must be reference, vectorized, or compiled")
        compile_enabled = context.device.type == "cuda" and parameters.get("compile", True)

        def selected_ns(gradient, steps=ns_steps, eps=1e-7):
            return original_ns(gradient, steps=steps, eps=eps)

        namespace["zeropower_via_newtonschulz5"] = original_ns if ns_steps == 3 else selected_ns
        namespace["Conv"].reset_parameters = (
            fast_conv_reset if parameters.get("fast_reset", False) else original_reset
        )
        namespace["Muon"] = original_muon
        if parameters.get("batched_muon", False):
            def selected_batched(gradients):
                return zeropower_batched(gradients, steps=ns_steps)

            batched_function = (torch.compile(selected_batched) if compile_enabled
                                else selected_batched)

            def make_muon(*args, **kwargs):
                return BatchedMuon(*args, **kwargs, batched_zeropower=batched_function)

            namespace["Muon"] = make_muon

        crop_kernel = crop_with_shifts
        if crop_impl == "compiled" and compile_enabled:
            crop_kernel = torch.compile(crop_with_shifts, fullgraph=True, dynamic=False)

        def selected_crop(images, crop_size):
            # RC3's ordinary build warmup has 2*batch_size images. Keep that
            # synthetic path eager and compile precisely the production shape.
            kernel = crop_kernel if len(images) == 50_000 else crop_with_shifts
            return batch_crop_vectorized(images, crop_size, kernel=kernel)

        namespace["batch_crop"] = original_crop if crop_impl == "reference" else selected_crop
        state = original_build(context)
        if crop_impl == "compiled" and compile_enabled and state.cfg["translate"] > 0:
            # Use real prepare() on synthetic data to preserve the exact source
            # strides after normalization, torch.where flipping and padding.
            # Guessing channels_last here caused timed recompilation in an
            # earlier study. Every subsequent trial calls prepare() afresh.
            generator = torch.Generator().manual_seed(0)
            synthetic = TrainingData(
                torch.randint(0, 256, (50_000, 3, 32, 32), dtype=torch.uint8,
                              generator=generator),
                torch.randint(0, context.num_classes, (50_000,), generator=generator),
            )
            namespace["prepare"](state, synthetic, 0)
            selected_crop(state.padded_images, state.images.shape[-2])
            torch.cuda.synchronize(state.device)
        return state

    namespace["build"] = build
