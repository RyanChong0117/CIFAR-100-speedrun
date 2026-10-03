"""Isolated variants installed only in generated experiment submissions.

The production recipe is copied verbatim before install() is appended. No
production defaults or harness code are changed. All real-data work stays in
prepare(); build only compiles and warms synthetic inputs.
"""

import torch
import torch.nn.functional as F

from benchmark.api import TrainingData


def zeropower_batched(gradients, steps=3, eps=1e-7):
    """Independent three-iteration Newton-Schulz updates for equal matrix shapes."""
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


def normalize_uint8(raw, mean, std, dtype):
    """Fuse deterministic preprocessing; keep random draws in the original code."""
    images = raw.float().div(255)
    images = (images - mean) / std
    return images.to(dtype).contiguous(memory_format=torch.channels_last)


def install(namespace, variant):
    """Install one change into a private, frozen submission module."""
    if variant not in ("batched_muon", "fused_prepare"):
        raise ValueError(f"Unknown compression variant: {variant}")
    original_build = namespace["build"]
    batched_function = zeropower_batched
    normalize_function = normalize_uint8

    if variant == "batched_muon":
        def make_muon(*args, **kwargs):
            return BatchedMuon(*args, **kwargs, batched_zeropower=batched_function)

        namespace["Muon"] = make_muon
    else:
        def prepare(state, data, seed):
            cfg, model, device = state.cfg, state.model, state.device
            model.reset()
            model.train()
            state.sgd, state.muon = namespace["make_optimizers"](state)
            raw = data.images.to(device, non_blocking=True)
            # Base warmup has 4,000 images. Compile only the production shape,
            # then explicitly warm it below before returning build().
            function = normalize_function if len(raw) == 50_000 else normalize_uint8
            images = function(raw, model.mean, model.std, state.dtype)
            state.labels = data.labels.to(device, non_blocking=True)
            model.init_whiten(images[:5000])
            if cfg["flip"]:
                images = namespace["batch_flip_lr"](images)
            state.images = images
            pad = cfg["translate"]
            if pad > 0:
                state.padded_images = F.pad(images, (pad,) * 4, "reflect")

        namespace["prepare"] = prepare

    def build(context):
        nonlocal batched_function, normalize_function
        compiled = context.device.type == "cuda" and context.parameters.get("compile", True)
        if compiled:
            if variant == "batched_muon":
                batched_function = torch.compile(zeropower_batched)
            else:
                normalize_function = torch.compile(normalize_uint8, fullgraph=True, dynamic=False)
        state = original_build(context)
        if variant == "fused_prepare" and context.device.type == "cuda":
            generator = torch.Generator().manual_seed(0)
            synthetic = TrainingData(
                torch.randint(0, 256, (50_000, 3, 32, 32), dtype=torch.uint8,
                              generator=generator),
                torch.randint(0, context.num_classes, (50_000,), generator=generator),
            )
            namespace["prepare"](state, synthetic, 0)
            torch.cuda.synchronize(context.device)
        return state

    namespace["build"] = build
