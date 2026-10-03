"""Isolated, opt-in performance variants for the frozen Airbench recipe."""

import torch

from dev.airbench_reference import Muon


def batch_crop_vectorized(images, crop_size):
    """One uniform integer (dy, dx) per image, exactly as in the reference.

    Advanced indexing copies into contiguous NCHW, matching the reference's
    output strides. Keeping randint outside compilation preserves its RNG use.
    """
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device)
    return crop_with_shifts(images, crop_size, shifts)


def crop_with_shifts(images, crop_size, shifts):
    r = (images.size(-1) - crop_size) // 2
    n = torch.arange(len(images), device=images.device)[:, None, None, None]
    c = torch.arange(images.size(1), device=images.device)[None, :, None, None]
    y = (
        torch.arange(crop_size, device=images.device)[None, None, :, None]
        + shifts[:, 0, None, None, None]
        + r
    )
    x = (
        torch.arange(crop_size, device=images.device)[None, None, None, :]
        + shifts[:, 1, None, None, None]
        + r
    )
    return images[n, c, y, x]


def zeropower_batched(G, steps=3, eps=1e-7):
    """Independent reference iterations; no cross-matrix normalization.

    The leading dimension batches only equal matrix shapes. BF16, coefficient
    order, transpose rule, epsilon, and all three iterations are unchanged.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    transpose = G.size(-2) > G.size(-1)
    if transpose:
        X = X.transpose(-2, -1)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.transpose(-2, -1) if transpose else X


class BatchedMuon(Muon):
    def __init__(
        self,
        params,
        lr,
        momentum,
        nesterov=True,
        zeropower=None,
        batched_zeropower=zeropower_batched,
    ):
        super().__init__(params, lr, momentum, nesterov, zeropower)
        self.batched_zeropower = batched_zeropower
        self.compatible = []
        for group in self.param_groups:
            shapes = {}
            for p in group["params"]:
                shapes.setdefault((len(p), p.numel() // len(p)), []).append(p)
            self.compatible.append((group, list(shapes.values())))

    @torch.no_grad()
    def step(self):
        for group, buckets in self.compatible:
            lr, momentum = group["lr"], group["momentum"]
            for bucket in buckets:
                active, matrices = [], []
                for p in bucket:
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                    p.mul_(len(p) ** 0.5 / p.norm())
                    active.append(p)
                    matrices.append(g.reshape(len(g), -1))
                if not active:
                    continue
                if len(active) == 1:
                    updates = [self.zeropower(matrices[0])]
                else:
                    updates = self.batched_zeropower(torch.stack(matrices)).unbind()
                for p, update in zip(active, updates, strict=True):
                    p.add_(update.reshape(p.shape), alpha=-lr)
