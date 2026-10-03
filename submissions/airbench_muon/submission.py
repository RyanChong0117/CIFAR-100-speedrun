"""Airbench-style CIFAR-100 recipe with Muon, ported from Keller Jordan's airbench94_muon.py.

Changes from the CIFAR-10 original (https://github.com/KellerJordan/cifar10-airbench):
- 100-way head; widths/epochs/lrs are JSON parameters because CIFAR-10 values don't transfer.
- No test-time augmentation (prohibited here): plain single-view inference.
- Model takes float32 [0, 1] inputs and normalises/casts internally; logits are float32.
- Global (adaptive) max pooling before the head, so input resolution can vary later.
- torch.compile + synthetic warmup of every graph (train, eval, Newton-Schulz) in build(),
  which is untimed; prepare() resets weights, BatchNorm stats and optimizers and fits the
  whitening layer on real data (timed).
"""

from math import ceil
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from benchmark.api import BuildContext, TrainingData

DEFAULTS = {
    "epochs": 7,  # 40 seeds (SXM4): 75.68% ± 0.28, 7.94 s; + max pool/autotune: 7.74 s
    "batch_size": 2000,
    "widths": [128, 512, 512],  # airbench94 used [64, 256, 256] for CIFAR-10
    # Per-group (or one for all): "conv_pool" (airbench), "pool_conv", or "stride".
    "downsample": "conv_pool",
    "depth": 3,  # convs per group (int, or list per group); 3 adds a residual third conv
    "optimizer": "muon",  # optimizer for conv filters: "muon" or "mars" (MARS-AdamW)
    "muon_lr": 0.24,
    "muon_momentum": 0.6,
    # MARS-AdamW (Yuan et al. 2024, github.com/AGI-Arena/MARS); defaults from the reference.
    "mars_lr": 3e-3,
    "mars_betas": [0.95, 0.99],
    "mars_gamma": 0.025,  # 0 = plain AdamW (with unit-norm gradient clipping)
    "mars_weight_decay": 0.0,
    "bias_lr": 0.053,
    "head_lr": 3.0,  # airbench used 0.67 for 10 classes; 100 classes want ~3-6x (2-4 plateau)
    "sgd_momentum": 0.85,
    "weight_decay": 2e-6,  # multiplied by batch_size, as in airbench
    "label_smoothing": 0.3,  # 0.2 was best before head_lr 3.0; 0.3 now (+~0.2 pt)
    "translate": 1,  # ±1 px; ±2 (airbench) is too strong for a 7-epoch run
    "translate_off_epochs": 0,  # final epochs trained without translation (augmentation annealing)
    "flip": True,
    "whiten_bias_epochs": 3,
    # LR shape for conv filters / head / BN biases: linear warmup, hold at peak, then linear decay
    # to 0, as fractions of total steps. 0 / 0 is airbench's plain linear decay.
    "lr_warmup_frac": 0.05,  # helps only with the large head_lr; with ls 0.3 allows 7 epochs
    "lr_hold_frac": 0.4,  # hold-then-decay: +0.9 pt vs plain decay; 0.4 best at 6-7 epochs
    # Weight EMA over the final fraction of training (0 = off), copied into the model at the
    # end; then BatchNorm stats are refreshed with this many no-grad training batches.
    "ema_frac": 0.0,
    "ema_decay": 0.9,
    "ema_bn_batches": 5,
    # Hard-example selection: from select_start_frac of the epochs onward, each epoch trains
    # only on the select_keep fraction of images with the highest most-recent training loss
    # (recorded for free during training). Batch size is unchanged, so epochs just have fewer
    # steps. select_mode "random" keeps the same count at random (ablation). 1.0 = off.
    "select_keep": 1.0,
    "select_start_frac": 0.5,
    "select_mode": "hard",
    "bn_momentum": 0.6,
    "global_pool": "max",  # "max": same as "adaptive" (AdaptiveMaxPool2d), faster backward
    "compile": True,
    # max-autotune: -0.9% train time on the same GPU; build ~190 s (untimed, limit 600 s).
    "compile_mode": "max-autotune",
}

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


#############################################
#               Muon optimizer              #
#############################################


def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """Approximately orthogonalise G with a quintic Newton-Schulz iteration (see airbench)."""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, nesterov=True, zeropower=zeropower_via_newtonschulz5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov))
        self.zeropower = zeropower

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                p.mul_(len(p) ** 0.5 / p.norm())  # normalise the weight
                update = self.zeropower(g.reshape(len(g), -1)).view(g.shape)
                p.add_(update, alpha=-lr)


class MARSAdamW(torch.optim.Optimizer):
    """MARS-AdamW, approximate variant: variance-reduced gradient from the previous step's
    gradient, clipped to unit norm, then a bias-corrected AdamW step.

    Follows the reference update_fn (mars_type="mars-adamw"), with three changes for this
    recipe: applied to 4-D conv filters (the reference only takes this path for 2-D
    weights, so its CNNs actually run plain AdamW); fp32 master weights and state because
    the model's parameters are fp16; and branch-free clipping (no host sync per tensor).
    """

    def __init__(self, params, lr, betas, gamma, weight_decay, eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=tuple(betas), gamma=gamma,
                                      weight_decay=weight_decay, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, (beta1, beta2), gamma = group["lr"], group["betas"], group["gamma"]
            wd, eps = group["weight_decay"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["master"] = p.detach().float().clone()
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["last_grad"] = torch.zeros_like(grad)
                state["step"] += 1
                step = state["step"]
                master, exp_avg, exp_avg_sq = state["master"], state["exp_avg"], state["exp_avg_sq"]
                c = (grad - state["last_grad"]).mul_(gamma * beta1 / (1 - beta1)).add_(grad)
                c.div_(torch.linalg.vector_norm(c).clamp_(min=1.0))
                exp_avg.mul_(beta1).add_(c, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(c, c, value=1 - beta2)
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                denom = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(eps)
                denom.mul_(bias_correction1)
                master.add_(master * wd + exp_avg / denom, alpha=-lr)
                p.copy_(master)
                state["last_grad"] = grad


#############################################
#            Network Definition             #
#############################################


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, stride=1):
        padding = "same" if stride == 1 else 1
        super().__init__(in_channels, out_channels, kernel_size=3, stride=stride,
                         padding=padding, bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])


class ConvGroup(nn.Module):
    """conv -> pool -> BN -> GELU, then depth-1 more conv/BN/GELU layers.

    depth=2 is airbench94. depth=3 adds a third conv with a residual connection around
    the last two, as in airbench96.
    """

    def __init__(self, channels_in, channels_out, bn_momentum, depth=2, downsample="conv_pool"):
        super().__init__()
        # conv_pool: conv at full res, then max-pool (airbench). pool_conv: max-pool first, so
        # the channel-expanding conv runs on 4x fewer pixels. stride: stride-2 conv, no pool.
        assert downsample in ("conv_pool", "pool_conv", "stride")
        self.downsample = downsample
        self.conv1 = Conv(channels_in, channels_out, stride=2 if downsample == "stride" else 1)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, bn_momentum)
        self.convs = nn.ModuleList(Conv(channels_out, channels_out) for _ in range(depth - 1))
        self.norms = nn.ModuleList(BatchNorm(channels_out, bn_momentum) for _ in range(depth - 1))
        self.residual = depth >= 3
        self.activ = nn.GELU()

    def forward(self, x):
        if self.downsample == "conv_pool":
            x = self.pool(self.conv1(x))
        elif self.downsample == "pool_conv":
            x = self.conv1(self.pool(x))
        else:
            x = self.conv1(x)
        x = self.activ(self.norm1(x))
        x0 = x
        for conv, norm in zip(self.convs, self.norms, strict=True):
            x = self.activ(norm(conv(x)))
        return x + x0 if self.residual else x


class GlobalMaxPool(nn.Module):
    """Global max over H, W. Same forward and gradient as AdaptiveMaxPool2d(1) + flatten, but
    its backward avoids that op's slow atomic-scatter kernel (-1.2% train time). (x.amax was
    tried too: it splits gradients between ties, and training diverged.)"""

    def forward(self, x):
        return x.flatten(2).max(dim=2).values


class CifarNet(nn.Module):
    def __init__(self, widths, num_classes, bn_momentum, dtype, depth=2, downsample="conv_pool",
                 global_pool="adaptive"):
        super().__init__()
        depths = depth if isinstance(depth, list) else [depth] * 3
        downs = downsample if isinstance(downsample, list) else [downsample] * 3
        self.dtype = dtype
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, widths[0], bn_momentum, depths[0], downs[0]),
            ConvGroup(widths[0], widths[1], bn_momentum, depths[1], downs[1]),
            ConvGroup(widths[1], widths[2], bn_momentum, depths[2], downs[2]),
            GlobalMaxPool() if global_pool == "max" else nn.AdaptiveMaxPool2d(1),
        )
        self.head = nn.Linear(widths[2], num_classes, bias=False)
        self.register_buffer("mean", torch.tensor(CIFAR100_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR100_STD).view(1, 3, 1, 1))
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
            elif mod is not self:
                mod.to(dtype)

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        w = self.head.weight.data
        w *= 1 / w.std()

    @torch.no_grad()
    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3)
        patches_flat = patches.reshape(-1, c * h * w).float()
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
        eigenvectors_scaled = eigenvectors.T.reshape(-1, c, h, w) / torch.sqrt(
            eigenvalues.view(-1, 1, 1, 1) + eps
        )
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

    def normalize(self, x):
        """float [0, 1] NCHW -> normalised, model dtype, channels_last."""
        x = (x - self.mean) / self.std
        return x.to(self.dtype).contiguous(memory_format=torch.channels_last)

    def forward(self, x, whiten_bias_grad=True, preprocessed=False):
        # The harness passes raw float32 [0, 1] images; training passes preprocessed ones.
        if not preprocessed:
            x = self.normalize(x)
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1)
        return (self.head(x) / x.size(-1)).float()


#############################################
#                Data / aug                 #
#############################################


def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device)
    images_out = torch.empty(
        (len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype
    )
    if r <= 2:
        for sy in range(-r, r + 1):
            for sx in range(-r, r + 1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[
                    mask, :, r + sy : r + sy + crop_size, r + sx : r + sx + crop_size
                ]
    else:
        images_tmp = torch.empty(
            (len(images), 3, crop_size, crop_size + 2 * r), device=images.device, dtype=images.dtype
        )
        for s in range(-r, r + 1):
            mask = shifts[:, 0] == s
            images_tmp[mask] = images[mask, :, r + s : r + s + crop_size, :]
        for s in range(-r, r + 1):
            mask = shifts[:, 1] == s
            images_out[mask] = images_tmp[mask, :, :, r + s : r + s + crop_size]
    return images_out


def epoch_images(state, epoch):
    """Airbench loader: pre-flipped, pre-padded images; fresh crops each epoch, and all
    images flipped together every other epoch (more diverse than independent flips)."""
    cfg = state.cfg
    translate = cfg["translate"] > 0 and epoch < cfg["epochs"] - cfg["translate_off_epochs"]
    if translate:
        images = batch_crop(state.padded_images, state.images.shape[-2])
    else:
        # Match batch_crop's (contiguous NCHW) layout so torch.compile doesn't recompile.
        images = state.images.contiguous()
    if cfg["flip"] and epoch % 2 == 1:
        images = images.flip(-1)
    return images


#############################################
#            Harness entry points           #
#############################################


def make_optimizers(state):
    cfg, model = state.cfg, state.model
    wd = cfg["weight_decay"] * cfg["batch_size"]
    bias_lr, head_lr = cfg["bias_lr"], cfg["head_lr"]
    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]
    fused = state.device.type == "cuda"
    sgd = torch.optim.SGD(param_configs, momentum=cfg["sgd_momentum"], nesterov=True, fused=fused)
    if cfg["optimizer"] == "mars":
        filter_opt = MARSAdamW(filter_params, lr=cfg["mars_lr"], betas=cfg["mars_betas"],
                               gamma=cfg["mars_gamma"], weight_decay=cfg["mars_weight_decay"])
    else:
        filter_opt = Muon(filter_params, lr=cfg["muon_lr"], momentum=cfg["muon_momentum"],
                          zeropower=state.zeropower)
    for opt in (sgd, filter_opt):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return sgd, filter_opt


def build(context: BuildContext):
    cfg = {**DEFAULTS, **context.parameters}
    device = context.device
    cuda = device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    dtype = torch.float16 if cuda else torch.float32
    model = CifarNet(
        cfg["widths"], context.num_classes, cfg["bn_momentum"], dtype, cfg["depth"],
        cfg["downsample"], cfg["global_pool"],
    )
    model = model.to(device, memory_format=torch.channels_last)
    zeropower = zeropower_via_newtonschulz5
    if cuda and cfg["compile"]:
        model.compile(mode=cfg["compile_mode"])
        zeropower = torch.compile(zeropower_via_newtonschulz5)
    state = SimpleNamespace(
        model=model, context=context, cfg=cfg, device=device, dtype=dtype,
        zeropower=zeropower, epoch_callback=None,
    )
    if cuda:
        warmup(state)
    return state


def warmup(state):
    """Compile every graph on synthetic data (untimed). prepare() resets all of it.

    Runs the real prepare/train code so tensor dtypes, strides and flags match exactly;
    any mismatch (e.g. memory layout) would make torch.compile recompile inside the timed
    trial. The short schedule crosses the whitening-bias cutoff so both graphs compile.
    """
    cfg, model, device = state.cfg, state.model, state.device
    bs = cfg["batch_size"]
    generator = torch.Generator().manual_seed(0)
    synthetic = TrainingData(
        torch.randint(0, 256, (2 * bs, 3, 32, 32), dtype=torch.uint8, generator=generator),
        torch.randint(0, state.context.num_classes, (2 * bs,), generator=generator),
    )
    state.cfg = {**cfg, "epochs": cfg["whiten_bias_epochs"] + 1}
    prepare(state, synthetic, 0)
    train(state)
    state.cfg = cfg
    # Evaluation graphs: full batches, the final 10000 % 1024 batch, and tiny batches.
    model.eval()
    eval_bs = state.context.eval_batch_size
    with torch.inference_mode():
        for n in (eval_bs, 10_000 % eval_bs or eval_bs, 1):
            model(torch.rand(n, 3, 32, 32, device=device))
    torch.cuda.synchronize(device)


def prepare(state, data: TrainingData, seed: int) -> None:
    cfg, model, device = state.cfg, state.model, state.device
    model.reset()  # weights and BatchNorm running stats
    model.train()
    state.sgd, state.filter_opt = make_optimizers(state)

    images = data.images.to(device, non_blocking=True).float().div_(255)
    images = model.normalize(images)
    state.labels = data.labels.to(device, non_blocking=True)
    model.init_whiten(images[:5000])
    if cfg["flip"]:
        images = batch_flip_lr(images)
    state.images = images
    pad = cfg["translate"]
    if pad > 0:
        state.padded_images = F.pad(images, (pad,) * 4, "reflect")


def lr_factor(cfg, step, total_steps):
    warm = cfg["lr_warmup_frac"] * total_steps
    hold_end = warm + cfg["lr_hold_frac"] * total_steps
    if step < warm:
        return (step + 1) / warm
    if step < hold_end:
        return 1.0
    return (total_steps - step) / (total_steps - hold_end)


def train(state) -> nn.Module:
    cfg, model = state.cfg, state.model
    sgd, filter_opt = state.sgd, state.filter_opt
    n, bs = len(state.labels), cfg["batch_size"]
    steps_per_epoch = n // bs
    selecting = cfg["select_keep"] < 1.0
    keep_steps = max(1, int(cfg["select_keep"] * n) // bs)
    select_start = cfg["select_start_frac"] * cfg["epochs"]
    # Steps per epoch, planned up front so the LR schedule spans the real step count.
    plan = []
    for e in range(ceil(cfg["epochs"])):
        epoch_steps = keep_steps if selecting and e >= select_start else steps_per_epoch
        if e + 1 > cfg["epochs"]:  # fractional final epoch
            epoch_steps = ceil((cfg["epochs"] - e) * epoch_steps)
        plan.append(epoch_steps)
    total_steps = sum(plan)
    sample_loss = torch.full((n,), float("inf"), device=state.labels.device) if selecting else None
    whiten_bias_steps = ceil(cfg["whiten_bias_epochs"] * steps_per_epoch)
    ema_start = total_steps - ceil(cfg["ema_frac"] * total_steps) if cfg["ema_frac"] else None
    params = [p for p in model.parameters() if p.requires_grad]
    ema = None

    step = 0
    for epoch, epoch_steps in enumerate(plan):
        model.train()
        images = epoch_images(state, epoch)
        if selecting and epoch >= select_start:
            k = keep_steps * bs
            if cfg["select_mode"] == "hard":
                pool = torch.topk(sample_loss, k, sorted=False).indices
            else:
                pool = torch.randperm(n, device=images.device)[:k]
            indices = pool[torch.randperm(k, device=images.device)]
        else:
            indices = torch.randperm(n, device=images.device)
        for i in range(epoch_steps):
            idxs = indices[i * bs : (i + 1) * bs]
            outputs = model(images[idxs], step < whiten_bias_steps, True)
            if sample_loss is None:
                F.cross_entropy(outputs, state.labels[idxs],
                                label_smoothing=cfg["label_smoothing"], reduction="sum").backward()
            else:
                loss = F.cross_entropy(outputs, state.labels[idxs],
                                       label_smoothing=cfg["label_smoothing"], reduction="none")
                sample_loss[idxs] = loss.detach().float()
                loss.sum().backward()
            for group in sgd.param_groups[:1]:
                group["lr"] = group["initial_lr"] * max(0.0, 1 - step / whiten_bias_steps)
            for group in sgd.param_groups[1:] + filter_opt.param_groups:
                group["lr"] = group["initial_lr"] * lr_factor(cfg, step, total_steps)
            sgd.step()
            filter_opt.step()
            model.zero_grad(set_to_none=True)
            step += 1
            if ema_start is not None and step >= ema_start:
                with torch.no_grad():
                    current = [p.float() for p in params]
                    if ema is None:
                        ema = current
                    else:
                        torch._foreach_lerp_(ema, current, 1 - cfg["ema_decay"])
        # Dev-only hook (dev/curve.py); always None under the harness.
        if state.epoch_callback is not None:
            state.epoch_callback(epoch + 1, step)
    if ema is not None:
        with torch.no_grad():
            for p, e in zip(params, ema, strict=True):
                p.copy_(e)
            # Averaged weights don't match the running BN stats; refresh them.
            model.train()
            indices = torch.randperm(n, device=images.device)
            for i in range(min(cfg["ema_bn_batches"], steps_per_epoch)):
                idxs = indices[i * bs : (i + 1) * bs]
                model(images[idxs], False, True)
    state.steps = step
    return model
