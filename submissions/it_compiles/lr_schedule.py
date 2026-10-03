"""Per-update LR schedules; the trial-specific lists are created in prepare."""

import math

SCHEDULE_DEFAULTS = {
    "lr_schedule": "cosine",
    "one_cycle_div_factor": 25.0,
    "one_cycle_final_div_factor": 10000.0,
}


def resolve_schedule(parameters):
    options = SCHEDULE_DEFAULTS | {
        key: parameters[key] for key in SCHEDULE_DEFAULTS if key in parameters
    }
    if options["lr_schedule"] not in ("cosine", "one_cycle"):
        raise ValueError("lr_schedule must be cosine or one_cycle")
    for key in ("one_cycle_div_factor", "one_cycle_final_div_factor"):
        value = options[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 1:
            raise ValueError(f"{key} must be finite and >= 1")
    return options


def learning_rates(total_steps, warmup_steps, peak, options=None):
    """Cosine preserves the old recipe; one-cycle matches PyTorch's two-phase LR.

    One-cycle uses cosine interpolation on the rise and fall, with constant SGD
    momentum. Warmup specifies the number of rising updates, not a fixed fraction
    of a longer training run. Zero warmup starts at peak LR; a single rising
    update also uses the peak, avoiding the reference scheduler's zero divisor.
    """
    options = resolve_schedule(options or {})
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("total_steps must be a positive integer")
    if type(warmup_steps) is not int or not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError("peak must be finite and positive")
    rates = []
    for step in range(total_steps):
        if options["lr_schedule"] == "cosine":
            if step < warmup_steps:
                scale = 0.1 + 0.9 * (step + 1) / warmup_steps
            else:
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
                scale = 0.001 + 0.999 * (1 + math.cos(math.pi * progress)) / 2
            rate = peak * scale
        else:
            initial = peak / options["one_cycle_div_factor"]
            minimum = initial / options["one_cycle_final_div_factor"]
            if warmup_steps > 1 and step < warmup_steps:
                progress = step / (warmup_steps - 1)
                rate = peak + (initial - peak) * (1 + math.cos(math.pi * progress)) / 2
            else:
                start = max(0, warmup_steps - 1)
                progress = (step - start) / max(1, total_steps - 1 - start)
                rate = minimum + (peak - minimum) * (1 + math.cos(math.pi * progress)) / 2
        rates.append(rate)
    return rates
