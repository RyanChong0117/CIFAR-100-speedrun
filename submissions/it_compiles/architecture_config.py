"""Data-independent model specifications, shared with development sweep tools."""

ARCHITECTURES = {
    "resnet9": (1, 0, 1),
    "resnet11": (1, 1, 1),
    "resnet15": (2, 1, 2),
}
ARCHITECTURE_KEYS = {
    "architecture",
    "stage_widths",
    "residual_blocks",
    "global_pool",
    "downsampling",
}


def resolve_architecture(parameters):
    """Widths describe stem/stages 1-3; depth counts two-convolution skip blocks."""
    architecture = parameters.get("architecture", "resnet11")
    if not isinstance(architecture, str) or architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {sorted(ARCHITECTURES)}")
    width = parameters.get("width", 64)
    if type(width) is not int or width < 1:
        raise ValueError("width must be a positive integer")
    widths = parameters.get("stage_widths", [width, 2 * width, 4 * width, 8 * width])
    blocks = parameters.get("residual_blocks", ARCHITECTURES[architecture])
    for name, values, size, minimum in (
        ("stage_widths", widths, 4, 1),
        ("residual_blocks", blocks, 3, 0),
    ):
        if (
            not isinstance(values, list | tuple)
            or len(values) != size
            or any(type(value) is not int or value < minimum for value in values)
        ):
            raise ValueError(f"{name} must contain {size} integers >= {minimum}")
    pool = parameters.get("global_pool", "max")
    downsampling = parameters.get("downsampling", "maxpool")
    if pool not in ("max", "avg"):
        raise ValueError("global_pool must be max or avg")
    if downsampling not in ("maxpool", "avgpool", "stride"):
        raise ValueError("downsampling must be maxpool, avgpool, or stride")
    return {
        "architecture": architecture,
        "stage_widths": list(widths),
        "residual_blocks": list(blocks),
        "global_pool": pool,
        "downsampling": downsampling,
    }
