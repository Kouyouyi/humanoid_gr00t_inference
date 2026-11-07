"""Optional SONIC output grid. Applied only after action denormalization."""

import numpy as np


def process_sonic_tokens(actions, mode="continuous"):
    if mode not in ("continuous", "grid"):
        raise ValueError("SONIC token mode must be continuous or grid")
    if mode == "continuous":
        return actions
    if "motion_token" not in actions:
        raise ValueError("SONIC grid mode requires motion_token actions")
    tokens = actions["motion_token"]
    if not np.isfinite(tokens).all():
        raise ValueError("SONIC tokens must be finite")
    # Exact binary grid; nearest with ties-to-even. No additional clipping.
    return {**actions, "motion_token": (np.rint(tokens * 16) / 16).astype(tokens.dtype)}
