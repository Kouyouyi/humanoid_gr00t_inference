"""RTC training primitives: clean committed prefix, loss only on future actions."""

import torch


def condition_training_prefix(actions, noisy_actions, action_mask, max_delay_steps):
    if not isinstance(max_delay_steps, int) or max_delay_steps < 0:
        raise ValueError("rtc_max_delay_steps must be a nonnegative integer")
    batch, horizon, _ = actions.shape
    if max_delay_steps >= horizon:
        raise ValueError("RTC delay must be smaller than the action horizon")
    if max_delay_steps == 0:
        return (
            noisy_actions,
            action_mask,
            torch.zeros(batch, dtype=torch.long, device=actions.device),
        )
    valid = action_mask.bool().any(dim=-1)
    # Padding must be a suffix: prefix conditioning cannot skip interior holes.
    if torch.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("RTC requires contiguous valid action steps")
    lengths = valid.sum(dim=1)
    if torch.any(lengths < 1):
        raise ValueError("RTC sample has no supervised action steps")
    limits = torch.clamp(lengths - 1, max=max_delay_steps)
    # Inclusive uniform sample per example, retaining >=1 supervised future step.
    delays = torch.floor(torch.rand(batch, device=actions.device) * (limits + 1)).long()
    prefix = torch.arange(horizon, device=actions.device)[None, :] < delays[:, None]
    conditioned = torch.where(prefix[:, :, None], actions, noisy_actions)
    loss_mask = action_mask * (~prefix[:, :, None]).to(action_mask.dtype)
    return conditioned, loss_mask, delays


def validate_rtc_options(options, input_horizon, output_horizon, trained_max_delay):
    required = ("action_horizon", "rtc_overlap_steps", "rtc_frozen_steps")
    for key in required:
        if type(options.get(key)) is not int:
            raise ValueError(f"{key} must be an integer")
    horizon, overlap, delay = (options[key] for key in required)
    if not (0 <= delay <= overlap <= horizon <= input_horizon and overlap <= output_horizon):
        raise ValueError(
            "RTC requires 0 <= frozen <= overlap <= input horizon and overlap <= output horizon"
        )
    if delay >= output_horizon:
        raise ValueError("RTC must leave at least one future action")
    if delay > trained_max_delay:
        raise ValueError(f"RTC delay {delay} exceeds trained maximum {trained_max_delay}")
    rate = options.get("rtc_ramp_rate", 4.0)
    if not isinstance(rate, (int, float)) or not 0 < rate < float("inf"):
        raise ValueError("rtc_ramp_rate must be positive and finite")
