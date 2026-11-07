"""Normalize an absolute, control-clock aligned RTC prefix for the action head."""

import numpy as np
import torch


def deployment_rtc_options(options, delay_steps):
    """Apply a pinned deployment delay only when the caller sends RTC inputs.

    Bootstrap/ordinary inference has no committed prefix. Never synthesize one
    from previous server output: only the client knows the executed control ticks.
    """
    if delay_steps is None or not options or not {"rtc_prefix", "rtc_delay_steps"} & options.keys():
        return options
    options = dict(options)
    requested = options.setdefault("rtc_delay_steps", delay_steps)
    if type(requested) is not int or requested != delay_steps:
        raise ValueError(
            f"RTC request delay {requested!r} does not match server delay {delay_steps}; "
            "configure the client from get_rtc_config()"
        )
    return options


def prepare_rtc_prefix(policy, states, options):
    delay = options.get("rtc_delay_steps")
    limit = policy.model.config.rtc_max_delay_steps
    horizon = policy.model.config.action_horizon
    if type(delay) is not int or not 0 <= delay <= limit or delay >= horizon:
        raise ValueError(
            f"rtc_delay_steps must be an integer in [0, {limit}] below horizon {horizon}"
        )
    prefix = options.get("rtc_prefix", {})
    keys = policy.modality_configs["action"].modality_keys
    if delay == 0:
        if prefix:
            raise ValueError("A zero-delay request must not include a prefix")
        return None, None, None
    if set(prefix) != set(keys):
        raise ValueError(f"rtc_prefix must contain exactly {keys}")
    configs = policy.modality_configs["action"].action_configs
    if configs is not None and any(c.rep.name != "ABSOLUTE" for c in configs):
        raise ValueError("RTC physical prefixes currently require absolute actions")
    batch = len(states)
    arrays = {}
    for key in keys:
        value = np.asarray(prefix[key], dtype=np.float32)
        params = policy.processor.state_action_processor.norm_params[policy.embodiment_tag.value][
            "action"
        ][key]
        dim = len(params["min"])
        if value.shape != (batch, delay, dim) or not np.isfinite(value).all():
            raise ValueError(f"Invalid RTC prefix {key}; expected {(batch, delay, dim)}")
        arrays[key] = value
    normalized = []
    for b in range(batch):
        values = policy.processor.state_action_processor.apply_action(
            {k: v[b] for k, v in arrays.items()}, policy.embodiment_tag.value, states[b]
        )
        normalized.append(np.concatenate([values[k] for k in keys], axis=-1))
    x = np.stack(normalized)
    packed = np.zeros((batch, horizon, policy.model.config.max_action_dim), dtype=np.float32)
    packed[:, :delay, : x.shape[-1]] = x
    model_options = dict(
        action_horizon=delay, rtc_overlap_steps=delay, rtc_frozen_steps=delay, rtc_ramp_rate=4.0
    )
    return torch.from_numpy(packed), model_options, arrays
