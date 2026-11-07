"""Optional episode-quality conditioning using existing language vocabulary."""

from numbers import Integral
import random


def validate_quality_score(value):
    """Accept integer scores 1..5, rejecting booleans and uncalibrated values."""
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= 5:
        raise ValueError(f"quality_score must be an integer from 1 to 5, got {value!r}")
    return int(value)


def condition_quality(language, metadata, *, enabled, training, dropout_prob, default_score):
    """Condition the action target, without reweighting or negating its loss.

    Training labels are required even on dropped examples. Evaluation never
    drops the condition and defaults to the checkpoint's requested score.
    Disabled checkpoints retain exactly their original text and RNG behavior.
    """
    if not enabled:
        if "quality_score" in metadata:
            raise ValueError("quality_score supplied to a checkpoint without quality conditioning")
        return language
    if not 0 <= dropout_prob <= 1:
        raise ValueError("quality_dropout_prob must be between 0 and 1")
    validate_quality_score(default_score)
    if training and "quality_score" not in metadata:
        raise ValueError("Quality-conditioned training requires quality_score for every example")
    score = validate_quality_score(metadata.get("quality_score", default_score))
    if training and dropout_prob > 0 and random.random() < dropout_prob:
        return language
    return f"{language}\nEpisode quality: {score}/5."
