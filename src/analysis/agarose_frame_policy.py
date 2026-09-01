"""Shared frame-policy handling for percent-time-on-agarose metrics."""

import numpy as np


AGAROSE_TIME_LOST_FRAME_POLICIES = (
    "corrected",
    "legacy",
    "interpolated-inclusive",
)


def agarose_percentage_masks(
    contact, interpolated_contact, lost, policy="interpolated-inclusive"
):
    """Return boolean numerator and denominator masks for an agarose percentage."""
    if policy not in AGAROSE_TIME_LOST_FRAME_POLICIES:
        raise ValueError(
            "agarose_time_lost_frame_policy must be 'corrected', 'legacy', or "
            f"'interpolated-inclusive', got {policy!r}"
        )

    # Contact arrays from circular-well detection are floats so they can carry
    # NaN at lost frames.  Equality avoids treating NaN as truthy when casting.
    contact = np.asarray(contact) == 1
    interpolated_contact = np.asarray(interpolated_contact) == 1
    lost = np.asarray(lost, dtype=bool)
    if contact.shape != lost.shape or interpolated_contact.shape != lost.shape:
        raise ValueError("Agarose contact and lost-frame masks must have matching shapes")

    valid = ~lost
    if policy == "corrected":
        return contact & valid, valid
    if policy == "legacy":
        return interpolated_contact, valid
    return interpolated_contact, np.ones_like(valid, dtype=bool)
