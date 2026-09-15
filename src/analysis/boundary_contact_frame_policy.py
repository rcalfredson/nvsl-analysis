"""Missing-frame policy helpers for boundary-contact event traces."""

import numpy as np

from src.utils.util import trueRegions


def interpolation_classifiable_contact(contact, lost):
    """Retain bracketed interpolation while marking terminal lost spans unknown."""
    contact = np.asarray(contact, dtype=float).copy()
    lost = np.asarray(lost, dtype=bool)
    if contact.shape != lost.shape:
        raise ValueError("contact and lost masks must have matching shapes")

    lost_regions = trueRegions(lost)
    if lost_regions and lost_regions[0].start == 0:
        contact[lost_regions[0]] = np.nan
    if lost_regions and lost_regions[-1].stop == lost.size:
        contact[lost_regions[-1]] = np.nan
    return contact


def complete_contact_regions(contact):
    """Return contact regions bounded on both sides by known non-contact frames."""
    contact = np.asarray(contact, dtype=float)
    regions = trueRegions(contact == 1)
    complete = []
    for region in regions:
        if region.start == 0 or region.stop == contact.size:
            continue
        if not np.isfinite(contact[region.start - 1]):
            continue
        if not np.isfinite(contact[region.stop]):
            continue
        complete.append(region)
    return complete
