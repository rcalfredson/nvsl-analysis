# src/utils/debug_fly_groups.py

import logging

logger = logging.getLogger("fly_group_debug")
handler = logging.FileHandler("debug_fly_groups.log")
formatter = logging.Formatter("%(asctime)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def log_fly_group(group_name: str, indices, vas):
    """
    Logs each fly in a given group.
    Each index refers to a row in vas (one fly).
    """
    if indices is None:
        logger.info(f"[{group_name}] No flies")
        return

    logger.info(f"[{group_name}] {len(indices)} flies")
    for idx in indices:
        try:
            va = vas[idx]
            logger.info(f"  idx={idx}, video='{va.fn}', fly={va.f}")
        except Exception as e:
            logger.info(f"  idx={idx} - ERROR retrieving fly info: {e}")
