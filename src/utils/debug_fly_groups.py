# src/utils/debug_fly_groups.py

import logging

logger = logging.getLogger("fly_group_debug")
logger.setLevel(logging.INFO)

_initialized = False


def init_fly_group_logging(log_path="debug_fly_groups.log"):
    """
    Initialize the file handler for fly group debug logging.
    This must be called explicitly by the main script when logging is desired.
    """
    global _initialized
    if _initialized:
        return

    handler = logging.FileHandler(str(log_path))
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    _initialized = True


def log_fly_group(group_name: str, indices, vas):
    """
    Logs each fly in a given group.
    Each index refers to a row in vas (one fly).
    """
    if not _initialized:
        return  # logging disabled
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
