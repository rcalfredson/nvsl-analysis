from __future__ import annotations

import argparse
import os

LOCAL_ANALYZE_CONFIG_FILE = ".analyze.local.env"


def parse_local_bool(value: str, *, key_name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(
        f"{LOCAL_ANALYZE_CONFIG_FILE}: {key_name} must be one of "
        "'1,true,yes,on,0,false,no,off'"
    )


def load_local_analyze_config(path=LOCAL_ANALYZE_CONFIG_FILE) -> dict[str, str]:
    cfg = {}
    if not os.path.exists(path):
        return cfg

    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(
                    f"{path}:{line_no}: expected KEY=VALUE, got {raw_line.rstrip()!r}"
                )
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                raise ValueError(f"{path}:{line_no}: empty key is not allowed")
            cfg[key] = value
    return cfg


def add_tracking_arguments(parser, config=None):
    """Local machine defaults are argparse defaults, so explicit flags win."""
    cfg = load_local_analyze_config() if config is None else config
    parser.add_argument(
        "--tracking-variant",
        default=cfg.get("TRACKING_VARIANT", "original"),
        help="Named trajectory variant; default: original acquisition tracking.",
    )
    parser.add_argument(
        "--tracking-manifest",
        default=cfg.get("TRACKING_MANIFEST"),
        help="Version 1 JSON mapping full recording paths to variant .trx paths.",
    )
    parser.add_argument(
        "--tracking-fallback",
        action=argparse.BooleanOptionalAction,
        default=parse_local_bool(
            cfg.get("TRACKING_FALLBACK", "false"), key_name="TRACKING_FALLBACK"
        ),
        help=(
            "Allow original tracking only when the requested variant has no "
            "manifest entry (default: strict)."
        ),
    )
