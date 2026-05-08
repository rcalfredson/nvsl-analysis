from __future__ import annotations

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
