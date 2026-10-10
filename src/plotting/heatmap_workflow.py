"""Build or reuse probability-map caches for manuscript heatmap recipes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile


def heatmap_analysis_command(root, command, export_path):
    """Normalize rendering options away while retaining analysis selections."""
    argv = shlex.split(command) if isinstance(command, str) else list(command)
    if len(argv) < 2 or Path(argv[1]).name != "analyze.py" or "--pltHm" not in argv:
        raise ValueError("Expected a python analyze.py --pltHm recipe")
    argv[:2] = [sys.executable, str(Path(root).resolve() / "analyze.py")]
    stripped = {"--pltHmVmin", "--pltHmVmax", "--fontFamily", "--fs",
                "--imgFormat", "--hm-map-export"}
    filtered = argv[:2]
    index = 2
    while index < len(argv):
        flag = argv[index].split("=", 1)[0]
        if flag in stripped:
            index += 1 if "=" in argv[index] else 2
        else:
            filtered.append(argv[index])
            index += 1
    if "--no-wall-cache" not in filtered:
        filtered.append("--no-wall-cache")
    return filtered + ["--hm-map-export", str(export_path)]


def heatmap_cache_provenance(root, command):
    """Match the experiment notebook's manifest format; no source videos needed."""
    root = Path(root).resolve()
    config = root / ".analyze.local.env"
    digest = hashlib.sha256()
    sources = [root / "analyze.py", *sorted((root / "src/analysis").rglob("*.py")),
               *sorted((root / "src/utils").rglob("*.py"))]
    for source in sources:
        digest.update(str(source.relative_to(root)).encode())
        digest.update(source.read_bytes())
    return {
        "version": 1,
        "command": heatmap_analysis_command(root, command, "<cache>"),
        "local_config": config.read_text() if config.exists() else None,
        "analysis_source_sha256": digest.hexdigest(),
    }


def _matches(path, provenance):
    try:
        if not path.is_file():
            return False
        saved = json.loads(path.with_suffix(".json").read_text())
        requested = dict(provenance)
        # The manuscript toggles this switch during Run All. It controls other
        # plot outputs and never changes the heatmap probability calculation.
        for record in (saved, requested):
            config = {}
            for line in (record.get("local_config") or "").splitlines():
                if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
                    continue
                key, value = (part.strip() for part in line.split("=", 1))
                if key != "ENABLE_DEFAULT_BETWEEN_REWARD_SLI_PLOTS":
                    config[key] = value
            record["local_config"] = config
        return saved == requested
    except (OSError, ValueError):
        return False


def ensure_heatmap_cache(root, command, target, *, rebuild=False, reuse_paths=()):
    """Reuse matching maps or analyze once in an isolated working directory.

    Input tracking changes require explicit rebuilding. A failed analysis leaves
    any previous cache intact. Alternate caches (e.g. from the scale experiment)
    are adopted only when their provenance matches the requested analysis.
    """
    root, target = Path(root).resolve(), Path(target).resolve()
    provenance = heatmap_cache_provenance(root, command)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rebuild:
        if _matches(target, provenance):
            print(f"Reusing heatmap probability maps: {target}")
            return target
        for candidate in map(Path, reuse_paths):
            if _matches(candidate, provenance):
                shutil.copy2(candidate, target)
                target.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
                print(f"Adopted heatmap probability maps from {candidate}")
                return target

    log = target.with_suffix(".log")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    with tempfile.TemporaryDirectory(prefix="analysis_", dir=target.parent) as work:
        work = Path(work)
        (work / "imgs").mkdir()
        config = root / ".analyze.local.env"
        if config.exists():
            shutil.copy2(config, work / config.name)
        fresh = work / "probability_maps.npz"
        argv = heatmap_analysis_command(root, command, fresh)
        print(f"Analyzing heatmaps; progress log: {log}", flush=True)
        with log.open("w") as stream:
            stream.write(shlex.join(argv) + "\n")
            stream.flush()
            subprocess.run(argv, cwd=work, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        if not fresh.is_file():
            raise FileNotFoundError(f"No fresh heatmap cache generated; inspect {log}")
        os.replace(fresh, target)
    target.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    return target
