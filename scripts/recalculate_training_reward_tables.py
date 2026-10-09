#!/usr/bin/env python3
"""Recalculate the reward-summary page's cohorts using analyze.py log summaries."""

import argparse
import csv
import glob
import hashlib
import html
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("training_reward_cohorts.json")
HEADERS = {
    "rpm": "rewards per minute:",
    "rpd": "rewards per distance traveled [m⁻¹], exp-minus-yoked:",
}
NUMBER = r"(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan)"


def load_manifest(path):
    manifest = json.loads(path.read_text())
    cohorts = manifest["cohorts"]
    if len(manifest["columns"]) != 6 or not cohorts:
        raise ValueError("Manifest must have six metadata columns and at least one cohort")
    ids = [c["id"] for c in cohorts]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", x) for x in ids):
        raise ValueError("Cohort IDs must be unique filename-safe identifiers")
    for c in cohorts:
        if len(c["metadata"]) != 6 or c["metadata"][4] not in ("HTL", "Large"):
            raise ValueError(f"Invalid metadata/chamber for {c['id']}")
        if not c["videos"] or any(not v.startswith("/media/") for v in c["videos"]):
            raise ValueError(f"Missing or invalid video patterns for {c['id']}")
    return manifest


def analysis_command(cohort, python):
    large = cohort["metadata"][4] == "Large"
    return [
        python, "-u", str(ROOT / "analyze.py"),
        "-v", ",".join(cohort["videos"]),
        "-f", "0-1" if large else "0-9",
        "--rCC" if large else "--rmCC", "15" if large else "5",
        "--reward-pi-sync", "reward", "--no-export-bundle-exit-early",
    ]


def validate_video_patterns(cohort):
    # Match analyze.py's explicit AVI_X filter without importing OpenCV.
    missing = [pattern for pattern in cohort["videos"] if not any(
        re.search(r"\.avi$", name, re.I) and Path(name).is_file()
        for name in glob.iglob(pattern)
    )]
    if missing:
        raise ValueError(f"{cohort['id']}: video patterns matched no .avi files: " + ", ".join(missing))


def parse_summaries(text):
    """Require one complete single-cohort summary for each requested metric."""
    metrics = {}
    for metric, header in HEADERS.items():
        blocks = re.split(r"(?m)^" + re.escape(header) + r"\s*$", text)
        if len(blocks) != 2:
            raise ValueError(f"Expected exactly one {metric} summary")
        block = blocks[1].strip().split("\n\n", 1)[0]
        n_match = re.search(r'^\s*n = (\d+)\s+\(in "\(\)" below if different\)\s*$', block, re.M)
        if not n_match:
            raise ValueError(f"Missing single-cohort sample size for {metric}")
        nominal_n = int(n_match[1])
        label = "exp fly" if metric == "rpm" else "exp-minus-yoked"
        rows = re.findall(
            rf"^\s*t([123]), {label}: ({NUMBER})\s+±\s*({NUMBER})(?: \((\d+)\))?\s*$",
            block, re.M,
        )
        if len(rows) != 3 or {r[0] for r in rows} != {"1", "2", "3"}:
            raise ValueError(f"Expected trainings 1, 2, and 3 for {metric}")
        values = {}
        for training, mean, delta, n in rows:
            if n and int(n) > nominal_n:
                raise ValueError(f"Per-training n exceeds global n for {metric}")
            values[training] = f"{mean} ±{delta}" + (f" ({n})" if n else "")
        metrics[metric] = {"n": nominal_n, "trainings": values}
    if metrics["rpm"]["n"] != metrics["rpd"]["n"]:
        raise ValueError("RPM and RPD global sample sizes disagree")
    return metrics


def atomic_write(path, text):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def render_table(manifest, results, metric):
    columns = manifest["columns"] + ["n", "training 1", "training 2", "training 3"]
    lines = ['<table style="width: 100%;" border="0">', "<tbody>", "<tr>"]
    lines.extend(f"<td><strong>{html.escape(c)}</strong></td>" for c in columns)
    lines.append("</tr>")
    for cohort in manifest["cohorts"]:
        result = results[cohort["id"]]["metrics"][metric]
        cells = cohort["metadata"] + [str(result["n"])] + [
            result["trainings"][str(t)] for t in (1, 2, 3)
        ]
        lines.append("<tr>")
        lines.extend(f"<td>{html.escape(c)}</td>" for c in cells)
        lines.append("</tr>")
    return "\n".join(lines + ["</tbody>", "</table>", ""])


def write_running_tables(out, manifest, results):
    for metric in HEADERS:
        buffer = io.StringIO()
        writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
        writer.writerow(["cohort_id"] + manifest["columns"] + ["n", "training 1", "training 2", "training 3"])
        for cohort in manifest["cohorts"]:
            if cohort["id"] not in results:
                continue
            result = results[cohort["id"]]["metrics"][metric]
            writer.writerow([cohort["id"]] + cohort["metadata"] + [result["n"]] + [
                result["trainings"][str(t)] for t in (1, 2, 3)
            ])
        atomic_write(out / f"{metric}_running.tsv", buffer.getvalue())


def source_digest():
    digest = hashlib.sha256()
    # Resume uses the current source contents, including uncommitted changes.
    paths = [ROOT / "analyze.py", Path(__file__)] + sorted((ROOT / "src").rglob("*.py"))
    local_settings = ROOT / ".analyze.local.env"
    if local_settings.exists():
        paths.append(local_settings)
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_batch(args):
    manifest = load_manifest(args.manifest)
    commands = {c["id"]: analysis_command(c, args.python) for c in manifest["cohorts"]}
    if args.dry_run:
        for c in manifest["cohorts"]:
            print(f"[{c['id']}] {' | '.join(c['metadata'])}")
            print(shlex.join(commands[c["id"]]))
        return

    out = args.out.resolve()
    state_path = out / "results.json"
    signature = hashlib.sha256(json.dumps({
        "manifest": manifest, "commands": commands, "source": source_digest(),
    }, sort_keys=True).encode()).hexdigest()
    if state_path.exists():
        if not args.resume:
            raise ValueError("Output already contains results; use --resume or a new --out directory")
        state = json.loads(state_path.read_text())
        if state["signature"] != signature:
            if state["results"]:
                raise ValueError("Manifest, commands, or analysis source changed; use a new --out directory")
            # A failed preflight has no completed results to invalidate.
            state["signature"] = signature
    else:
        state = {"signature": signature, "results": {}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    atomic_write(out / "cohorts.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    atomic_write(state_path, json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    write_running_tables(out, manifest, state["results"])

    for index, cohort in enumerate(manifest["cohorts"], 1):
        key = cohort["id"]
        if key in state["results"]:
            print(f"[{index}/{len(commands)}] {key}: already complete", flush=True)
            continue
        command = commands[key]
        validate_video_patterns(cohort)
        log_path = out / "logs" / f"{key}.log"
        atomic_write(out / "logs" / f"{key}.command.json", json.dumps(command, indent=2) + "\n")
        print(f"[{index}/{len(commands)}] {key}: running; log: {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise ValueError(f"{key}: analysis exited {completed.returncode}; see {log_path}. Resume to retry.")
        try:
            metrics = parse_summaries(log_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValueError(f"{key}: {exc}; see {log_path}") from exc
        state["results"][key] = {"metrics": metrics, "command": command}
        atomic_write(state_path, json.dumps(state, indent=2, ensure_ascii=False) + "\n")
        write_running_tables(out, manifest, state["results"])

    for metric in HEADERS:
        atomic_write(out / f"{metric}_table_html.txt", render_table(manifest, state["results"], metric))
    print(f"Complete: {out / 'rpm_table_html.txt'} and {out / 'rpd_table_html.txt'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=ROOT / "exports" / "training_reward_tables")
    parser.add_argument("--python", default=sys.executable, help="Python environment for analyze.py")
    parser.add_argument("--dry-run", action="store_true", help="Print all cohort commands without running or writing")
    parser.add_argument("--resume", action="store_true", help="Reuse successful results and retry the first incomplete cohort")
    args = parser.parse_args()
    try:
        run_batch(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
