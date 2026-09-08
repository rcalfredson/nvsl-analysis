#!/usr/bin/env bash
# Sliding-film recordings from July/August 2025.
# Run export, then manifest; review the physical-fly mapping before render.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

case "${1:-}" in
    export|manifest|render)
        # Backward-compatible form: a stage alone means the standard film slide.
        film_cohort=film-slide
        film_action=$1
        ;;
    *)
        film_cohort=${1:-}
        film_action=${2:-}
        ;;
esac

case "$film_cohort" in
    film-slide)
        film_default_out=exports/film_slide_paired
        date_patterns=('2025-07-19' '2025-07-2[012358]' '2025-07-31' '2025-08-0[124]')
        cameras=(c31 c41 c51 c61)
        ;;
    mock-slide)
        film_default_out=exports/mock_slide_paired
        date_patterns=('2025-07-19' '2025-07-2[012358]' '2025-07-31' '2025-08-0[124]')
        cameras=(c32 c42 c52 c62)
        ;;
    antennae-removed)
        film_default_out=exports/antennae_removed_slide_paired
        date_patterns=('2025-08-0[56789]' '2025-08-1[0123]')
        cameras=(c31 c41 c51 c61)
        ;;
    *)
        echo "Unknown cohort: ${film_cohort:-<missing>}" >&2
        echo "Choose film-slide, mock-slide, or antennae-removed." >&2
        exit 2
        ;;
esac

film_python=${FILM_PYTHON:-python}
film_out=${FILM_PAIR_DIR:-$film_default_out}
film_root='/media/Synology4/Yang Chen'
before_patterns=()
after_patterns=()
for date_pattern in "${date_patterns[@]}"; do
    for camera in "${cameras[@]}"; do
        if [[ "$film_cohort" == film-slide && "$date_pattern" == '2025-07-31' && "$camera" == c31 ]]; then
            continue
        fi
        before_patterns+=("$film_root/$date_pattern/${camera}_*")
        after_patterns+=("$film_root/$date_pattern/2nd round/${camera}_*")
    done
done
before_videos=$(IFS=,; echo "${before_patterns[*]}")
after_videos=$(IFS=,; echo "${after_patterns[*]}")
common=(-f 0-1 --rCC 15 --pltHm --hm-periods training --sb 10
        --pltHmVmin 1e-6 --pltHmVmax 1e-3 --imgFormat pdf
        --fontFamily Arial --fs 20)
before_window=(--num-trainings 2 --hm-sync-bucket 5 --hm-sync-bucket-tail-minutes 5)
after_window=(--num-trainings 1 --hm-sync-bucket 1 --hm-sync-bucket-head-minutes 5)
pairing=(--hm-pair-manifest "$film_out/pairs.csv"
         --hm-pair-before-report "$film_out/before.json"
         --hm-pair-after-report "$film_out/after.json")

case "$film_action" in
    export)
        mkdir -p -- "$film_out"
        "$film_python" analyze.py -v "$before_videos" "${common[@]}" "${before_window[@]}" \
            --hm-pair-export "$film_out/before.json"
        "$film_python" analyze.py -v "$after_videos" "${common[@]}" "${after_window[@]}" \
            --hm-pair-export "$film_out/after.json"
        ;;
    manifest)
        "$film_python" - "$film_out" <<'PY'
import csv
import json
from pathlib import Path
import re
import sys

out = Path(sys.argv[1])


def indexed(side):
    report = json.loads((out / f"{side}.json").read_text())
    result = {}
    for row in report["records"]:
        path = Path(row["video"])
        dates = [part for part in path.parts[:-1]
                 if re.fullmatch(r"\d{4}-\d{2}-\d{2}", part)]
        camera = re.match(r"(c\d+)_", path.name)
        if len(dates) != 1 or camera is None:
            raise SystemExit(f"Cannot identify date/camera: {path}; create pairs.csv manually")
        key = (dates[0], camera.group(1), str(row["fly"]))
        if key in result:
            raise SystemExit(f"Ambiguous {side} match {key}: {result[key]['video']} and {path}; create pairs.csv manually")
        result[key] = row
    return result


before, after = indexed("before"), indexed("after")
shared = before.keys() & after.keys()
unmatched_path = out / "unmatched.csv"
unmatched_count = 0
with unmatched_path.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["side", "date", "camera", "fly", "video", "reason"])
    for side, records, other in (("before", before, after), ("after", after, before)):
        for key in sorted(records.keys() - other.keys()):
            missing_side = "after" if side == "before" else "before"
            writer.writerow([side, *key, records[key]["video"], f"missing_{missing_side}_identity"])
            unmatched_count += 1
path = out / "pairs.csv"
with path.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["pair_id", "before_video", "before_fly", "after_video", "after_fly"])
    for key in sorted(shared):
        left, right = before[key], after[key]
        writer.writerow(["_".join(key), left["video"], left["fly"], right["video"], right["fly"]])
print(f"Wrote {len(shared)} candidate physical-fly pairs to {path}")
print(f"Excluded {unmatched_count} unmatched identities; report: {unmatched_path}")
if not shared:
    raise SystemExit("No shared recording/fly identities; pairs.csv contains only its header. See unmatched.csv.")
print("Review the mapping before rendering: same date/camera/fly must identify the same physical fly.")
print("Ineligible pairs remain in this manifest so the render audits can explain their exclusion.")
PY
        ;;
    render)
        "$film_python" analyze.py -v "$before_videos" "${common[@]}" "${before_window[@]}" \
            "${pairing[@]}" --hm-pair-side before --hm-pair-audit "$film_out/before_audit.csv"
        cp -- imgs/heatmaps2.pdf "$film_out/T2_SB5_last5min_paired.pdf"
        cp -- imgs/heatmaps.png "$film_out/T2_SB5_last5min_paired.png"
        "$film_python" analyze.py -v "$after_videos" "${common[@]}" "${after_window[@]}" \
            "${pairing[@]}" --hm-pair-side after --hm-pair-audit "$film_out/after_audit.csv"
        cp -- imgs/heatmaps2.pdf "$film_out/T1_SB1_first5min_paired.pdf"
        cp -- imgs/heatmaps.png "$film_out/T1_SB1_first5min_paired.png"
        ;;
    *)
        echo "Usage: bash scripts/run_film_slide_paired_heatmaps.sh COHORT {export|manifest|render}" >&2
        echo "Cohorts: film-slide, mock-slide, antennae-removed" >&2
        echo "Backward compatible: omit COHORT to use film-slide." >&2
        echo "Optional: FILM_PYTHON=/path/to/python FILM_PAIR_DIR=exports/film_slide_paired" >&2
        exit 2
        ;;
esac
