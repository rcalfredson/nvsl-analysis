#!/usr/bin/env bash
set -euo pipefail

# Generate the four per-episode debug exports consumed by
# scripts/collate_episode_audit.py for the AR vs PFNd turnback/containment audit.
#
# Expected dataset variables:
#   INTACT_PFND  PFNd>Kir intact large-chamber video list
#   AR_CTRL      AR Ctrl>Kir large-chamber video list
#
# Useful dry run:
#   PRINT_ONLY=1 scripts/run_episode_audit_debug_exports.sh

PRINT_ONLY="${PRINT_ONLY:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FLIES_ARG="${FLIES_ARG:-0-1}"
REWARD_CIRCLE_CUTOFF="${REWARD_CIRCLE_CUTOFF:-15}"

TURNBACK_PAIRS_MM="${TURNBACK_PAIRS_MM:-3:5,8:10,13:15}"
RETURN_RADII_MM="${RETURN_RADII_MM:-6,11,16}"
TURNBACK_MIN_WALKING_FRACTION="${TURNBACK_MIN_WALKING_FRACTION:-0.0}"

OUT_DIR="${OUT_DIR:-tmp/episode_audit}"
BUNDLE_DIR="${BUNDLE_DIR:-exports/episode_audit}"

require_dataset_vars() {
  for var_name in "$@"; do
    if [[ -z "${!var_name:-}" ]]; then
      echo "Missing required dataset variable: $var_name" >&2
      echo "Source video_lists.log or export the variable before running this script." >&2
      exit 1
    fi
  done
}

run_cmd() {
  printf '\n'
  printf '%q ' "$@"
  printf '\n'

  if [[ "$PRINT_ONLY" != "1" ]]; then
    "$@"
  fi
}

run_turnback_export() {
  local dataset="$1"
  local group_slug="$2"
  local group_label="$3"

  run_cmd \
    "$PYTHON_BIN" analyze.py \
    -v "$dataset" \
    -f "$FLIES_ARG" \
    --rCC "$REWARD_CIRCLE_CUTOFF" \
    --export-turnback-excursion-bin-sli-bundle \
      "$BUNDLE_DIR/turnbackPairs_minEpSb5Filt_wall_${group_slug}_flatLgc_T2_p3-5_8-10_13-15_debug.npz" \
    --turnback-excursion-bin-debug-episodes-csv \
      "$OUT_DIR/turnbackPairs_minEpSb5Filt_wall_${group_slug}_flatLgc_T2_p3-5_8-10_13-15_debug.csv" \
    --turnback-excursion-bin-radius-pairs-mm "$TURNBACK_PAIRS_MM" \
    --turnback-excursion-bin-trainings 2 \
    --turnback-excursion-bin-skip-first-sync-buckets 1 \
    --turnback-excursion-bin-min-walking-fraction "$TURNBACK_MIN_WALKING_FRACTION" \
    --best-worst-trn 2 \
    --sli-use-training-mean \
    --sli-select-skip-first-sync-buckets 1 \
    --sli-select-keep-first-sync-buckets 4 \
    --require-exp-target-sync-bucket \
    --export-group-label "$group_label"
}

run_return_prob_export() {
  local dataset="$1"
  local group_slug="$2"
  local group_label="$3"

  run_cmd \
    "$PYTHON_BIN" analyze.py \
    -v "$dataset" \
    -f "$FLIES_ARG" \
    --rCC "$REWARD_CIRCLE_CUTOFF" \
    --export-return-prob-outer-radius-sli-bundle \
      "$BUNDLE_DIR/returnProb_minEpSb5Filt_wall_${group_slug}_flatLgc_T2_r6_11_16_debug.npz" \
    --return-prob-outer-radius-debug-episodes-csv \
      "$OUT_DIR/returnProb_minEpSb5Filt_wall_${group_slug}_flatLgc_T2_r6_11_16_debug.csv" \
    --return-prob-outer-radius-outer-radii-mm "$RETURN_RADII_MM" \
    --return-prob-outer-radius-trainings 2 \
    --return-prob-outer-radius-skip-first-sync-buckets 1 \
    --best-worst-trn 2 \
    --sli-use-training-mean \
    --sli-select-skip-first-sync-buckets 1 \
    --require-exp-target-sync-bucket \
    --export-group-label "$group_label"
}

require_dataset_vars INTACT_PFND AR_CTRL

mkdir -p "$OUT_DIR" "$BUNDLE_DIR"

run_turnback_export "$INTACT_PFND" "intact_pfnKir" "PFNd>Kir FLC"
run_turnback_export "$AR_CTRL" "ar_ctrlKir" "AR Ctrl>Kir FLC"
run_return_prob_export "$INTACT_PFND" "intact_pfnKir" "PFNd>Kir FLC"
run_return_prob_export "$AR_CTRL" "ar_ctrlKir" "AR Ctrl>Kir FLC"
