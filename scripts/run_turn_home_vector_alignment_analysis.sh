#!/usr/bin/env bash
set -euo pipefail

DATE_TAG="${DATE_TAG:-$(date +%F)}"
PRINT_ONLY="${PRINT_ONLY:-0}"
PLOT_ONLY="${PLOT_ONLY:-0}"
OUT_DIR="${OUT_DIR:-exports/turn_home_vector_alignment}"
VIDEO_LISTS_FILE="${VIDEO_LISTS_FILE:-video_lists.log}"

TURN_FILTER="${TURN_FILTER:-all}"
TURN_ANCHOR="${TURN_ANCHOR:-frame}"
TURN_MIN_TURNS="${TURN_MIN_TURNS:-5}"

TURN_ANGULAR_SOURCE="${TURN_ANGULAR_SOURCE:-path_no_head_body}"
TURN_ANGULAR_SMALL_DEG_S="${TURN_ANGULAR_SMALL_DEG_S:-80}"
TURN_ANGULAR_LARGE_DEG_S="${TURN_ANGULAR_LARGE_DEG_S:-120}"
TURN_SCORE_THRESHOLD="${TURN_SCORE_THRESHOLD:-1.4}"
TURN_PATH_MIN_SPEED_MM_S="${TURN_PATH_MIN_SPEED_MM_S:-2}"
TURN_MIN_SEGMENTS="${TURN_MIN_SEGMENTS:-2}"

SYNC_SKIP="${SYNC_SKIP:-1}"
SYNC_KEEP="${SYNC_KEEP:-4}"
TRAININGS="${TRAININGS:-1,2}"
SLI_SELECT_TRAINING="${SLI_SELECT_TRAINING:-2}"
SLI_SELECT_SKIP="${SLI_SELECT_SKIP:-1}"
SLI_SELECT_KEEP="${SLI_SELECT_KEEP:-4}"
TOP_SLI_FRACTION="${TOP_SLI_FRACTION:-0.2}"
BOTTOM_SLI_FRACTION="${BOTTOM_SLI_FRACTION:-0.5}"
RUN_SLI_SUBSETS="${RUN_SLI_SUBSETS:-1}"
FLIES="${FLIES:-0-1}"
RCC="${RCC:-15}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-nvsl}}"

GROUP_VARS=(INTACT_CTRL INTACT_PFND AR_CTRL)
GROUP_SLUGS=(intact_ctrlKir intact_pfnKir ar_ctrlKir)
GROUP_LABELS=("Ctrl>Kir FLC" "PFNd>Kir FLC" "AR Ctrl>Kir FLC")

require_dataset_vars() {
  for var_name in "$@"; do
    if [[ -z "${!var_name:-}" ]]; then
      echo "Missing required dataset variable: $var_name" >&2
      echo "Define it in the environment or in $VIDEO_LISTS_FILE before running." >&2
      exit 1
    fi
  done
}

load_dataset_vars_from_log() {
  local line
  if [[ ! -f "$VIDEO_LISTS_FILE" ]]; then
    return
  fi

  while IFS= read -r line; do
    case "$line" in
      export\ INTACT_CTRL=*|export\ INTACT_PFND=*|export\ AR_CTRL=*)
        eval "$line"
        ;;
    esac
  done < "$VIDEO_LISTS_FILE"
}

run_cmd() {
  printf '\n'
  printf '%q ' "$@"
  printf '\n'

  if [[ "$PRINT_ONLY" != "1" ]]; then
    "$@"
  fi
}

join_by_comma() {
  local IFS=,
  echo "$*"
}

turn_filter_slug() {
  case "$TURN_FILTER" in
    all)
      echo "allTurns"
      ;;
    exclude_wall_contact)
      echo "noWall"
      ;;
    *)
      echo "Unknown TURN_FILTER: $TURN_FILTER" >&2
      exit 1
      ;;
  esac
}

if [[ "$PLOT_ONLY" == "0" ]]; then
  load_dataset_vars_from_log
  require_dataset_vars "${GROUP_VARS[@]}"
elif [[ "$PLOT_ONLY" != "1" ]]; then
  echo "PLOT_ONLY must be 0 or 1." >&2
  exit 1
fi
mkdir -p "$OUT_DIR"
mkdir -p "$MPLCONFIGDIR"

filter_slug="$(turn_filter_slug)"
run_slug="turnHomeVectorAlignment_${filter_slug}_${TURN_ANGULAR_SOURCE}_score${TURN_SCORE_THRESHOLD}_small${TURN_ANGULAR_SMALL_DEG_S}_large${TURN_ANGULAR_LARGE_DEG_S}_sb2-5_${DATE_TAG}"

run_alignment_set() {
  local subset_slug="$1"
  local subset_title="$2"
  local sli_group="$3"
  local fraction_flag="${4:-}"
  local fraction_value="${5:-}"

  local subset_part=""
  local subset_flags=()
  if [[ "$sli_group" != "all" ]]; then
    subset_part="_${subset_slug}_sliT${SLI_SELECT_TRAINING}Sb2-5"
    subset_flags=(
      --turn-home-vector-alignment-sli-group "$sli_group"
      "$fraction_flag" "$fraction_value"
      --best-worst-trn "$SLI_SELECT_TRAINING"
      --sli-use-training-mean
      --sli-select-skip-first-sync-buckets "$SLI_SELECT_SKIP"
      --sli-select-keep-first-sync-buckets "$SLI_SELECT_KEEP"
    )
  fi

  local set_slug="${run_slug}${subset_part}"
  local bundles=()

  for i in "${!GROUP_VARS[@]}"; do
    local var_name="${GROUP_VARS[$i]}"
    local dataset=""
    if [[ "$PLOT_ONLY" != "1" ]]; then
      dataset="${!var_name}"
    fi
    local group_slug="${GROUP_SLUGS[$i]}"
    local group_label="${GROUP_LABELS[$i]}"
    local bundle="${OUT_DIR}/${set_slug}_${group_slug}.npz"
    bundles+=("$bundle")

    if [[ "$PLOT_ONLY" == "1" ]]; then
      if [[ "$PRINT_ONLY" != "1" && ! -f "$bundle" ]]; then
        echo "Missing expected bundle for PLOT_ONLY=1: $bundle" >&2
        exit 1
      fi
    else
      run_cmd \
        python analyze.py \
        -v "$dataset" \
        -f "$FLIES" \
        --rCC "$RCC" \
        --export-turn-home-vector-alignment-sli-bundle "$bundle" \
        "${subset_flags[@]}" \
        --turn-home-vector-alignment-trainings "$TRAININGS" \
        --turn-home-vector-alignment-include-pre \
        --turn-home-vector-alignment-turn-filter "$TURN_FILTER" \
        --turn-home-vector-alignment-anchor "$TURN_ANCHOR" \
        --turn-home-vector-alignment-min-turns "$TURN_MIN_TURNS" \
        --turn-home-vector-alignment-skip-first-sync-buckets "$SYNC_SKIP" \
        --turn-home-vector-alignment-keep-first-sync-buckets "$SYNC_KEEP" \
        --behavior-state-turn-angular-source "$TURN_ANGULAR_SOURCE" \
        --behavior-state-turn-angular-small-deg-s "$TURN_ANGULAR_SMALL_DEG_S" \
        --behavior-state-turn-angular-large-deg-s "$TURN_ANGULAR_LARGE_DEG_S" \
        --behavior-state-turn-score-threshold "$TURN_SCORE_THRESHOLD" \
        --behavior-state-turn-path-min-speed-mm-s "$TURN_PATH_MIN_SPEED_MM_S" \
        --behavior-state-turn-min-segments "$TURN_MIN_SEGMENTS" \
        --export-group-label "$group_label"
    fi
  done

  run_cmd \
    python -m scripts.plot_overlay_training_metric_scalar_bars \
    --input "Ctrl>Kir FLC=${bundles[0]}" \
    --input "PFNd>Kir FLC=${bundles[1]}" \
    --input "AR Ctrl>Kir FLC=${bundles[2]}" \
    --out "${OUT_DIR}/${set_slug}.png" \
    --title "${subset_title}: turn home-vector alignment improvement, sync buckets 2-5" \
    --ylabel $'Home-vector alignment\nimprovement during turn (deg)' \
    --points \
    --stats

  run_cmd \
    python -m scripts.plot_overlay_training_metric_scalar_bars \
    --input "Ctrl>Kir FLC=${bundles[0]}" \
    --input "PFNd>Kir FLC=${bundles[1]}" \
    --input "AR Ctrl>Kir FLC=${bundles[2]}" \
    --out "${OUT_DIR}/${set_slug}_deltaFromPre.png" \
    --title "${subset_title}: training change in turn home-vector alignment improvement, sync buckets 2-5" \
    --ylabel $'Training - pre-training change\nin turn improvement (deg)' \
    --baseline-delta-panel "Pre-training" \
    --baseline-delta-target-panel "Training 1" \
    --baseline-delta-target-panel "Training 2" \
    --points \
    --stats

  run_cmd \
    python -m scripts.stats_turn_home_vector_alignment \
    --input "Ctrl>Kir FLC=${bundles[0]}" \
    --input "PFNd>Kir FLC=${bundles[1]}" \
    --input "AR Ctrl>Kir FLC=${bundles[2]}" \
    --out-tsv "${OUT_DIR}/${set_slug}_stats.tsv"

  local bundle_csv
  bundle_csv="$(join_by_comma "${bundles[@]}")"
  printf '\n[%s] Bundles: %s\n' "$subset_title" "$bundle_csv"
  printf '[%s] Overview plot: %s/%s.png\n' "$subset_title" "$OUT_DIR" "$set_slug"
  printf '[%s] Delta-from-pre plot: %s/%s_deltaFromPre.png\n' "$subset_title" "$OUT_DIR" "$set_slug"
  printf '[%s] Stats TSV: %s/%s_stats.tsv\n' "$subset_title" "$OUT_DIR" "$set_slug"
}

run_alignment_set "" "All flies" all

if [[ "$RUN_SLI_SUBSETS" == "1" ]]; then
  run_alignment_set top20 "Top 20% SLI" top --top-sli-fraction "$TOP_SLI_FRACTION"
  run_alignment_set bottom50 "Bottom 50% SLI" bottom --bottom-sli-fraction "$BOTTOM_SLI_FRACTION"
elif [[ "$RUN_SLI_SUBSETS" != "0" ]]; then
  echo "RUN_SLI_SUBSETS must be 0 or 1." >&2
  exit 1
fi
