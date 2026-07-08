#!/usr/bin/env bash
set -euo pipefail

DATE_TAG="${DATE_TAG:-$(date +%F)}"
PRINT_ONLY="${PRINT_ONLY:-0}"
OUT_DIR="${OUT_DIR:-exports/turn_home_vector_alignment}"
VIDEO_LISTS_FILE="${VIDEO_LISTS_FILE:-video_lists.log}"

TURN_FILTER="${TURN_FILTER:-all}"
TURN_ANCHOR="${TURN_ANCHOR:-frame}"
TURN_MIN_TURNS="${TURN_MIN_TURNS:-1}"

TURN_ANGULAR_SOURCE="${TURN_ANGULAR_SOURCE:-path_no_head_body}"
TURN_ANGULAR_SMALL_DEG_S="${TURN_ANGULAR_SMALL_DEG_S:-100}"
TURN_ANGULAR_LARGE_DEG_S="${TURN_ANGULAR_LARGE_DEG_S:-140}"
TURN_SCORE_THRESHOLD="${TURN_SCORE_THRESHOLD:-1.4}"
TURN_PATH_MIN_SPEED_MM_S="${TURN_PATH_MIN_SPEED_MM_S:-2}"
TURN_MIN_SEGMENTS="${TURN_MIN_SEGMENTS:-2}"

SYNC_SKIP="${SYNC_SKIP:-1}"
SYNC_KEEP="${SYNC_KEEP:-4}"
TRAININGS="${TRAININGS:-1,2}"
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

load_dataset_vars_from_log
require_dataset_vars "${GROUP_VARS[@]}"
mkdir -p "$OUT_DIR"
mkdir -p "$MPLCONFIGDIR"

filter_slug="$(turn_filter_slug)"
run_slug="turnHomeVectorAlignment_${filter_slug}_${TURN_ANGULAR_SOURCE}_score${TURN_SCORE_THRESHOLD}_small${TURN_ANGULAR_SMALL_DEG_S}_large${TURN_ANGULAR_LARGE_DEG_S}_sb2-5_${DATE_TAG}"

bundles=()

for i in "${!GROUP_VARS[@]}"; do
  var_name="${GROUP_VARS[$i]}"
  dataset="${!var_name}"
  group_slug="${GROUP_SLUGS[$i]}"
  group_label="${GROUP_LABELS[$i]}"
  bundle="${OUT_DIR}/${run_slug}_${group_slug}.npz"
  bundles+=("$bundle")

  run_cmd \
    python analyze.py \
    -v "$dataset" \
    -f "$FLIES" \
    --rCC "$RCC" \
    --export-turn-home-vector-alignment-sli-bundle "$bundle" \
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
done

run_cmd \
  python -m scripts.plot_overlay_training_metric_scalar_bars \
  --input "Ctrl>Kir FLC=${bundles[0]}" \
  --input "PFNd>Kir FLC=${bundles[1]}" \
  --input "AR Ctrl>Kir FLC=${bundles[2]}" \
  --out "${OUT_DIR}/${run_slug}.png" \
  --title "Turn home-vector alignment improvement, sync buckets 2-5" \
  --ylabel "Home-vector alignment improvement during turn (deg)" \
  --points \
  --stats

run_cmd \
  python -m scripts.stats_turn_home_vector_alignment \
  --input "Ctrl>Kir FLC=${bundles[0]}" \
  --input "PFNd>Kir FLC=${bundles[1]}" \
  --input "AR Ctrl>Kir FLC=${bundles[2]}" \
  --out-tsv "${OUT_DIR}/${run_slug}_stats.tsv"

bundle_csv="$(join_by_comma "${bundles[@]}")"
printf '\nBundles: %s\n' "$bundle_csv"
printf 'Overview plot: %s/%s.png\n' "$OUT_DIR" "$run_slug"
printf 'Stats TSV: %s/%s_stats.tsv\n' "$OUT_DIR" "$run_slug"
