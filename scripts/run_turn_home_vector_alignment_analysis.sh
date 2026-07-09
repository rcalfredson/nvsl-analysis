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
TURN_RADIAL_BINS="${TURN_RADIAL_BINS:-}"

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

slug_decimal() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/_}"
  echo "$value"
}

fraction_percent_slug() {
  local fraction="$1"
  local pct
  pct="$(awk -v f="$fraction" 'BEGIN { printf "%g", f * 100 }')"
  slug_decimal "$pct"
}

parse_radial_bin() {
  local spec="$1"
  local __lo_var="$2"
  local __hi_var="$3"
  local item="${spec//[[:space:]]/}"
  local lo=""
  local hi=""

  if [[ "$item" == *:* ]]; then
    lo="${item%%:*}"
    hi="${item#*:}"
  elif [[ "$item" == *-* ]]; then
    lo="${item%%-*}"
    hi="${item#*-}"
  else
    echo "Radial bins must use lo-hi or lo:hi format, got: $spec" >&2
    exit 1
  fi

  if [[ -z "$lo" || -z "$hi" ]]; then
    echo "Radial bins require non-empty lower and upper bounds, got: $spec" >&2
    exit 1
  fi

  printf -v "$__lo_var" '%s' "$lo"
  printf -v "$__hi_var" '%s' "$hi"
}

bundle_with_suffix() {
  local bundle="$1"
  local suffix="$2"
  echo "${bundle%.npz}_${suffix}.npz"
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

SLI_GROUPS=(all)
SLI_SLUGS=("")
SLI_TITLES=("All flies")
if [[ "$RUN_SLI_SUBSETS" == "1" ]]; then
  top_pct="$(fraction_percent_slug "$TOP_SLI_FRACTION")"
  bottom_pct="$(fraction_percent_slug "$BOTTOM_SLI_FRACTION")"
  SLI_GROUPS+=(top bottom)
  SLI_SLUGS+=("top${top_pct}_sliT${SLI_SELECT_TRAINING}Sb$((SLI_SELECT_SKIP + 1))-$((SLI_SELECT_SKIP + SLI_SELECT_KEEP))")
  SLI_SLUGS+=("bottom${bottom_pct}_sliT${SLI_SELECT_TRAINING}Sb$((SLI_SELECT_SKIP + 1))-$((SLI_SELECT_SKIP + SLI_SELECT_KEEP))")
  SLI_TITLES+=("Top ${top_pct}% SLI" "Bottom ${bottom_pct}% SLI")
elif [[ "$RUN_SLI_SUBSETS" != "0" ]]; then
  echo "RUN_SLI_SUBSETS must be 0 or 1." >&2
  exit 1
fi

RADIAL_SLUGS=()
RADIAL_TITLES=()
if [[ -n "$TURN_RADIAL_BINS" ]]; then
  IFS=',' read -r -a radial_bins <<< "$TURN_RADIAL_BINS"
  for radial_bin in "${radial_bins[@]}"; do
    radial_lo=""
    radial_hi=""
    parse_radial_bin "$radial_bin" radial_lo radial_hi
    RADIAL_SLUGS+=("r$(slug_decimal "$radial_lo")_$(slug_decimal "$radial_hi")mm")
    RADIAL_TITLES+=("${radial_lo}-${radial_hi} mm")
  done
fi

plot_alignment_outputs() {
  local set_slug="$1"
  local title_prefix="$2"
  local bundle_ctrl="$3"
  local bundle_pfn="$4"
  local bundle_ar="$5"

  run_cmd \
    python -m scripts.plot_overlay_training_metric_scalar_bars \
    --input "Ctrl>Kir FLC=${bundle_ctrl}" \
    --input "PFNd>Kir FLC=${bundle_pfn}" \
    --input "AR Ctrl>Kir FLC=${bundle_ar}" \
    --out "${OUT_DIR}/${set_slug}.png" \
    --title "${title_prefix}: turn home-vector alignment improvement, sync buckets 2-5" \
    --ylabel $'Home-vector alignment\nimprovement during turn (deg)' \
    --points \
    --stats

  run_cmd \
    python -m scripts.plot_overlay_training_metric_scalar_bars \
    --input "Ctrl>Kir FLC=${bundle_ctrl}" \
    --input "PFNd>Kir FLC=${bundle_pfn}" \
    --input "AR Ctrl>Kir FLC=${bundle_ar}" \
    --out "${OUT_DIR}/${set_slug}_deltaFromPre.png" \
    --title "${title_prefix}: training change in turn home-vector alignment improvement, sync buckets 2-5" \
    --ylabel $'Training - pre-training change\nin turn improvement (deg)' \
    --baseline-delta-panel "Pre-training" \
    --baseline-delta-target-panel "Training 1" \
    --baseline-delta-target-panel "Training 2" \
    --points \
    --stats

  run_cmd \
    python -m scripts.stats_turn_home_vector_alignment \
    --input "Ctrl>Kir FLC=${bundle_ctrl}" \
    --input "PFNd>Kir FLC=${bundle_pfn}" \
    --input "AR Ctrl>Kir FLC=${bundle_ar}" \
    --out-tsv "${OUT_DIR}/${set_slug}_stats.tsv"

  printf '[%s] Overview plot: %s/%s.png\n' "$title_prefix" "$OUT_DIR" "$set_slug"
  printf '[%s] Delta-from-pre plot: %s/%s_deltaFromPre.png\n' "$title_prefix" "$OUT_DIR" "$set_slug"
  printf '[%s] Stats TSV: %s/%s_stats.tsv\n' "$title_prefix" "$OUT_DIR" "$set_slug"
}

run_alignment_set() {
  local sli_group_csv
  sli_group_csv="$(join_by_comma "${SLI_GROUPS[@]}")"

  local subset_flags=(
    --turn-home-vector-alignment-sli-groups "$sli_group_csv"
    --top-sli-fraction "$TOP_SLI_FRACTION"
    --bottom-sli-fraction "$BOTTOM_SLI_FRACTION"
    --best-worst-trn "$SLI_SELECT_TRAINING"
    --sli-use-training-mean
    --sli-select-skip-first-sync-buckets "$SLI_SELECT_SKIP"
    --sli-select-keep-first-sync-buckets "$SLI_SELECT_KEEP"
  )

  local radial_flags=()
  if [[ -n "$TURN_RADIAL_BINS" ]]; then
    radial_flags=(
      --turn-home-vector-alignment-radius-ranges-mm "$TURN_RADIAL_BINS"
    )
  fi

  local set_slug="${run_slug}"
  local base_bundles=()

  for i in "${!GROUP_VARS[@]}"; do
    local var_name="${GROUP_VARS[$i]}"
    local dataset=""
    if [[ "$PLOT_ONLY" != "1" ]]; then
      dataset="${!var_name}"
    fi
    local group_slug="${GROUP_SLUGS[$i]}"
    local group_label="${GROUP_LABELS[$i]}"
    local bundle="${OUT_DIR}/${set_slug}_${group_slug}.npz"
    base_bundles+=("$bundle")

    if [[ "$PLOT_ONLY" == "1" ]]; then
      if [[ "$PRINT_ONLY" != "1" ]]; then
        if [[ -z "$TURN_RADIAL_BINS" ]]; then
          if [[ ! -f "$bundle" ]]; then
            echo "Missing expected bundle for PLOT_ONLY=1: $bundle" >&2
            exit 1
          fi
        else
          for radial_slug in "${RADIAL_SLUGS[@]}"; do
            local radial_bundle
            radial_bundle="$(bundle_with_suffix "$bundle" "$radial_slug")"
            if [[ ! -f "$radial_bundle" ]]; then
              echo "Missing expected bundle for PLOT_ONLY=1: $radial_bundle" >&2
              exit 1
            fi
          done
        fi
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
        "${radial_flags[@]}" \
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

  for subset_i in "${!SLI_GROUPS[@]}"; do
    local subset_slug="${SLI_SLUGS[$subset_i]}"
    local subset_title="${SLI_TITLES[$subset_i]}"
    local subset_set_slug="$set_slug"
    local subset_bundles=()
    if [[ -n "$subset_slug" ]]; then
      subset_set_slug="${set_slug}_${subset_slug}"
    fi
    for bundle in "${base_bundles[@]}"; do
      if [[ -n "$subset_slug" ]]; then
        subset_bundles+=("$(bundle_with_suffix "$bundle" "$subset_slug")")
      else
        subset_bundles+=("$bundle")
      fi
    done

    if [[ -z "$TURN_RADIAL_BINS" ]]; then
      if [[ "$PLOT_ONLY" == "1" && "$PRINT_ONLY" != "1" ]]; then
        for bundle in "${subset_bundles[@]}"; do
          if [[ ! -f "$bundle" ]]; then
            echo "Missing expected bundle for PLOT_ONLY=1: $bundle" >&2
            exit 1
          fi
        done
      fi

      local bundle_csv
      bundle_csv="$(join_by_comma "${subset_bundles[@]}")"
      printf '\n[%s] Bundles: %s\n' "$subset_title" "$bundle_csv"
      plot_alignment_outputs \
        "$subset_set_slug" \
        "$subset_title" \
        "${subset_bundles[0]}" \
        "${subset_bundles[1]}" \
        "${subset_bundles[2]}"
    else
      for i in "${!RADIAL_SLUGS[@]}"; do
        local radial_slug="${RADIAL_SLUGS[$i]}"
        local radial_title="${RADIAL_TITLES[$i]}"
        local radial_set_slug="${subset_set_slug}_${radial_slug}"
        local radial_bundles=()
        for bundle in "${subset_bundles[@]}"; do
          radial_bundles+=("$(bundle_with_suffix "$bundle" "$radial_slug")")
        done

        if [[ "$PLOT_ONLY" == "1" && "$PRINT_ONLY" != "1" ]]; then
          for bundle in "${radial_bundles[@]}"; do
            if [[ ! -f "$bundle" ]]; then
              echo "Missing expected bundle for PLOT_ONLY=1: $bundle" >&2
              exit 1
            fi
          done
        fi

        local title_prefix="${subset_title}, ${radial_title}"
        local bundle_csv
        bundle_csv="$(join_by_comma "${radial_bundles[@]}")"
        printf '\n[%s] Bundles: %s\n' "$title_prefix" "$bundle_csv"
        plot_alignment_outputs \
          "$radial_set_slug" \
          "$title_prefix" \
          "${radial_bundles[0]}" \
          "${radial_bundles[1]}" \
          "${radial_bundles[2]}"
      done
    fi
  done
}

run_alignment_set
