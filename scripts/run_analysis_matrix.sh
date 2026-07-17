#!/usr/bin/env bash
set -euo pipefail

DATE_TAG="${DATE_TAG:-$(date +%F)}"
PRINT_ONLY="${PRINT_ONLY:-0}"
RUN_TURNBACK_HOME_VECTOR_ALIGNMENT_ONLY="${RUN_TURNBACK_HOME_VECTOR_ALIGNMENT_ONLY:-0}"
RUN_FLAT_HTL_TURNBACK_PAIRS="${RUN_FLAT_HTL_TURNBACK_PAIRS:-0}"
RUN_FLAT_HTL_TURNBACK_HOME_VECTOR_VARIANT="${RUN_FLAT_HTL_TURNBACK_HOME_VECTOR_VARIANT:-0}"
FLAT_HTL_TURNBACK_HOME_VECTOR_ALIGNMENT_THRESHOLD="${FLAT_HTL_TURNBACK_HOME_VECTOR_ALIGNMENT_THRESHOLD:-0}"
TURNBACK_EXCURSION_BIN_MIN_WALKING_FRACTION="${TURNBACK_EXCURSION_BIN_MIN_WALKING_FRACTION:-0.0}"
RETURN_LEG_TORTUOSITY_EXAMPLES="${RETURN_LEG_TORTUOSITY_EXAMPLES:-0}"
RETURN_LEG_TORTUOSITY_EXAMPLES_PER_BIN="${RETURN_LEG_TORTUOSITY_EXAMPLES_PER_BIN:-6}"
RETURN_LEG_TORTUOSITY_EXAMPLES_MAX_PER_FLY="${RETURN_LEG_TORTUOSITY_EXAMPLES_MAX_PER_FLY:-0}"
POST_WALL_DEPARTURE_TORTUOSITY_EXAMPLES="${POST_WALL_DEPARTURE_TORTUOSITY_EXAMPLES:-0}"
POST_WALL_DEPARTURE_TORTUOSITY_EXAMPLES_NUM="${POST_WALL_DEPARTURE_TORTUOSITY_EXAMPLES_NUM:-12}"
POST_WALL_DEPARTURE_TORTUOSITY_EXAMPLES_MAX_PER_FLY="${POST_WALL_DEPARTURE_TORTUOSITY_EXAMPLES_MAX_PER_FLY:-1}"
TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES="${TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES:-0}"
TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES_NUM="${TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES_NUM:-24}"
TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES_MAX_PER_FLY="${TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES_MAX_PER_FLY:-4}"
TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES_RANK_MODE="${TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES_RANK_MODE:-abs_border_minus_reentry_mean}"
TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES_RANDOM_SEED="${TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES_RANDOM_SEED:-1}"
TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES_SAMPLE_CROSSING_FILTER="${TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES_SAMPLE_CROSSING_FILTER:-all}"
TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES_DIR="${TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES_DIR:-imgs/turnback_home_vector_alignment_debug}"
TURNBACK_HOME_VECTOR_ALIGNMENT_EXCLUDE_SAMPLE_CROSSINGS="${TURNBACK_HOME_VECTOR_ALIGNMENT_EXCLUDE_SAMPLE_CROSSINGS:-0}"
TURNBACK_HOME_VECTOR_ALIGNMENT_HEADING_ESTIMATOR="${TURNBACK_HOME_VECTOR_ALIGNMENT_HEADING_ESTIMATOR:-mean}"
TURNBACK_HOME_VECTOR_ALIGNMENT_HOME_VECTOR_ANCHOR="${TURNBACK_HOME_VECTOR_ALIGNMENT_HOME_VECTOR_ANCHOR:-intersection}"
TURNBACK_HOME_VECTOR_ALIGNMENT_REUSE_EXISTING_BUNDLES="${TURNBACK_HOME_VECTOR_ALIGNMENT_REUSE_EXISTING_BUNDLES:-1}"
TURNBACK_HOME_VECTOR_ALIGNMENT_IMG_FORMAT="${TURNBACK_HOME_VECTOR_ALIGNMENT_IMG_FORMAT:-png}"
TURNBACK_HOME_VECTOR_ALIGNMENT_IMG_FORMAT="${TURNBACK_HOME_VECTOR_ALIGNMENT_IMG_FORMAT#.}"
TURNBACK_HOME_VECTOR_ALIGNMENT_IMG_FORMAT="${TURNBACK_HOME_VECTOR_ALIGNMENT_IMG_FORMAT,,}"
TURNBACK_HOME_VECTOR_ALIGNMENT_FONT_FAMILY="${TURNBACK_HOME_VECTOR_ALIGNMENT_FONT_FAMILY:-}"
TURNBACK_HOME_VECTOR_ALIGNMENT_FS="${TURNBACK_HOME_VECTOR_ALIGNMENT_FS:-}"

GROUP_VARS=(INTACT_CTRL INTACT_PFND AR_CTRL)
GROUP_SLUGS=(intact_ctrlKir intact_pfnKir ar_ctrlKir)
GROUP_LABELS=("Ctrl" "PFNd>Kir" "Antennae removed")

# Cohorts used specifically by the dual-circle turnback and turnback
# home-vector alignment analyses. Keep the general GROUP_* arrays above
# unchanged so the other analysis families in this matrix retain AR Ctrl>Kir.
TURNBACK_COMPARISON_GROUP="${TURNBACK_COMPARISON_GROUP:-mbkc_kir}"
VIDEO_LISTS_FILE="${VIDEO_LISTS_FILE:-video_lists.log}"
MBKC_HEADER="${MBKC_HEADER:-UAS>>CsC (X); 19B03-lexA (MBKC)/otd-flp; 0273Gal4/lexAop>>Kir}"
MBKC_SUBHEADER="${MBKC_SUBHEADER:-Flat-lower chamber reward circle shrink in T2, T3, closer to the center  10d old flies}"

TURNBACK_GROUP_VARS=(INTACT_CTRL INTACT_PFND)
TURNBACK_GROUP_SLUGS=(intact_ctrlKir intact_pfnKir)
TURNBACK_GROUP_LABELS=("Ctrl" "PFNd>Kir")
declare -A TURNBACK_HOME_VECTOR_ALIGNMENT_BUNDLES=()

FLAT_HTL_GROUP_VARS=(
  FLAT_HTL_CTRL
  FLAT_HTL_HIND_TARSI_GENITALIA_GLUED
  FLAT_HTL_ANTENNAE_REMOVED
)
FLAT_HTL_GROUP_SLUGS=(
  ctrl
  hindTarsiRemoved_genitaliaGlued
  antennaeRemoved
)
FLAT_HTL_GROUP_LABELS=(
  "Ctrl"
  "Hind tarsi removed + genitalia glued"
  "Antennae removed"
)

load_mbkc_dataset_from_log() {
  if [[ -n "${MBKC_KIR:-}" ]]; then
    return
  fi
  if [[ ! -f "$VIDEO_LISTS_FILE" ]]; then
    return
  fi

  MBKC_KIR="$({
    awk -v header="$MBKC_HEADER" -v subheader="$MBKC_SUBHEADER" '
      $0 == header { in_section = 1; next }
      in_section && $0 == subheader { want_command = 1; next }
      want_command && /^python analyze\.py / {
        marker = " -v \""
        start = index($0, marker)
        if (!start) exit 2
        value = substr($0, start + length(marker))
        stop = index(value, "\"")
        if (!stop) exit 2
        print substr(value, 1, stop - 1)
        exit
      }
    ' "$VIDEO_LISTS_FILE"
  } || true)"
}

case "$TURNBACK_COMPARISON_GROUP" in
  ar_ctrl)
    TURNBACK_GROUP_VARS+=(AR_CTRL)
    TURNBACK_GROUP_SLUGS+=(ar_ctrlKir)
    TURNBACK_GROUP_LABELS+=("Antennae removed")
    TURNBACK_PLOT_SUFFIX=""
    ;;
  mbkc_kir)
    load_mbkc_dataset_from_log
    TURNBACK_GROUP_VARS+=(MBKC_KIR)
    TURNBACK_GROUP_SLUGS+=(intact_mbkcKir)
    TURNBACK_GROUP_LABELS+=("MBKC>Kir")
    TURNBACK_PLOT_SUFFIX="_vs_mbkcKir"
    ;;
  *)
    echo "TURNBACK_COMPARISON_GROUP must be ar_ctrl or mbkc_kir, got: $TURNBACK_COMPARISON_GROUP" >&2
    exit 1
    ;;
esac

if [[ "$TURNBACK_HOME_VECTOR_ALIGNMENT_REUSE_EXISTING_BUNDLES" != "0" && "$TURNBACK_HOME_VECTOR_ALIGNMENT_REUSE_EXISTING_BUNDLES" != "1" ]]; then
  echo "TURNBACK_HOME_VECTOR_ALIGNMENT_REUSE_EXISTING_BUNDLES must be 0 or 1." >&2
  exit 1
fi
if [[ -z "$TURNBACK_HOME_VECTOR_ALIGNMENT_IMG_FORMAT" ]]; then
  echo "TURNBACK_HOME_VECTOR_ALIGNMENT_IMG_FORMAT must not be empty." >&2
  exit 1
fi
if [[ "$RUN_TURNBACK_HOME_VECTOR_ALIGNMENT_ONLY" != "0" && "$RUN_TURNBACK_HOME_VECTOR_ALIGNMENT_ONLY" != "1" ]]; then
  echo "RUN_TURNBACK_HOME_VECTOR_ALIGNMENT_ONLY must be 0 or 1." >&2
  exit 1
fi

require_dataset_vars() {
  for var_name in "$@"; do
    if [[ -z "${!var_name:-}" ]]; then
      echo "Missing required dataset variable: $var_name" >&2
      echo "Define it with the corresponding video list before running this script." >&2
      exit 1
    fi
  done
}

if [[ "$RUN_FLAT_HTL_TURNBACK_PAIRS" == "1" ]]; then
  require_dataset_vars "${FLAT_HTL_GROUP_VARS[@]}"
elif [[ "$RUN_FLAT_HTL_TURNBACK_PAIRS" != "0" ]]; then
  echo "RUN_FLAT_HTL_TURNBACK_PAIRS must be 0 or 1." >&2
  exit 1
elif [[ "$RUN_TURNBACK_HOME_VECTOR_ALIGNMENT_ONLY" == "1" ]]; then
  if [[ "$TURNBACK_HOME_VECTOR_ALIGNMENT_REUSE_EXISTING_BUNDLES" != "1" ]]; then
    require_dataset_vars "${TURNBACK_GROUP_VARS[@]}"
  fi
else
  require_dataset_vars "${GROUP_VARS[@]}"
  require_dataset_vars "${TURNBACK_GROUP_VARS[@]}"
fi

run_cmd() {
  printf '\n'
  printf '%q ' "$@"
  printf '\n'

  if [[ "$PRINT_ONLY" != "1" ]]; then
    "$@"
  fi
}

turnback_home_vector_alignment_plot_style_flags() {
  local -n out_flags="$1"
  out_flags=(--image-format "$TURNBACK_HOME_VECTOR_ALIGNMENT_IMG_FORMAT")
  if [[ -n "$TURNBACK_HOME_VECTOR_ALIGNMENT_FONT_FAMILY" ]]; then
    out_flags+=(--fontFamily "$TURNBACK_HOME_VECTOR_ALIGNMENT_FONT_FAMILY")
  fi
  if [[ -n "$TURNBACK_HOME_VECTOR_ALIGNMENT_FS" ]]; then
    out_flags+=(--fs "$TURNBACK_HOME_VECTOR_ALIGNMENT_FS")
  fi
}

join_by_comma() {
  local IFS=,
  echo "$*"
}

slug_decimal() {
  local value="$1"
  value="${value//- /m}"
  value="${value//-/m}"
  value="${value//./p}"
  echo "$value"
}

run_flat_htl_turnback_pairs() {
  local pairs_label="$1"
  local pairs_arg="$2"
  local filter_tag="$3"
  local wall_tag="$4"
  local variant_suffix="$5"
  shift 5
  local extra_flags=("$@")

  local filter_flags=()
  case "$filter_tag" in
    noFilt)
      filter_flags=(--min-turnback-episodes 0)
      ;;
    minEpFilt)
      filter_flags=()
      ;;
    minEpSb5Filt)
      filter_flags=(--require-exp-target-sync-bucket)
      ;;
    minEpPiFilt)
      filter_flags=(--require-exp-pi-threshold-bucket)
      ;;
    *)
      echo "Unknown flat-HTL turnback filter tag: $filter_tag" >&2
      exit 1
      ;;
  esac

  local wall_flags=()
  case "$wall_tag" in
    wall)
      wall_flags=()
      ;;
    noWall)
      wall_flags=(--turnback-excursion-bin-exclude-wall-contact)
      ;;
    *)
      echo "Unknown wall tag: $wall_tag" >&2
      exit 1
      ;;
  esac

  local bundles=()

  for i in "${!FLAT_HTL_GROUP_VARS[@]}"; do
    local var_name="${FLAT_HTL_GROUP_VARS[$i]}"
    local dataset="${!var_name}"
    local group_slug="${FLAT_HTL_GROUP_SLUGS[$i]}"
    local group_label="${FLAT_HTL_GROUP_LABELS[$i]}"
    local bundle="exports/turnbackPairs_${filter_tag}_${wall_tag}_${group_slug}_flatHtl_T2_p${pairs_label}${variant_suffix}_${DATE_TAG}.npz"

    bundles+=("$bundle")

    run_cmd \
      python analyze.py \
      -v "$dataset" \
      -f 0-9 \
      --rmCC 5 \
      --export-turnback-excursion-bin-sli-bundle "$bundle" \
      --turnback-excursion-bin-radius-pairs-mm "$pairs_arg" \
      --turnback-excursion-bin-min-walking-fraction "$TURNBACK_EXCURSION_BIN_MIN_WALKING_FRACTION" \
      --turnback-excursion-bin-trainings 2 \
      --turnback-excursion-bin-skip-first-sync-buckets 1 \
      --best-worst-trn 2 \
      --sli-use-training-mean \
      --sli-select-skip-first-sync-buckets 1 \
      --sli-select-keep-first-sync-buckets 4 \
      --export-group-label "$group_label" \
      "${filter_flags[@]}" \
      "${wall_flags[@]}" \
      "${extra_flags[@]}"

    run_cmd \
      python -m scripts.plot_turnback_excursion_bin_sli_bundles \
      --bundles "$bundle" \
      --out "exports/turnbackPairs_${filter_tag}_${wall_tag}_${group_slug}_flatHtl_T2_p${pairs_label}${variant_suffix}_top20Bottom50_sliT2Sb2-5_${DATE_TAG}.png" \
      --sli-extremes both \
      --top-sli-fraction 0.2 \
      --bottom-sli-fraction 0.5 \
      --standalone-extreme-labels \
      --stats
  done

  local bundle_csv
  bundle_csv="$(join_by_comma "${bundles[@]}")"

  run_cmd \
    python -m scripts.plot_turnback_excursion_bin_sli_bundles \
    --bundles "$bundle_csv" \
    --out "exports/turnbackPairs_${filter_tag}_${wall_tag}_flatHtl_T2_p${pairs_label}${variant_suffix}_top20_sliT2Sb2-5_${DATE_TAG}.png" \
    --sli-extremes top \
    --top-sli-fraction 0.2 \
    --stats

  run_cmd \
    python -m scripts.plot_turnback_excursion_bin_sli_bundles \
    --bundles "$bundle_csv" \
    --out "exports/turnbackPairs_${filter_tag}_${wall_tag}_flatHtl_T2_p${pairs_label}${variant_suffix}_${DATE_TAG}.png" \
    --stats
}

if [[ "$RUN_FLAT_HTL_TURNBACK_PAIRS" == "1" ]]; then
  if [[ "$RUN_FLAT_HTL_TURNBACK_HOME_VECTOR_VARIANT" != "0" && "$RUN_FLAT_HTL_TURNBACK_HOME_VECTOR_VARIANT" != "1" ]]; then
    echo "RUN_FLAT_HTL_TURNBACK_HOME_VECTOR_VARIANT must be 0 or 1." >&2
    exit 1
  fi

  for wall_tag in wall noWall; do
    # run_flat_htl_turnback_pairs \
    #   "2-3_3-4_4-5" \
    #   "2:3,3:4,4:5" \
    #   minEpSb5Filt \
    #   "$wall_tag" \
    #   ""

    if [[ "$RUN_FLAT_HTL_TURNBACK_HOME_VECTOR_VARIANT" == "1" ]]; then
      threshold_slug="$(slug_decimal "$FLAT_HTL_TURNBACK_HOME_VECTOR_ALIGNMENT_THRESHOLD")"
      run_flat_htl_turnback_pairs \
        "2-3_3-4_4-5" \
        "2:3,3:4,4:5" \
        minEpSb5Filt \
        "$wall_tag" \
        "_homeVecAlignGe${threshold_slug}" \
        --turnback-excursion-bin-require-home-vector-alignment \
        --turnback-excursion-bin-home-vector-alignment-threshold "$FLAT_HTL_TURNBACK_HOME_VECTOR_ALIGNMENT_THRESHOLD"
    fi
  done
  exit 0
fi

turnback_home_vector_alignment_example_flags() {
  local -n out_flags="$1"
  local subset_slug="$2"
  local filter_tag="$3"
  local wall_tag="$4"
  local group_slug="$5"
  local pair_label="$6"

  out_flags=()
  if [[ "$TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES" != "1" ]]; then
    return
  fi

  local subset_part=""
  if [[ -n "$subset_slug" ]]; then
    subset_part="_${subset_slug}"
  fi
  local sample_cross_part=""
  if [[ "$TURNBACK_HOME_VECTOR_ALIGNMENT_EXCLUDE_SAMPLE_CROSSINGS" == "1" ]]; then
    sample_cross_part="_noSampleCross"
  fi
  local example_crossing_part=""
  if [[ "$TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES_SAMPLE_CROSSING_FILTER" != "all" ]]; then
    example_crossing_part="_${TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES_SAMPLE_CROSSING_FILTER}"
  fi
  local estimator_part
  estimator_part="$(turnback_home_vector_alignment_heading_estimator_suffix)"
  local home_anchor_part
  home_anchor_part="$(turnback_home_vector_alignment_home_vector_anchor_suffix)"
  local example_dir="${TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES_DIR}/turnbackHomeVectorAlignment${subset_part}_${filter_tag}_${wall_tag}_${group_slug}_flatLgc_T2_p${pair_label}_sb2-5${estimator_part}${home_anchor_part}${sample_cross_part}${example_crossing_part}_${TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES_RANK_MODE}_${DATE_TAG}"
  out_flags=(
    --export-turnback-home-vector-alignment-examples "$example_dir"
    --turnback-home-vector-alignment-examples-num "$TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES_NUM"
    --turnback-home-vector-alignment-examples-max-per-fly "$TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES_MAX_PER_FLY"
    --turnback-home-vector-alignment-examples-rank-mode "$TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES_RANK_MODE"
    --turnback-home-vector-alignment-examples-random-seed "$TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES_RANDOM_SEED"
    --turnback-home-vector-alignment-examples-sampling-boundary-crossing-filter "$TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES_SAMPLE_CROSSING_FILTER"
    --imgFormat pdf
  )
}

turnback_home_vector_alignment_sample_crossing_flags() {
  local -n out_flags="$1"
  out_flags=()
  if [[ "$TURNBACK_HOME_VECTOR_ALIGNMENT_EXCLUDE_SAMPLE_CROSSINGS" == "1" ]]; then
    out_flags=(--turnback-home-vector-alignment-exclude-sampling-boundary-crossings)
  fi
}

turnback_home_vector_alignment_sample_crossing_suffix() {
  if [[ "$TURNBACK_HOME_VECTOR_ALIGNMENT_EXCLUDE_SAMPLE_CROSSINGS" == "1" ]]; then
    echo "_noSampleCross"
  fi
}

turnback_home_vector_alignment_heading_estimator_suffix() {
  case "$TURNBACK_HOME_VECTOR_ALIGNMENT_HEADING_ESTIMATOR" in
    mean)
      ;;
    adaptive_mean_one_point)
      echo "_adaptiveMeanOnePoint"
      ;;
    one_point)
      echo "_onePoint"
      ;;
    reentry_mean)
      echo "_reentryMean"
      ;;
    endpoint)
      echo "_endpoint"
      ;;
    *)
      echo "Unknown turnback home-vector heading estimator: $TURNBACK_HOME_VECTOR_ALIGNMENT_HEADING_ESTIMATOR" >&2
      exit 1
      ;;
  esac
}

turnback_home_vector_alignment_home_vector_anchor_suffix() {
  case "$TURNBACK_HOME_VECTOR_ALIGNMENT_HOME_VECTOR_ANCHOR" in
    intersection)
      ;;
    reentry)
      echo "_reentryHome"
      ;;
    *)
      echo "Unknown turnback home-vector anchor: $TURNBACK_HOME_VECTOR_ALIGNMENT_HOME_VECTOR_ANCHOR" >&2
      exit 1
      ;;
  esac
}

run_return_prob() {
  local filter_tag="$1"
  local wall_tag="$2"

  local radii_label="6_11_16"
  local radii_arg="6,11,16"

  local filter_flags=()
  case "$filter_tag" in
    noFilt)
      filter_flags=(--min-between-reward-trajectories 0)
      ;;
    minEpFilt)
      filter_flags=()
      ;;
    minEpSb5Filt)
      filter_flags=(--require-exp-target-sync-bucket)
      ;;
    minEpPiFilt)
      filter_flags=(--require-exp-pi-threshold-bucket)
      ;;
    *)
      echo "Unknown return-prob filter tag: $filter_tag" >&2
      exit 1
      ;;
  esac

  local wall_flags=()
  case "$wall_tag" in
    wall)
      wall_flags=()
      ;;
    noWall)
      wall_flags=(--return-prob-outer-radius-exclude-wall-contact)
      ;;
    *)
      echo "Unknown wall tag: $wall_tag" >&2
      exit 1
      ;;
  esac

  local bundles=()

  for i in "${!GROUP_VARS[@]}"; do
    local var_name="${GROUP_VARS[$i]}"
    local dataset="${!var_name}"
    local group_slug="${GROUP_SLUGS[$i]}"
    local group_label="${GROUP_LABELS[$i]}"
    local bundle="exports/returnProb_${filter_tag}_${wall_tag}_${group_slug}_flatLgc_T2_r${radii_label}_${DATE_TAG}.npz"

    bundles+=("$bundle")

    run_cmd \
      python analyze.py \
      -v "$dataset" \
      -f 0-1 \
      --rCC 15 \
      --export-return-prob-outer-radius-sli-bundle "$bundle" \
      --return-prob-outer-radius-outer-radii-mm "$radii_arg" \
      --return-prob-outer-radius-trainings 2 \
      --return-prob-outer-radius-skip-first-sync-buckets 1 \
      --best-worst-trn 2 \
      --sli-use-training-mean \
      --sli-select-skip-first-sync-buckets 1 \
      --export-group-label "$group_label" \
      "${filter_flags[@]}" \
      "${wall_flags[@]}"
  done

  local bundle_csv
  bundle_csv="$(join_by_comma "${bundles[@]}")"

  run_cmd \
    python -m scripts.plot_return_prob_outer_radius_sli_bundles \
    --bundles "$bundle_csv" \
    --out "exports/returnProb_${filter_tag}_${wall_tag}_flatLgc_T2_r${radii_label}_${DATE_TAG}.png" \
    --metric ratio \
    --mode exp \
    --stats
}

run_turnback_pairs() {
  local pairs_label="$1"
  local pairs_arg="$2"
  local filter_tag="$3"
  local wall_tag="$4"

  local filter_flags=()
  case "$filter_tag" in
    noFilt)
      filter_flags=(--min-turnback-episodes 0)
      ;;
    minEpFilt)
      filter_flags=()
      ;;
    minEpSb5Filt)
      filter_flags=(--require-exp-target-sync-bucket)
      ;;
    minEpPiFilt)
      filter_flags=(--require-exp-pi-threshold-bucket)
      ;;
    *)
      echo "Unknown turnback filter tag: $filter_tag" >&2
      exit 1
      ;;
  esac

  local wall_flags=()
  case "$wall_tag" in
    wall)
      wall_flags=()
      ;;
    noWall)
      wall_flags=(--turnback-excursion-bin-exclude-wall-contact)
      ;;
    *)
      echo "Unknown wall tag: $wall_tag" >&2
      exit 1
      ;;
  esac

  local bundles=()

  for i in "${!TURNBACK_GROUP_VARS[@]}"; do
    local var_name="${TURNBACK_GROUP_VARS[$i]}"
    local dataset="${!var_name}"
    local group_slug="${TURNBACK_GROUP_SLUGS[$i]}"
    local group_label="${TURNBACK_GROUP_LABELS[$i]}"
    local bundle="exports/turnbackPairs_${filter_tag}_${wall_tag}_${group_slug}_flatLgc_T2_p${pairs_label}_${DATE_TAG}.npz"

    bundles+=("$bundle")

    run_cmd \
      python analyze.py \
      -v "$dataset" \
      -f 0-1 \
      --rCC 15 \
      --export-turnback-excursion-bin-sli-bundle "$bundle" \
      --turnback-excursion-bin-radius-pairs-mm "$pairs_arg" \
      --turnback-excursion-bin-trainings 2 \
      --turnback-excursion-bin-skip-first-sync-buckets 1 \
      --best-worst-trn 2 \
      --sli-use-training-mean \
      --sli-select-skip-first-sync-buckets 1 \
      --sli-select-keep-first-sync-buckets 4 \
      --export-group-label "$group_label" \
      "${filter_flags[@]}" \
      "${wall_flags[@]}"

    run_cmd \
      python -m scripts.plot_turnback_excursion_bin_sli_bundles \
      --bundles "$bundle" \
      --out "exports/turnbackPairs_${filter_tag}_${wall_tag}_${group_slug}_flatLgc_T2_p${pairs_label}_top20Bottom50_sliT2Sb2-5_${DATE_TAG}.png" \
      --sli-extremes both \
      --top-sli-fraction 0.2 \
      --bottom-sli-fraction 0.5 \
      --standalone-extreme-labels \
      --stats
  done

  local bundle_csv
  bundle_csv="$(join_by_comma "${bundles[@]}")"

  run_cmd \
    python -m scripts.plot_turnback_excursion_bin_sli_bundles \
    --bundles "$bundle_csv" \
    --out "exports/turnbackPairs_${filter_tag}_${wall_tag}_flatLgc_T2_p${pairs_label}_top20_sliT2Sb2-5${TURNBACK_PLOT_SUFFIX}_${DATE_TAG}.png" \
    --sli-extremes top \
    --top-sli-fraction 0.2 \
    --stats

  run_cmd \
    python -m scripts.plot_turnback_excursion_bin_sli_bundles \
    --bundles "$bundle_csv" \
    --out "exports/turnbackPairs_${filter_tag}_${wall_tag}_flatLgc_T2_p${pairs_label}${TURNBACK_PLOT_SUFFIX}_${DATE_TAG}.png" \
    --stats
}

run_turnback_home_vector_alignment() {
  local pair_label="$1"
  local inner_radius_mm="$2"
  local outer_radius_mm="$3"
  local filter_tag="$4"
  local wall_tag="$5"

  local filter_flags=()
  case "$filter_tag" in
    noFilt)
      filter_flags=(--min-turnback-episodes 0)
      ;;
    minEpFilt)
      filter_flags=()
      ;;
    minEpSb5Filt)
      filter_flags=(--require-exp-target-sync-bucket)
      ;;
    minEpPiFilt)
      filter_flags=(--require-exp-pi-threshold-bucket)
      ;;
    *)
      echo "Unknown turnback home-vector alignment filter tag: $filter_tag" >&2
      exit 1
      ;;
  esac

  local wall_flags=()
  case "$wall_tag" in
    wall)
      wall_flags=()
      ;;
    noWall)
      wall_flags=(--turnback-home-vector-alignment-exclude-wall-contact)
      ;;
    *)
      echo "Unknown wall tag: $wall_tag" >&2
      exit 1
      ;;
  esac

  local bundles=()
  local estimator_suffix
  estimator_suffix="$(turnback_home_vector_alignment_heading_estimator_suffix)"
  local home_anchor_suffix
  home_anchor_suffix="$(turnback_home_vector_alignment_home_vector_anchor_suffix)"
  local sample_cross_suffix
  sample_cross_suffix="$(turnback_home_vector_alignment_sample_crossing_suffix)"
  local sample_cross_flags=()
  turnback_home_vector_alignment_sample_crossing_flags sample_cross_flags

  for i in "${!TURNBACK_GROUP_VARS[@]}"; do
    local var_name="${TURNBACK_GROUP_VARS[$i]}"
    local dataset="${!var_name:-}"
    local group_slug="${TURNBACK_GROUP_SLUGS[$i]}"
    local group_label="${TURNBACK_GROUP_LABELS[$i]}"
    local bundle="exports/turnbackHomeVectorAlignment_${filter_tag}_${wall_tag}_${group_slug}_flatLgc_T2_p${pair_label}_sb2-5${estimator_suffix}${home_anchor_suffix}${sample_cross_suffix}_${DATE_TAG}.npz"
    local example_flags=()
    turnback_home_vector_alignment_example_flags \
      example_flags \
      "" \
      "$filter_tag" \
      "$wall_tag" \
      "$group_slug" \
      "$pair_label"

    bundles+=("$bundle")
    TURNBACK_HOME_VECTOR_ALIGNMENT_BUNDLES["all|${pair_label}|${i}"]="$bundle"

    if [[ "$TURNBACK_HOME_VECTOR_ALIGNMENT_REUSE_EXISTING_BUNDLES" == "1" && "$TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES" != "1" && -f "$bundle" ]]; then
      printf '\n[%s, %s] Reusing existing bundle; analysis skipped: %s\n' "$group_label" "$pair_label" "$bundle"
    else
      if [[ -z "$dataset" ]]; then
        echo "Missing required dataset variable for bundle export: $var_name" >&2
        exit 1
      fi
      run_cmd \
        python analyze.py \
        -v "$dataset" \
        -f 0-1 \
        --rCC 15 \
        --export-turnback-home-vector-alignment-sli-bundle "$bundle" \
        --turnback-home-vector-alignment-inner-radius-mm "$inner_radius_mm" \
        --turnback-home-vector-alignment-outer-radius-mm "$outer_radius_mm" \
        --turnback-home-vector-alignment-trainings 2 \
        --turnback-home-vector-alignment-skip-first-sync-buckets 1 \
        --turnback-home-vector-alignment-keep-first-sync-buckets 4 \
        --turnback-home-vector-alignment-window-radius-frames 2 \
        --turnback-home-vector-alignment-max-interpolated-heading-frames 1 \
        --best-worst-trn 2 \
        --sli-use-training-mean \
        --sli-select-skip-first-sync-buckets 1 \
        --sli-select-keep-first-sync-buckets 4 \
        --export-group-label "$group_label" \
        --turnback-home-vector-alignment-heading-estimator "$TURNBACK_HOME_VECTOR_ALIGNMENT_HEADING_ESTIMATOR" \
        --turnback-home-vector-alignment-home-vector-anchor "$TURNBACK_HOME_VECTOR_ALIGNMENT_HOME_VECTOR_ANCHOR" \
        "${sample_cross_flags[@]}" \
        "${filter_flags[@]}" \
        "${wall_flags[@]}" \
        "${example_flags[@]}"
    fi
  done

  local plot_style_flags=()
  turnback_home_vector_alignment_plot_style_flags plot_style_flags
  run_cmd \
    python -m scripts.plot_overlay_training_metric_scalar_bars \
    --input "${TURNBACK_GROUP_LABELS[0]}=${bundles[0]}" \
    --input "${TURNBACK_GROUP_LABELS[1]}=${bundles[1]}" \
    --input "${TURNBACK_GROUP_LABELS[2]}=${bundles[2]}" \
    --out "exports/turnbackHomeVectorAlignment_${filter_tag}_${wall_tag}_flatLgc_T2_p${pair_label}_sb2-5${estimator_suffix}${home_anchor_suffix}${sample_cross_suffix}${TURNBACK_PLOT_SUFFIX}_${DATE_TAG}.${TURNBACK_HOME_VECTOR_ALIGNMENT_IMG_FORMAT}" \
    --title "Home-vector heading alignment at re-entry, ${inner_radius_mm}/${outer_radius_mm} mm" \
    --ylabel "Home-vector heading alignment at re-entry" \
    --swarm \
    --stats \
    "${plot_style_flags[@]}"
}

run_turnback_home_vector_alignment_sli_subset() {
  local subset_slug="$1"
  local subset_title="$2"
  local sli_group="$3"
  local sli_fraction_flag="$4"
  local sli_fraction="$5"
  local pair_label="$6"
  local inner_radius_mm="$7"
  local outer_radius_mm="$8"
  local filter_tag="$9"
  local wall_tag="${10}"

  local subset_flags=(
    --turnback-home-vector-alignment-sli-group "$sli_group"
    "$sli_fraction_flag" "$sli_fraction"
  )

  run_turnback_home_vector_alignment_subset_impl \
    "$subset_slug" \
    "$subset_title" \
    "$pair_label" \
    "$inner_radius_mm" \
    "$outer_radius_mm" \
    "$filter_tag" \
    "$wall_tag" \
    "${subset_flags[@]}"
}

run_turnback_home_vector_alignment_subset_impl() {
  local subset_slug="$1"
  local subset_title="$2"
  local pair_label="$3"
  local inner_radius_mm="$4"
  local outer_radius_mm="$5"
  local filter_tag="$6"
  local wall_tag="$7"
  shift 7
  local subset_flags=("$@")

  local filter_flags=()
  case "$filter_tag" in
    noFilt)
      filter_flags=(--min-turnback-episodes 0)
      ;;
    minEpFilt)
      filter_flags=()
      ;;
    minEpSb5Filt)
      filter_flags=(--require-exp-target-sync-bucket)
      ;;
    minEpPiFilt)
      filter_flags=(--require-exp-pi-threshold-bucket)
      ;;
    *)
      echo "Unknown turnback home-vector alignment filter tag: $filter_tag" >&2
      exit 1
      ;;
  esac

  local wall_flags=()
  case "$wall_tag" in
    wall)
      wall_flags=()
      ;;
    noWall)
      wall_flags=(--turnback-home-vector-alignment-exclude-wall-contact)
      ;;
    *)
      echo "Unknown wall tag: $wall_tag" >&2
      exit 1
      ;;
  esac

  local bundles=()
  local estimator_suffix
  estimator_suffix="$(turnback_home_vector_alignment_heading_estimator_suffix)"
  local home_anchor_suffix
  home_anchor_suffix="$(turnback_home_vector_alignment_home_vector_anchor_suffix)"
  local sample_cross_suffix
  sample_cross_suffix="$(turnback_home_vector_alignment_sample_crossing_suffix)"
  local sample_cross_flags=()
  turnback_home_vector_alignment_sample_crossing_flags sample_cross_flags

  for i in "${!TURNBACK_GROUP_VARS[@]}"; do
    local var_name="${TURNBACK_GROUP_VARS[$i]}"
    local dataset="${!var_name:-}"
    local group_slug="${TURNBACK_GROUP_SLUGS[$i]}"
    local group_label="${TURNBACK_GROUP_LABELS[$i]}"
    local bundle="exports/turnbackHomeVectorAlignment_${subset_slug}_${filter_tag}_${wall_tag}_${group_slug}_flatLgc_T2_p${pair_label}_sliT2Sb2-5${estimator_suffix}${home_anchor_suffix}${sample_cross_suffix}_${DATE_TAG}.npz"
    local example_flags=()
    turnback_home_vector_alignment_example_flags \
      example_flags \
      "$subset_slug" \
      "$filter_tag" \
      "$wall_tag" \
      "$group_slug" \
      "$pair_label"

    bundles+=("$bundle")
    TURNBACK_HOME_VECTOR_ALIGNMENT_BUNDLES["${subset_slug}|${pair_label}|${i}"]="$bundle"

    if [[ "$TURNBACK_HOME_VECTOR_ALIGNMENT_REUSE_EXISTING_BUNDLES" == "1" && "$TURNBACK_HOME_VECTOR_ALIGNMENT_EXAMPLES" != "1" && -f "$bundle" ]]; then
      printf '\n[%s, %s, %s] Reusing existing bundle; analysis skipped: %s\n' "$subset_title" "$group_label" "$pair_label" "$bundle"
    else
      if [[ -z "$dataset" ]]; then
        echo "Missing required dataset variable for bundle export: $var_name" >&2
        exit 1
      fi
      run_cmd \
        python analyze.py \
        -v "$dataset" \
        -f 0-1 \
        --rCC 15 \
        --export-turnback-home-vector-alignment-sli-bundle "$bundle" \
        "${subset_flags[@]}" \
        --turnback-home-vector-alignment-inner-radius-mm "$inner_radius_mm" \
        --turnback-home-vector-alignment-outer-radius-mm "$outer_radius_mm" \
        --turnback-home-vector-alignment-trainings 2 \
        --turnback-home-vector-alignment-skip-first-sync-buckets 1 \
        --turnback-home-vector-alignment-keep-first-sync-buckets 4 \
        --turnback-home-vector-alignment-window-radius-frames 2 \
        --turnback-home-vector-alignment-max-interpolated-heading-frames 1 \
        --best-worst-trn 2 \
        --sli-use-training-mean \
        --sli-select-skip-first-sync-buckets 1 \
        --sli-select-keep-first-sync-buckets 4 \
        --export-group-label "$group_label" \
        --turnback-home-vector-alignment-heading-estimator "$TURNBACK_HOME_VECTOR_ALIGNMENT_HEADING_ESTIMATOR" \
        --turnback-home-vector-alignment-home-vector-anchor "$TURNBACK_HOME_VECTOR_ALIGNMENT_HOME_VECTOR_ANCHOR" \
        "${sample_cross_flags[@]}" \
        "${filter_flags[@]}" \
        "${wall_flags[@]}" \
        "${example_flags[@]}"
    fi
  done

  local plot_style_flags=()
  turnback_home_vector_alignment_plot_style_flags plot_style_flags
  run_cmd \
    python -m scripts.plot_overlay_training_metric_scalar_bars \
    --input "${TURNBACK_GROUP_LABELS[0]}=${bundles[0]}" \
    --input "${TURNBACK_GROUP_LABELS[1]}=${bundles[1]}" \
    --input "${TURNBACK_GROUP_LABELS[2]}=${bundles[2]}" \
    --out "exports/turnbackHomeVectorAlignment_${subset_slug}_${filter_tag}_${wall_tag}_flatLgc_T2_p${pair_label}_sliT2Sb2-5${estimator_suffix}${home_anchor_suffix}${sample_cross_suffix}${TURNBACK_PLOT_SUFFIX}_${DATE_TAG}.${TURNBACK_HOME_VECTOR_ALIGNMENT_IMG_FORMAT}" \
    --title "${subset_title}: home-vector heading alignment at re-entry, ${inner_radius_mm}/${outer_radius_mm} mm" \
    --ylabel "Home-vector heading alignment at re-entry" \
    --swarm \
    --stats \
    "${plot_style_flags[@]}"
}

run_turnback_home_vector_alignment_top20() {
  run_turnback_home_vector_alignment_sli_subset \
    top20 \
    "Top 20% SLI" \
    top \
    --top-sli-fraction \
    0.20 \
    "$@"
}

run_turnback_home_vector_alignment_bottom50() {
  run_turnback_home_vector_alignment_sli_subset \
    bottom50 \
    "Bottom 50% SLI" \
    bottom \
    --bottom-sli-fraction \
    0.50 \
    "$@"
}

plot_combined_turnback_home_vector_alignment() {
  local subset_slug="$1"
  local subset_title="$2"
  local filter_tag="$3"
  local wall_tag="$4"
  local estimator_suffix
  estimator_suffix="$(turnback_home_vector_alignment_heading_estimator_suffix)"
  local home_anchor_suffix
  home_anchor_suffix="$(turnback_home_vector_alignment_home_vector_anchor_suffix)"
  local sample_cross_suffix
  sample_cross_suffix="$(turnback_home_vector_alignment_sample_crossing_suffix)"
  local plot_prefix="turnbackHomeVectorAlignment_${filter_tag}_${wall_tag}_flatLgc_T2_p3-5_8-10_13-15_sb2-5"
  local title="Home-vector heading alignment at re-entry"
  if [[ "$subset_slug" != "all" ]]; then
    plot_prefix="turnbackHomeVectorAlignment_${subset_slug}_${filter_tag}_${wall_tag}_flatLgc_T2_p3-5_8-10_13-15_sliT2Sb2-5"
    title="${subset_title}: ${title}"
  fi

  local pair_labels=("3-5" "8-10" "13-15")
  local region_labels=("3/5 mm" "8/10 mm" "13/15 mm")
  local plot_style_flags=()
  turnback_home_vector_alignment_plot_style_flags plot_style_flags
  local plot_args=(
    python -m scripts.plot_overlay_training_metric_scalar_bars
    --out "exports/${plot_prefix}${estimator_suffix}${home_anchor_suffix}${sample_cross_suffix}${TURNBACK_PLOT_SUFFIX}_${DATE_TAG}.${TURNBACK_HOME_VECTOR_ALIGNMENT_IMG_FORMAT}"
    --title "$title"
    --xlabel "Inner/outer radius pair from reward center (mm)"
    --ylabel "Home-vector heading alignment at re-entry"
    --clustered-layout
    --swarm
    --stats
    "${plot_style_flags[@]}"
  )

  local pair_i group_i key bundle
  for pair_i in "${!pair_labels[@]}"; do
    for group_i in "${!TURNBACK_GROUP_LABELS[@]}"; do
      key="${subset_slug}|${pair_labels[$pair_i]}|${group_i}"
      bundle="${TURNBACK_HOME_VECTOR_ALIGNMENT_BUNDLES[$key]:-}"
      if [[ -z "$bundle" ]]; then
        echo "Missing recorded turnback home-vector alignment bundle for $key" >&2
        exit 1
      fi
      plot_args+=(
        --input "${region_labels[$pair_i]}|${TURNBACK_GROUP_LABELS[$group_i]}=${bundle}"
      )
    done
  done

  run_cmd "${plot_args[@]}"
}

run_return_leg_tortuosity_bins() {
  local bins_label="$1"
  local bins_arg="$2"
  local filter_tag="$3"
  local wall_tag="$4"
  local summary_tag="$5"
  local top_fraction="$6"
  local binning_mode="${7:-absolute_distance}"

  local filter_flags=()
  case "$filter_tag" in
    noFilt)
      filter_flags=(--min-between-reward-trajectories 0)
      ;;
    minEpFilt)
      filter_flags=()
      ;;
    minEpSb5Filt)
      filter_flags=(--require-exp-target-sync-bucket)
      ;;
    minEpPiFilt)
      filter_flags=(--require-exp-pi-threshold-bucket)
      ;;
    *)
      echo "Unknown return-leg tortuosity filter tag: $filter_tag" >&2
      exit 1
      ;;
  esac

  local wall_flags=()
  case "$wall_tag" in
    wall)
      wall_flags=()
      ;;
    noWall)
      wall_flags=(--return-leg-tortuosity-excursion-bin-exclude-wall-contact)
      ;;
    postWall)
      wall_flags=(
        --return-leg-tortuosity-excursion-bin-return-start-mode
        post_last_wall_max
      )
      ;;
    offWallFrames)
      wall_flags=(
        --return-leg-tortuosity-excursion-bin-exclude-wall-contact-frames
      )
      ;;
    *)
      echo "Unknown wall tag: $wall_tag" >&2
      exit 1
      ;;
  esac

  local bundles=()
  for i in "${!GROUP_VARS[@]}"; do
    local var_name="${GROUP_VARS[$i]}"
    local dataset="${!var_name}"
    local group_slug="${GROUP_SLUGS[$i]}"
    local group_label="${GROUP_LABELS[$i]}"
    local bundle="exports/returnLegTortuosityBins_${summary_tag}_${filter_tag}_${wall_tag}_${group_slug}_flatLgc_T2_b${bins_label}_${DATE_TAG}.npz"
    local binning_flags=()
    case "$binning_mode" in
      absolute_distance)
        binning_flags=(
          --return-leg-tortuosity-excursion-bin-radius-pairs-mm "$bins_arg"
          --return-leg-tortuosity-excursion-bin-top-fraction "$top_fraction"
        )
        ;;
      per_fly_quartile)
        binning_flags=(
          --return-leg-tortuosity-excursion-bin-binning-mode per_fly_quartile
          --return-leg-tortuosity-excursion-bin-top-fraction 1.0
        )
        ;;
      *)
        echo "Unknown return-leg tortuosity binning mode: $binning_mode" >&2
        exit 1
        ;;
    esac
    local example_flags=()
    if [[ "$RETURN_LEG_TORTUOSITY_EXAMPLES" == "1" && "$binning_mode" == "absolute_distance" ]]; then
      local example_dir="exports/returnLegTortuosityExamples_${summary_tag}_${filter_tag}_${wall_tag}_${group_slug}_flatLgc_T2_b${bins_label}_${DATE_TAG}"
      example_flags=(
        --export-return-leg-tortuosity-excursion-bin-examples "$example_dir"
        --return-leg-tortuosity-excursion-bin-examples-per-bin "$RETURN_LEG_TORTUOSITY_EXAMPLES_PER_BIN"
        --return-leg-tortuosity-excursion-bin-examples-max-per-fly "$RETURN_LEG_TORTUOSITY_EXAMPLES_MAX_PER_FLY"
      )
    fi

    bundles+=("$bundle")
    run_cmd \
      python analyze.py \
      -v "$dataset" \
      -f 0-1 \
      --rCC 15 \
      --export-return-leg-tortuosity-excursion-bin-sli-bundle "$bundle" \
      --return-leg-tortuosity-excursion-bin-trainings 2 \
      --return-leg-tortuosity-excursion-bin-skip-first-sync-buckets 1 \
      --best-worst-trn 2 \
      --sli-use-training-mean \
      --sli-select-skip-first-sync-buckets 1 \
      --export-group-label "$group_label" \
      "${binning_flags[@]}" \
      "${filter_flags[@]}" \
      "${wall_flags[@]}" \
      "${example_flags[@]}"
  done

  local bundle_csv
  bundle_csv="$(join_by_comma "${bundles[@]}")"
  run_cmd \
    python -m scripts.plot_return_leg_tortuosity_excursion_bin_sli_bundles \
    --bundles "$bundle_csv" \
    --out "exports/returnLegTortuosityBins_${summary_tag}_${filter_tag}_${wall_tag}_flatLgc_T2_b${bins_label}_${DATE_TAG}.png" \
    --stats
}

run_post_wall_departure_tortuosity() {
  local filter_tag="$1"
  local filter_flags=()
  case "$filter_tag" in
    noFilt)
      filter_flags=(--min-between-reward-trajectories 0)
      ;;
    minEpFilt)
      filter_flags=()
      ;;
    minEpSb5Filt)
      filter_flags=(--require-exp-target-sync-bucket)
      ;;
    minEpPiFilt)
      filter_flags=(--require-exp-pi-threshold-bucket)
      ;;
    *)
      echo "Unknown post-wall departure tortuosity filter tag: $filter_tag" >&2
      exit 1
      ;;
  esac

  local bundles=()
  for i in "${!GROUP_VARS[@]}"; do
    local var_name="${GROUP_VARS[$i]}"
    local dataset="${!var_name}"
    local group_slug="${GROUP_SLUGS[$i]}"
    local group_label="${GROUP_LABELS[$i]}"
    local bundle="exports/postWallDepartureTortuosity_${filter_tag}_${group_slug}_flatLgc_T2_${DATE_TAG}.npz"
    local example_flags=()
    if [[ "$POST_WALL_DEPARTURE_TORTUOSITY_EXAMPLES" == "1" ]]; then
      local example_dir="exports/postWallDepartureTortuosityExamples_${filter_tag}_${group_slug}_flatLgc_T2_${DATE_TAG}"
      example_flags=(
        --export-post-wall-departure-tortuosity-examples "$example_dir"
        --post-wall-departure-tortuosity-examples-num "$POST_WALL_DEPARTURE_TORTUOSITY_EXAMPLES_NUM"
        --post-wall-departure-tortuosity-examples-max-per-fly "$POST_WALL_DEPARTURE_TORTUOSITY_EXAMPLES_MAX_PER_FLY"
      )
    fi

    bundles+=("$bundle")
    run_cmd \
      python analyze.py \
      -v "$dataset" \
      -f 0-1 \
      --rCC 15 \
      --export-post-wall-departure-tortuosity-sli-bundle "$bundle" \
      --post-wall-departure-tortuosity-trainings 2 \
      --post-wall-departure-tortuosity-skip-first-sync-buckets 1 \
      --best-worst-trn 2 \
      --sli-use-training-mean \
      --sli-select-skip-first-sync-buckets 1 \
      --export-group-label "$group_label" \
      "${filter_flags[@]}" \
      "${example_flags[@]}"
  done

  local bundle_csv
  bundle_csv="$(join_by_comma "${bundles[@]}")"
  run_cmd \
    python -m scripts.plot_post_wall_departure_tortuosity_sli_bundles \
    --bundles "$bundle_csv" \
    --out "exports/postWallDepartureTortuosity_${filter_tag}_flatLgc_T2_${DATE_TAG}.png" \
    --stats
}

# ---------------------------------------------------------------------
# Fraction of trajectories within radius, historical return-prob name
# radial config: 6 / 11 / 16 mm
# ---------------------------------------------------------------------

# for filter_tag in noFilt minEpFilt minEpSb5Filt minEpPiFilt; do
#   for wall_tag in wall noWall; do
#     run_return_prob "$filter_tag" "$wall_tag"
#   done
# done

# ---------------------------------------------------------------------
# Turnback ratio
# radial config: 3/5, 8/10, 13/15 mm
# full filter x wall-contact matrix
# Each run writes the all-learner comparison plus one top-20% vs bottom-50%
# plot per cohort, ranked by mean T2 SLI over sync buckets 2-5.
# ---------------------------------------------------------------------

# for filter_tag in noFilt minEpFilt minEpSb5Filt minEpPiFilt; do
#   for wall_tag in wall noWall; do
#     run_turnback_pairs \
#       "3-5_8-10_13-15" \
#       "3:5,8:10,13:15" \
#       "$filter_tag" \
#       "$wall_tag"
#   done
# done

# ---------------------------------------------------------------------
# Turnback ratio
# radial config: 4/6, 9/11, 14/16 mm
# only 5-episode minimum + T2 SB5 presence filter
# ---------------------------------------------------------------------

# for wall_tag in wall noWall; do
#   run_turnback_pairs \
#     "4-6_9-11_14-16" \
#     "4:6,9:11,14:16" \
#     minEpSb5Filt \
#     "$wall_tag"
# done

# ---------------------------------------------------------------------
# Turnback ratio
# radial config: 2/5, 8/11, 14/17 mm
# only 5-episode minimum + T2 SB5 presence filter
# ---------------------------------------------------------------------

# for wall_tag in wall noWall; do
#   run_turnback_pairs \
#     "2-5_8-11_14-17" \
#     "2:5,8:11,14:17" \
#     minEpSb5Filt \
#     "$wall_tag"
# done

# ---------------------------------------------------------------------
# Dual-circle turnback ratio for the MBKC comparison.
# Matches the default 3/5, 8/10, 13/15 mm geometry and the active
# home-vector pass below: T2 SB5-presence filter, wall-contact included.
# ---------------------------------------------------------------------

if [[ "$RUN_TURNBACK_HOME_VECTOR_ALIGNMENT_ONLY" != "1" ]]; then
  run_turnback_pairs \
    "3-5_8-10_13-15" \
    "3:5,8:10,13:15" \
    minEpSb5Filt \
    wall
fi

# ---------------------------------------------------------------------
# Turnback home-vector heading alignment at successful re-entry
# Metric: cos(heading - home vector), using displacement-derived heading.
# Heading estimator: mean with radius 2, i.e. two-point symmetric sampling
# around the inner-circle border crossing.
# Default analysis window: Training 2, sync buckets 2 through 5.
# Episode geometry: absolute inner/outer radii from reward-circle center,
# matching turnback-ratio default pairs: 3/5, 8/10, 13/15 mm.
# wall: include all successful re-entry episodes
# noWall: discard successful re-entry episodes overlapping wall contact
# ---------------------------------------------------------------------

for filter_tag in minEpSb5Filt; do
  for wall_tag in wall; do
    run_turnback_home_vector_alignment "3-5" 3 5 "$filter_tag" "$wall_tag"
    run_turnback_home_vector_alignment "8-10" 8 10 "$filter_tag" "$wall_tag"
    run_turnback_home_vector_alignment "13-15" 13 15 "$filter_tag" "$wall_tag"
    plot_combined_turnback_home_vector_alignment all "All flies" "$filter_tag" "$wall_tag"
  done
done

# ---------------------------------------------------------------------
# Top-20% learner subset: turnback home-vector heading alignment
# SLI ranking: Training 2, sync buckets 2 through 5.
# Episode geometry: absolute inner/outer radii from reward-circle center,
# matching turnback-ratio default pairs: 3/5, 8/10, 13/15 mm.
# wall: include all successful re-entry episodes
# noWall: discard successful re-entry episodes overlapping wall contact
# ---------------------------------------------------------------------

for filter_tag in minEpSb5Filt; do
  for wall_tag in wall; do
    run_turnback_home_vector_alignment_top20 "3-5" 3 5 "$filter_tag" "$wall_tag"
    run_turnback_home_vector_alignment_top20 "8-10" 8 10 "$filter_tag" "$wall_tag"
    run_turnback_home_vector_alignment_top20 "13-15" 13 15 "$filter_tag" "$wall_tag"
    plot_combined_turnback_home_vector_alignment top20 "Top 20% SLI" "$filter_tag" "$wall_tag"
  done
done

# ---------------------------------------------------------------------
# Bottom-50% learner subset: turnback home-vector heading alignment
# SLI ranking: Training 2, sync buckets 2 through 5.
# Episode geometry: absolute inner/outer radii from reward-circle center,
# matching turnback-ratio default pairs: 3/5, 8/10, 13/15 mm.
# wall: include all successful re-entry episodes
# noWall: discard successful re-entry episodes overlapping wall contact
# ---------------------------------------------------------------------

for filter_tag in minEpSb5Filt; do
  for wall_tag in wall; do
    run_turnback_home_vector_alignment_bottom50 "3-5" 3 5 "$filter_tag" "$wall_tag"
    run_turnback_home_vector_alignment_bottom50 "8-10" 8 10 "$filter_tag" "$wall_tag"
    run_turnback_home_vector_alignment_bottom50 "13-15" 13 15 "$filter_tag" "$wall_tag"
    plot_combined_turnback_home_vector_alignment bottom50 "Bottom 50% SLI" "$filter_tag" "$wall_tag"
  done
done

if [[ "$RUN_TURNBACK_HOME_VECTOR_ALIGNMENT_ONLY" == "1" ]]; then
  exit 0
fi

# ---------------------------------------------------------------------
# Top-25% mean return-leg tortuosity by max-distance bin
# radial config: 3-5, 8-10, 13-15 mm from reward-circle center
# wall: standard full-episode maximum return-leg start
# noWall: discard episodes containing wall contact
# postWall: keep episodes and use the maximum after final wall contact
# offWallFrames: keep episodes but omit wall-contact steps from path distance
# only 5-episode minimum + T2 SB5 presence filter
# Set RETURN_LEG_TORTUOSITY_EXAMPLES=1 to export ranked trajectory galleries.
# ---------------------------------------------------------------------

run_return_leg_tortuosity_bins \
  "3-5_8-10_13-15_15-20_20-25_25-30" \
  "3:5,8:10,13:15,15:20,20:25,25:30" \
  noFilt \
  offWallFrames \
  all \
  1.0

# ---------------------------------------------------------------------
# Mean tortuosity after the final wall departure.
# Only between-reward trajectories containing wall contact contribute.
# Set POST_WALL_DEPARTURE_TORTUOSITY_EXAMPLES=1 for ranked path galleries.
# ---------------------------------------------------------------------

run_post_wall_departure_tortuosity minEpSb5Filt

# ---------------------------------------------------------------------
# Mean return-leg tortuosity by per-fly maximum-distance quartile
# Q1-Q4 are four equal-count bins formed independently for each fly.
# Top-fraction tortuosity aggregation is intentionally disabled.
# ---------------------------------------------------------------------

# for wall_tag in wall noWall postWall; do
#   run_return_leg_tortuosity_bins \
#     "quartiles" \
#     "" \
#     minEpSb5Filt \
#     "$wall_tag" \
#     mean \
#     1.0 \
#     per_fly_quartile
# done
