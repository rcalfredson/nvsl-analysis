#!/usr/bin/env bash
set -euo pipefail

DATE_TAG="2026-06-12"
PRINT_ONLY="${PRINT_ONLY:-0}"
RETURN_LEG_TORTUOSITY_EXAMPLES="${RETURN_LEG_TORTUOSITY_EXAMPLES:-0}"
RETURN_LEG_TORTUOSITY_EXAMPLES_PER_BIN="${RETURN_LEG_TORTUOSITY_EXAMPLES_PER_BIN:-6}"
RETURN_LEG_TORTUOSITY_EXAMPLES_MAX_PER_FLY="${RETURN_LEG_TORTUOSITY_EXAMPLES_MAX_PER_FLY:-0}"

GROUP_VARS=(INTACT_CTRL INTACT_PFND AR_CTRL)
GROUP_SLUGS=(intact_ctrlKir intact_pfnKir ar_ctrlKir)
GROUP_LABELS=("Ctrl>Kir FLC" "PFNd>Kir FLC" "AR Ctrl>Kir FLC")

for var_name in "${GROUP_VARS[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "Missing required dataset variable: $var_name" >&2
    echo "Define it with the corresponding video list before running this script." >&2
    exit 1
  fi
done

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

  for i in "${!GROUP_VARS[@]}"; do
    local var_name="${GROUP_VARS[$i]}"
    local dataset="${!var_name}"
    local group_slug="${GROUP_SLUGS[$i]}"
    local group_label="${GROUP_LABELS[$i]}"
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
      --export-group-label "$group_label" \
      "${filter_flags[@]}" \
      "${wall_flags[@]}"
  done

  local bundle_csv
  bundle_csv="$(join_by_comma "${bundles[@]}")"

  run_cmd \
    python -m scripts.plot_turnback_excursion_bin_sli_bundles \
    --bundles "$bundle_csv" \
    --out "exports/turnbackPairs_${filter_tag}_${wall_tag}_flatLgc_T2_p${pairs_label}_${DATE_TAG}.png" \
    --stats
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
# Top-25% mean return-leg tortuosity by max-distance bin
# radial config: 3-5, 8-10, 13-15 mm from reward-circle center
# wall: standard full-episode maximum return-leg start
# noWall: discard episodes containing wall contact
# postWall: keep episodes and use the maximum after final wall contact
# only 5-episode minimum + T2 SB5 presence filter
# Set RETURN_LEG_TORTUOSITY_EXAMPLES=1 to export ranked trajectory galleries.
# ---------------------------------------------------------------------

for wall_tag in wall noWall postWall; do
  run_return_leg_tortuosity_bins \
    "3-5_8-10_13-15" \
    "3:5,8:10,13:15" \
    minEpSb5Filt \
    "$wall_tag" \
    top25 \
    0.25
done

# ---------------------------------------------------------------------
# Mean return-leg tortuosity by per-fly maximum-distance quartile
# Q1-Q4 are four equal-count bins formed independently for each fly.
# Top-fraction tortuosity aggregation is intentionally disabled.
# ---------------------------------------------------------------------

for wall_tag in wall noWall postWall; do
  run_return_leg_tortuosity_bins \
    "quartiles" \
    "" \
    minEpSb5Filt \
    "$wall_tag" \
    mean \
    1.0 \
    per_fly_quartile
done
