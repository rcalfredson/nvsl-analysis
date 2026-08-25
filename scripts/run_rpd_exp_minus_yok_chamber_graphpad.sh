#!/usr/bin/env bash
set -euo pipefail

VIDEO_LISTS_FILE="${VIDEO_LISTS_FILE:-video_lists.log}"
OUT_DIR="${OUT_DIR:-exports/rpd_exp_minus_yok_chambers}"
OUT_CSV="${OUT_CSV:-$OUT_DIR/rpd_exp_minus_yok_by_group_and_chamber.csv}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PRINT_ONLY="${PRINT_ONLY:-0}"
REUSE_EXISTING_NPZ="${REUSE_EXISTING_NPZ:-0}"
REFRESH_DATASETS="${REFRESH_DATASETS:-}"
WRITE_CSV="${WRITE_CSV:-1}"
LIST_DATASETS="${LIST_DATASETS:-0}"

RPD_TRAININGS="${RPD_TRAININGS:-2}"
RPD_SKIP_FIRST_SYNC_BUCKETS="${RPD_SKIP_FIRST_SYNC_BUCKETS:-1}"
RPD_KEEP_FIRST_SYNC_BUCKETS="${RPD_KEEP_FIRST_SYNC_BUCKETS:-4}"
RPD_POOLED_VALIDITY="${RPD_POOLED_VALIDITY:-window}"
RPD_POOLED_MIN_REWARDS="${RPD_POOLED_MIN_REWARDS:-5}"

DATASET_VARS=(
  RPD_FLAT_HTL_SENSORY_CTRL RPD_AGAROSE_HTL_SENSORY_CTRL
  RPD_FLAT_HTL_AR_CTRL RPD_AGAROSE_HTL_AR_CTRL
  RPD_FLAT_HTL_PFND_CTRL RPD_AGAROSE_HTL_PFND_CTRL
  RPD_FLAT_HTL_PFND RPD_AGAROSE_HTL_PFND
  RPD_FLAT_HTL_AR_KIR_CTRL RPD_AGAROSE_HTL_AR_KIR_CTRL
  RPD_FLAT_HTL_AR_PFND RPD_AGAROSE_HTL_AR_PFND
  RPD_FLAT_LARGE_CTRL RPD_AGAROSE_LARGE_CTRL
  RPD_FLAT_LARGE_AR_CTRL RPD_AGAROSE_LARGE_AR_CTRL
  RPD_FLAT_LARGE_PFND RPD_AGAROSE_LARGE_PFND
  RPD_FLAT_LARGE_AR_PFND RPD_AGAROSE_LARGE_AR_PFND
)
DATASET_SLUGS=(
  sensory_ctrl_flat_htl sensory_ctrl_agarose_htl
  ar_ctrl_flat_htl ar_ctrl_agarose_htl
  pfn_ctrl_flat_htl pfn_ctrl_agarose_htl
  pfnd_flat_htl pfnd_agarose_htl
  antenna_removed_ctrl_flat_htl antenna_removed_ctrl_agarose_htl
  antenna_removed_pfnd_flat_htl antenna_removed_pfnd_agarose_htl
  ctrl_flat_large ctrl_agarose_large
  ar_ctrl_flat_large ar_ctrl_agarose_large
  pfnd_flat_large pfnd_agarose_large
  ar_pfnd_flat_large ar_pfnd_agarose_large
)
LEGACY_BUNDLE_SLUGS=(
  "" ""
  ar_ctrlKir_flat_htl ar_ctrlKir_agarose_htl
  ctrlKir_flat_htl ctrlKir_agarose_htl
  pfnD_Kir_flat_htl pfnD_Kir_agarose_htl
  "" ""
  "" ""
  ctrlKir_flat_large ctrlKir_agarose_large
  ar_ctrlKir_flat_large ar_ctrlKir_agarose_large
  pfnD_Kir_flat_large pfnD_Kir_agarose_large
  "" ""
)
DATASET_LABELS=(
  "Antennae-intact control (AR-matched) | flat HTL" "Antennae-intact control (AR-matched) | agarose HTL"
  "Antennae-removed control | flat HTL" "Antennae-removed control | agarose HTL"
  "Ctrl>Kir (PFNd-matched) | flat HTL" "Ctrl>Kir (PFNd-matched) | agarose HTL"
  "PFNd>Kir | flat HTL" "PFNd>Kir | agarose HTL"
  "Antennae-removed Ctrl>Kir | flat HTL" "Antennae-removed Ctrl>Kir | agarose HTL"
  "Antennae-removed PFNd>Kir | flat HTL" "Antennae-removed PFNd>Kir | agarose HTL"
  "Ctrl>Kir | flat large" "Ctrl>Kir | agarose large"
  "Antennae removed Ctrl>Kir | flat large" "Antennae removed Ctrl>Kir | agarose large"
  "PFNd>Kir | flat large" "PFNd>Kir | agarose large"
  "Antennae removed PFNd>Kir | flat large" "Antennae removed PFNd>Kir | agarose large"
)
DATASET_HEADERS=(
  "Sight flies UAS-CSCV; +; 0273Gal4" "Sight flies UAS-CSCV; +; 0273Gal4"
  "Sight flies UAS-CSCV; +; 0273Gal4" "Sight flies UAS-CSCV; +; 0273Gal4"
  "Summary for PFNd (16D01)>Kir (normal and antenna glued) and 26E07>Kir-flat"
  "Summary for PFNd (16D01)>Kir (normal and antenna glued) and 26E07>Kir-flat"
  "Summary for PFNd (16D01)>Kir (normal and antenna glued) and 26E07>Kir-flat"
  "Summary for PFNd (16D01)>Kir (normal and antenna glued) and 26E07>Kir-flat"
  "Summary for antennae-removed Ctrl>Kir and PFNd>Kir HTL"
  "Summary for antennae-removed Ctrl>Kir and PFNd>Kir HTL"
  "Summary for antennae-removed Ctrl>Kir and PFNd>Kir HTL"
  "Summary for antennae-removed Ctrl>Kir and PFNd>Kir HTL"
  "Ctrl group: UAS>>CsC (X); ctrl-lexA/otd-flp; 0273Gal4/lexAop>>Kir"
  "Ctrl group: UAS>>CsC (X); ctrl-lexA/otd-flp; 0273Gal4/lexAop>>Kir"
  "Ctrl group: UAS>>CsC (X); ctrl-lexA/otd-flp; 0273Gal4/lexAop>>Kir"
  "Ctrl group: UAS>>CsC (X); ctrl-lexA/otd-flp; 0273Gal4/lexAop>>Kir"
  "UAS>>CsC (X); 16D01-lexA (PFNd)/otd-flp; 0273Gal4/lexAop>>Kir"
  "UAS>>CsC (X); 16D01-lexA (PFNd)/otd-flp; 0273Gal4/lexAop>>Kir"
  "UAS>>CsC (X); 16D01-lexA (PFNd)/otd-flp; 0273Gal4/lexAop>>Kir"
  "UAS>>CsC (X); 16D01-lexA (PFNd)/otd-flp; 0273Gal4/lexAop>>Kir"
)
DATASET_SUBHEADERS=(
  "Flat" "agarose"
  "Flat-Antenna removed" "Agarose-AR"
  "ctrl>Kir-flat 2023-02" "ctrl>Kir-agarose"
  "16D01>Kir-flat" "16D01>Kir-agarose"
  "Ctrl>Kir Flat-AR" "Ctrl>Kir Agarose-AR"
  "PFNd>Kir Flat-AR" "PFNd>Kir Agarose-AR"
  "Flat-lower chamber reward circle shrink in T2, T3, closer to the center  10d old flies"
  "Agarose-lower chamber reward circle shrink in T2, T3, closer to the center  10d old flies"
  "Flat AR-lower chamber reward circle shrink in T2, T3, closer to the center  10d old flies"
  "Agarose AR-lower chamber reward circle shrink in T2, T3, closer to the center  10d old flies"
  "Flat-lower chamber reward circle shrink in T2, T3, closer to the center  10d old flies"
  "Agarose-lower chamber reward circle shrink in T2, T3, closer to the center  10d old flies"
  "Flat AR-lower chamber reward circle shrink in T2, T3, closer to the center  10d old flies"
  "Agarose AR-lower chamber reward circle shrink in T2, T3, closer to the center  10d old flies"
)

if [[ "$PRINT_ONLY" != "0" && "$PRINT_ONLY" != "1" ]]; then
  echo "PRINT_ONLY must be 0 or 1." >&2
  exit 1
fi
if [[ "$REUSE_EXISTING_NPZ" != "0" && "$REUSE_EXISTING_NPZ" != "1" ]]; then
  echo "REUSE_EXISTING_NPZ must be 0 or 1." >&2
  exit 1
fi
if [[ "$WRITE_CSV" != "0" && "$WRITE_CSV" != "1" ]]; then
  echo "WRITE_CSV must be 0 or 1." >&2
  exit 1
fi
if [[ "$LIST_DATASETS" != "0" && "$LIST_DATASETS" != "1" ]]; then
  echo "LIST_DATASETS must be 0 or 1." >&2
  exit 1
fi
if [[ -z "$REFRESH_DATASETS" ]]; then
  if [[ "$REUSE_EXISTING_NPZ" == "1" ]]; then
    REFRESH_DATASETS="none"
  else
    REFRESH_DATASETS="all"
  fi
elif [[ "$REUSE_EXISTING_NPZ" == "1" ]]; then
  echo "REUSE_EXISTING_NPZ=1 conflicts with an explicit REFRESH_DATASETS value." >&2
  exit 1
fi
REFRESH_DATASETS="${REFRESH_DATASETS//[[:space:]]/}"

dataset_is_selected() {
  local slug="$1"
  if [[ "$REFRESH_DATASETS" == "all" ]]; then
    return 0
  fi
  if [[ "$REFRESH_DATASETS" == "none" ]]; then
    return 1
  fi
  case ",$REFRESH_DATASETS," in
    *,"$slug",*) return 0 ;;
    *) return 1 ;;
  esac
}

validate_refresh_datasets() {
  if [[ "$REFRESH_DATASETS" == "all" || "$REFRESH_DATASETS" == "none" ]]; then
    return
  fi
  local requested=()
  IFS=',' read -r -a requested <<< "$REFRESH_DATASETS"
  local wanted slug found
  for wanted in "${requested[@]}"; do
    found=0
    for slug in "${DATASET_SLUGS[@]}"; do
      if [[ "$wanted" == "$slug" ]]; then
        found=1
        break
      fi
    done
    if [[ "$found" != "1" ]]; then
      echo "Unknown REFRESH_DATASETS slug: $wanted" >&2
      echo "Run with LIST_DATASETS=1 to see valid slugs." >&2
      exit 1
    fi
  done
}

validate_refresh_datasets

video_list_from_log() {
  local header="$1"
  local subheader="$2"
  awk -v header="$header" -v subheader="$subheader" '
    $0 == header { in_section = 1; next }
    in_section && $0 == subheader { want_command = 1; next }
    want_command && /^python(2)? analyze\.py / {
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
}

if [[ "$LIST_DATASETS" == "1" ]]; then
  if [[ ! -f "$VIDEO_LISTS_FILE" ]]; then
    echo "Video-list file not found: $VIDEO_LISTS_FILE" >&2
    exit 1
  fi
  printf 'slug\tCSV column\tvideo_lists section\tvideo_lists cohort\toutput NPZ\t-v video list\n'
  for i in "${!DATASET_SLUGS[@]}"; do
    var_name="${DATASET_VARS[$i]}"
    videos="${!var_name:-}"
    if [[ -z "$videos" ]]; then
      videos="$(video_list_from_log "${DATASET_HEADERS[$i]}" "${DATASET_SUBHEADERS[$i]}")"
    fi
    if [[ -z "$videos" ]]; then
      echo "Could not find ${DATASET_LABELS[$i]} in $VIDEO_LISTS_FILE." >&2
      echo "Set $var_name explicitly to override the lookup." >&2
      exit 1
    fi
    printf '%s\t%s\t%s\t%s\t%s/%s.npz\t%s\n' "${DATASET_SLUGS[$i]}" "${DATASET_LABELS[$i]}" "${DATASET_HEADERS[$i]}" "${DATASET_SUBHEADERS[$i]}" "$OUT_DIR" "${DATASET_SLUGS[$i]}" "$videos"
  done
  exit 0
fi

resolve_datasets() {
  if [[ ! -f "$VIDEO_LISTS_FILE" ]]; then
    echo "Video-list file not found: $VIDEO_LISTS_FILE" >&2
    exit 1
  fi
  for i in "${!DATASET_VARS[@]}"; do
    if ! dataset_is_selected "${DATASET_SLUGS[$i]}"; then
      continue
    fi
    local var_name="${DATASET_VARS[$i]}"
    if [[ -n "${!var_name:-}" ]]; then
      continue
    fi
    local value
    value="$(video_list_from_log "${DATASET_HEADERS[$i]}" "${DATASET_SUBHEADERS[$i]}")"
    if [[ -z "$value" ]]; then
      echo "Could not find ${DATASET_LABELS[$i]} in $VIDEO_LISTS_FILE." >&2
      echo "Set $var_name explicitly to override the lookup." >&2
      exit 1
    fi
    printf -v "$var_name" '%s' "$value"
  done
}

run_cmd() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "$PRINT_ONLY" != "1" ]]; then
    "$@"
  fi
}

if [[ "$REFRESH_DATASETS" != "none" ]]; then
  resolve_datasets
fi
if [[ "$PRINT_ONLY" != "1" ]]; then
  mkdir -p "$OUT_DIR" "$(dirname "$OUT_CSV")"
fi

bundles=()
for i in "${!DATASET_VARS[@]}"; do
  canonical_bundle="$OUT_DIR/${DATASET_SLUGS[$i]}.npz"
  bundle="$canonical_bundle"
  legacy_slug="${LEGACY_BUNDLE_SLUGS[$i]}"
  if ! dataset_is_selected "${DATASET_SLUGS[$i]}" && [[ ! -f "$canonical_bundle" ]] && [[ -n "$legacy_slug" ]] && [[ -f "$OUT_DIR/$legacy_slug.npz" ]]; then
    bundle="$OUT_DIR/$legacy_slug.npz"
    if [[ "$WRITE_CSV" == "1" ]]; then
      printf '[rpd_dataset] reusing legacy NPZ for %s: %s\n' "${DATASET_SLUGS[$i]}" "$bundle"
    fi
  fi
  bundles+=("$bundle")
  if ! dataset_is_selected "${DATASET_SLUGS[$i]}"; then
    continue
  fi

  var_name="${DATASET_VARS[$i]}"
  videos="${!var_name}"
  printf '[rpd_dataset] %s | %s | source: %s > %s | output: %s\n' "${DATASET_SLUGS[$i]}" "${DATASET_LABELS[$i]}" "${DATASET_HEADERS[$i]}" "${DATASET_SUBHEADERS[$i]}" "$bundle"
  if [[ "${DATASET_SLUGS[$i]}" == *_large ]]; then
    fly_range="0-1"
    circle_args=(--rCC 15)
  else
    fly_range="0-9"
    circle_args=(--rmCC 5)
  fi
  run_cmd "$PYTHON_BIN" analyze.py \
    -v "$videos" -f "$fly_range" "${circle_args[@]}" \
    --rpd-total-export "$bundle" \
    --rpd-total-value-mode exp_minus_yok \
    --rpd-total-trainings "$RPD_TRAININGS" \
    --rpd-total-skip-first-sync-buckets "$RPD_SKIP_FIRST_SYNC_BUCKETS" \
    --rpd-total-keep-first-sync-buckets "$RPD_KEEP_FIRST_SYNC_BUCKETS" \
    --rpd-pooled-validity "$RPD_POOLED_VALIDITY" \
    --rpd-pooled-min-rewards "$RPD_POOLED_MIN_REWARDS"
done

if [[ "$WRITE_CSV" == "1" ]]; then
  if [[ "$PRINT_ONLY" != "1" ]]; then
    missing_bundles=()
    for i in "${!bundles[@]}"; do
      if [[ ! -f "${bundles[$i]}" ]]; then
        missing_bundles+=("${DATASET_SLUGS[$i]}")
      fi
    done
    if [[ "${#missing_bundles[@]}" -gt 0 ]]; then
      echo "Cannot write the complete CSV; missing NPZ dataset(s):" >&2
      printf '  %s\n' "${missing_bundles[@]}" >&2
      echo "Refresh those slugs or set WRITE_CSV=0 for an NPZ-only partial run." >&2
      exit 1
    fi
  fi

  csv_cmd=(
    "$PYTHON_BIN" scripts/export_graphpad_csv.py rpd-exp-minus-yok-npz
  )
  for i in "${!bundles[@]}"; do
    csv_cmd+=(--input "${DATASET_LABELS[$i]}=${bundles[$i]}")
  done
  csv_cmd+=(--panel 1 --out "$OUT_CSV")
  run_cmd "${csv_cmd[@]}"
fi
