#!/usr/bin/env bash
set -euo pipefail

# Reproduce the heatmap PDFs used by the experiments listed below.
#
# Run every recipe:
#   scripts/generate_heatmaps.sh
#
# Run one or more recipes:
#   scripts/generate_heatmaps.sh flat_htl_training epg_post
#
# Override the date in artifact names or inspect commands without running them:
#   DATE_TAG=2026-09-04 scripts/generate_heatmaps.sh
#   PRINT_ONLY=1 scripts/generate_heatmaps.sh

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATE_TAG="${DATE_TAG:-$(date +%F)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PRINT_ONLY="${PRINT_ONLY:-0}"
SOURCE_PDF="imgs/heatmaps2.pdf"

RECIPES=(
  flat_htl_training
  agarose_htl_training
  flat_htl_pre
  agarose_htl_pre
  ctrl_kir_training
  ctrl_kir_post
  ar_training
  ar_post
  pfnd_kir_training
  pfnd_kir_post
  mbkc_kir_training
  mbkc_kir_post
  epg_kir_training
  epg_kir_post
)

FLAT_HTL_VIDEOS='/media/Synology4/Yang Chen/2024-03-04/c3[12]_*,/media/Synology4/Yang Chen/2024-03-04/c4[12]_*,/media/Synology4/Yang Chen/2024-03-14/c4[12]_*,/media/Synology4/Yang Chen/2024-03-18/c5[12]_*,/media/Synology4/Yang Chen/2024-03-18/c6[12]_*,/media/Synology4/Yang Chen/2024-06-12/c[12]_*'
AGAROSE_HTL_VIDEOS='/media/Synology4/Yang Chen/2024-03-1[34]/c5[12]_*,/media/Synology4/Yang Chen/2024-03-1[347]/c6[12]_*,/media/Synology4/Yang Chen/2024-03-18/c[12]_*'
CTRL_KIR_VIDEOS='/media/Synology4/Yang Chen/2025-05-30/afternoon/c5[12]_*,/media/Synology4/Yang Chen/2025-05-30/afternoon/c6[12]_*,/media/Synology4/Yang Chen/2025-06-02/c5[12]_*,/media/Synology4/Yang Chen/2025-06-29/afternoon/c5[12]_*,/media/Synology4/Yang Chen/2025-06-30/afternoon/c5[12]_*,/media/Synology4/Yang Chen/2025-06-30/afternoon/c6[12]_*,/media/Synology4/Yang Chen/2025-07-0[367]/afternoon/c5[12]_*,/media/Synology4/Yang Chen/2025-07-0[367]/afternoon/c6[12]_*,/media/Synology4/Yang Chen/2025-07-0[45]/c5[12]_*,/media/Synology4/Yang Chen/2025-07-05/c6[12]_*,/media/Synology4/Yang Chen/2025-07-11/c5[12]_*,/media/Synology4/Yang Chen/2025-07-11/c6[12]_*,/media/Synology4/Yang Chen/2025-07-26/afternoon/c3[12]_*,/media/Synology4/Yang Chen/2025-07-26/afternoon/c6[12]_*'
AR_VIDEOS='/media/Synology4/Yang Chen/2025-07-17/c4[12]_*,/media/Synology4/Yang Chen/2025-07-17/c5[12]_*,/media/Synology4/Yang Chen/2025-07-17/c6[12]_*,/media/Synology4/Yang Chen/2025-07-18/night/c5[12]_*,/media/Synology4/Yang Chen/2025-07-18/night/c6[12]_*,/media/Synology4/Yang Chen/2025-07-2[09]/night/c3[12]_*,/media/Synology4/Yang Chen/2025-07-2[01]/night/c4[12]_*,/media/Synology4/Yang Chen/2025-07-21/night/c6[12]_*,/media/Synology4/Yang Chen/2025-07-27/c6[12]_*,/media/Synology4/Yang Chen/2025-07-27/c5[12]_*,/media/Synology4/Yang Chen/2025-07-27/afternoon/c3[12]_*,/media/Synology4/Yang Chen/2025-07-27/afternoon/c4[12]_*,/media/Synology4/Yang Chen/2025-07-2[79]/night/c4[12]_*,/media/Synology4/Yang Chen/2025-07-30/night/c3[12]_*,/media/Synology4/Yang Chen/2025-07-30/night/c4[12]_*,/media/Synology4/Yang Chen/2025-08-02/night/c3[12]_*'
PFND_KIR_VIDEOS='/media/Synology4/Yang Chen/2025-05-31/c5[12]_*,/media/Synology4/Yang Chen/2025-05-31/c6[12]_*,/media/Synology4/Yang Chen/2025-05-31/afternoon/c5[12]_*,/media/Synology4/Yang Chen/2025-06-05/c6[12]_*,/media/Synology4/Yang Chen/2025-06-28/afternoon/c5[12]_*,/media/Synology4/Yang Chen/2025-06-29/afternoon/c6[12]_*,/media/Synology4/Yang Chen/2025-07-0[124]/afternoon/c5[12]_*,/media/Synology4/Yang Chen/2025-07-0[124]/afternoon/c6[12]_*,/media/Synology4/Yang Chen/2025-07-06/c6[12]_*,/media/Synology4/Yang Chen/2025-07-12/night/c32_*,/media/Synology4/Yang Chen/2025-07-21/night/c5[12]_*,/media/Synology4/Yang Chen/2025-07-22/night/c3[12]_*'
MBKC_KIR_VIDEOS='/media/Synology4/Yang Chen/2025-06-0[17]/c5[12]_*,/media/Synology4/Yang Chen/2025-06-01/c6[12]_*,/media/Synology4/Yang Chen/2025-07-0[36]/c5[12]_*,/media/Synology4/Yang Chen/2025-07-0[34]/c6[12]_*,/media/Synology4/Yang Chen/2025-07-05/afternoon/c5[12]_*,/media/Synology4/Yang Chen/2025-07-05/afternoon/c6[12]_*,/media/Synology4/Yang Chen/2025-07-11/c3[12]_*,/media/Synology4/Yang Chen/2025-07-11/c4[12]_*,/media/Synology4/Yang Chen/2025-07-11/afternoon/c3[12]_*,/media/Synology4/Yang Chen/2025-07-26/c3[12]_*,/media/Synology4/Yang Chen/2025-07-26/c4[12]_*,/media/Synology4/Yang Chen/2025-07-30/c3[12]_*,/media/Synology4/Yang Chen/2025-07-30/c4[12]_*,/media/Synology4/Yang Chen/2025-07-30/c5[12]_*,/media/Synology4/Yang Chen/2025-07-30/afternoon/c5[12]_*'
EPG_KIR_VIDEOS='/media/Synology4/Yang Chen/2025-07-0[7]/c52_*,/media/Synology4/Yang Chen/2025-07-0[79]/c6[12]_*,/media/Synology4/Yang Chen/2025-07-0[9]/c5[12]_*,/media/Synology4/Yang Chen/2025-07-10/c6[12]_*,/media/Synology4/Yang Chen/2025-07-10/c5[12]_*,/media/Synology4/Yang Chen/2025-07-12/c4[12]_*,/media/Synology4/Yang Chen/2025-07-1[4]/night/c3[12]_*,/media/Synology4/Yang Chen/2025-07-1[4]/night/c4[12]_*,/media/Synology4/Yang Chen/2025-07-21/night/c6[12]_*,/media/Synology4/Yang Chen/2025-07-29/c3[12]_*,/media/Synology4/Yang Chen/2025-07-29/c4[12]_*,/media/Synology4/Yang Chen/2025-08-02/night/c4[12]_*,/media/Synology4/Yang Chen/2025-08-03/c5[12]_*,/media/Synology4/Yang Chen/2025-08-03/c6[12]_*'

usage() {
  printf 'Usage: %s [recipe ...]\n\nAvailable recipes:\n' "$0"
  printf '  %s\n' "${RECIPES[@]}"
}

print_command() {
  printf '  %q' "$@"
  printf '\n'
}

run_heatmap() {
  local recipe="$1"
  local artifact_stem="$2"
  local videos="$3"
  shift 3

  local target_pdf="imgs/${artifact_stem}_${DATE_TAG}.pdf"
  local command=("$PYTHON_BIN" analyze.py -v "$videos" "$@")

  printf '\n[%s]\n' "$recipe"
  print_command "${command[@]}"

  if [[ "$PRINT_ONLY" == "1" ]]; then
    print_command mv -- "$SOURCE_PDF" "$target_pdf"
    return
  fi

  "${command[@]}"
  if [[ ! -f "$SOURCE_PDF" ]]; then
    printf 'Expected heatmap was not created: %s\n' "$SOURCE_PDF" >&2
    exit 1
  fi
  mv -- "$SOURCE_PDF" "$target_pdf"
  printf 'Saved %s\n' "$target_pdf"
}

run_recipe() {
  case "$1" in
    flat_htl_training)
      run_heatmap "$1" heatmaps2_t2sb5_flat_htl "$FLAT_HTL_VIDEOS" \
        -f 0-9 --rmCC 5 --pltHm --num-trainings 2 --hm-sync-bucket 5 \
        --hm-periods training --imgFormat pdf --fs 20 \
        --pltHmVmin 1e-6 --pltHmVmax 1e-3
      ;;
    agarose_htl_training)
      run_heatmap "$1" heatmaps2_t2sb5_agarose_htl "$AGAROSE_HTL_VIDEOS" \
        -f 0-9 --rmCC 5 --pltHm --num-trainings 2 --hm-sync-bucket 5 \
        --hm-periods training --imgFormat pdf --fs 20 \
        --pltHmVmin 1e-6 --pltHmVmax 1e-3
      ;;
    flat_htl_pre)
      run_heatmap "$1" heatmaps2_pre_flat_htl "$FLAT_HTL_VIDEOS" \
        -f 0-9 --rmCC 5 --pltHm --hm-periods pre --hm-pre-minutes 10 \
        --pltHmVmin 1e-6 --pltHmVmax 1e-3 --imgFormat pdf --fs 20
      ;;
    agarose_htl_pre)
      run_heatmap "$1" heatmaps2_pre_agarose_htl "$AGAROSE_HTL_VIDEOS" \
        -f 0-9 --rmCC 5 --pltHm --hm-periods pre --hm-pre-minutes 10 \
        --pltHmVmin 1e-6 --pltHmVmax 1e-3 --imgFormat pdf --fs 20
      ;;
    ctrl_kir_training)
      run_heatmap "$1" fig4_hm_ctrlKir_flatLgc_T2 "$CTRL_KIR_VIDEOS" \
        -f 0-1 --rCC 15 --pltHm --hm-periods training --num-trainings 2 \
        --pltHmVmin 1e-6 --pltHmVmax 1e-3 --imgFormat pdf --fs 20
      ;;
    ctrl_kir_post)
      run_heatmap "$1" fig4_hm_ctrlKir_flatLgc_T2Post "$CTRL_KIR_VIDEOS" \
        -f 0-1 --rCC 15 --pltHm --hm-periods post --num-trainings 2 --rpib 3 \
        --pltHmVmin 1e-6 --pltHmVmax 1e-3 --imgFormat pdf --fs 20
      ;;
    ar_training)
      run_heatmap "$1" figExt15_hm_ar_flatLgc_T2 "$AR_VIDEOS" \
        -f 0-1 --rCC 15 --pltHm --hm-periods training --num-trainings 2 \
        --pltHmVmin 1e-6 --pltHmVmax 1e-3 --imgFormat pdf --fs 20
      ;;
    ar_post)
      run_heatmap "$1" figExt15_hm_ar_flatLgc_T2Post "$AR_VIDEOS" \
        -f 0-1 --rCC 15 --pltHm --hm-periods post --num-trainings 2 --rpib 3 \
        --pltHmVmin 1e-6 --pltHmVmax 1e-3 --imgFormat pdf --fs 20
      ;;
    pfnd_kir_training)
      run_heatmap "$1" figExt15_hm_PFNdKir_flatLgc_T2 "$PFND_KIR_VIDEOS" \
        -f 0-1 --rCC 15 --pltHm --hm-periods training --num-trainings 2 \
        --pltHmVmin 1e-6 --pltHmVmax 1e-3 --imgFormat pdf --fs 20
      ;;
    pfnd_kir_post)
      run_heatmap "$1" figExt15_hm_PFNdKir_flatLgc_T2Post "$PFND_KIR_VIDEOS" \
        -f 0-1 --rCC 15 --pltHm --hm-periods post --num-trainings 2 --rpib 3 \
        --pltHmVmin 1e-6 --pltHmVmax 1e-3 --imgFormat pdf --fs 20
      ;;
    mbkc_kir_training)
      run_heatmap "$1" figExt15_hm_mbkcKir_flatLgc_T2 "$MBKC_KIR_VIDEOS" \
        -f 0-1 --rCC 15 --pltHm --hm-periods training --num-trainings 2 \
        --pltHmVmin 1e-6 --pltHmVmax 1e-3 --imgFormat pdf --fs 20
      ;;
    mbkc_kir_post)
      run_heatmap "$1" figExt15_hm_mbkcKir_flatLgc_T2Post "$MBKC_KIR_VIDEOS" \
        -f 0-1 --rCC 15 --pltHm --hm-periods post --num-trainings 2 --rpib 3 \
        --pltHmVmin 1e-6 --pltHmVmax 1e-3 --imgFormat pdf --fs 20
      ;;
    epg_kir_training)
      run_heatmap "$1" figExt15_hm_epgKir_flatLgc_T2 "$EPG_KIR_VIDEOS" \
        -f 0-1 --rCC 15 --pltHm --hm-periods training --num-trainings 2 \
        --pltHmVmin 1e-6 --pltHmVmax 1e-3 --imgFormat pdf --fs 20
      ;;
    epg_kir_post)
      run_heatmap "$1" figExt15_hm_epgKir_flatLgc_T2Post "$EPG_KIR_VIDEOS" \
        -f 0-1 --rCC 15 --pltHm --hm-periods post --num-trainings 2 --rpib 3 \
        --pltHmVmin 1e-6 --pltHmVmax 1e-3 --imgFormat pdf --fs 20
      ;;
    *)
      printf 'Unknown recipe: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
}

if [[ "$PRINT_ONLY" != "0" && "$PRINT_ONLY" != "1" ]]; then
  printf 'PRINT_ONLY must be 0 or 1.\n' >&2
  exit 2
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p imgs

if (( $# == 0 )); then
  set -- "${RECIPES[@]}"
fi

for recipe in "$@"; do
  run_recipe "$recipe"
done
