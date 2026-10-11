"""Read the film-slide cohorts from their existing heatmap recipe."""
import json
import os
from pathlib import Path
import subprocess
import sys


def film_slide_video_lists(cohort):
    root = Path(__file__).resolve().parents[2]
    output = subprocess.check_output(
        ['bash', str(root / 'scripts/run_film_slide_paired_heatmaps.sh'),
         cohort, 'video-lists'], cwd=root, text=True,
        env={**os.environ, 'FILM_PYTHON': sys.executable},
    )
    return json.loads(output)
