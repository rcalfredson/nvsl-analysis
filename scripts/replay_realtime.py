#!/usr/bin/env python3
"""CLI entry point for the all-fly tracking and live heatmap replay."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plotting.realtime_replay import main


if __name__ == "__main__":
    main()
