"""Archive and plan sharing of manuscript data-panel sources."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def archive_data_stage(stage, root):
    """Preserve a stage's source, full statistics, and keyed-source provenance."""
    root = Path(root)
    source, archive = root / stage['archive_source'], root / stage['archive_output']
    if not source.is_file():
        raise FileNotFoundError(source)
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, archive)
    stats_output = stage.get('learning_stats_output')
    if stats_output:
        stats_source, stats_archive = root / 'learning_stats.csv', root / stats_output
        if not stats_source.is_file():
            raise FileNotFoundError(stats_source)
        stats_archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stats_source, stats_archive)
        metadata = {
            'generated_utc': datetime.now(timezone.utc).isoformat(),
            'analysis_command': stage['command'],
            'learning_stats': stats_output,
            'source_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'learning_stats_sha256': hashlib.sha256(stats_archive.read_bytes()).hexdigest(),
        }
        archive.with_suffix('.provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')


def data_panel_copy_paths(recipe, panel, root, target_dir):
    """Keep GraphPad at its panel path; share other artifacts in panel_data/."""
    import re

    match = re.fullmatch(r'(ED)?(\d+)[a-z]?(?:_[A-Za-z0-9]+)*', panel)
    if not match:
        raise ValueError(f'Invalid manuscript panel label: {panel}')
    root, target_dir = Path(root), Path(target_dir)
    figure_dir = f'edfig{match.group(2)}' if match.group(1) else f'fig{match.group(2)}'
    panel_name = panel[2:] if match.group(1) else panel
    primary = Path(recipe['panels'][panel]['output'])
    copies = [(root / primary, target_dir / figure_dir / (panel_name + primary.suffix))]
    artifacts = list(recipe.get('artifacts', []))
    for stage in recipe.get('analysis_stages', [recipe]):
        if stage.get('learning_stats_output'):
            artifacts.append(stage['learning_stats_output'])
        if stage.get('archive_output'):
            sidecar = Path(stage['archive_output']).with_suffix('.provenance.json')
            if (root / sidecar).is_file():
                artifacts.append(str(sidecar))
    seen = {primary}
    for artifact in artifacts:
        path = Path(artifact)
        if path in seen:
            continue
        seen.add(path)
        copies.append((root / path, target_dir / figure_dir / (panel_name + '_data') / path.name))
    return copies
