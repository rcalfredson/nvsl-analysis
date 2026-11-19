"""Replay flat-cohort exports using native selection and compare saved CSVs."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


def normalized_csv(path, aliases):
    with path.open(newline='') as stream:
        rows = list(csv.reader(stream))
    return [[aliases.get(cell, cell) for cell in row] for row in rows
            if not (row and row[0].startswith('# command:'))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    reference, output = args.reference.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {}
    for source in ('offline', 'original'):
        manifest = json.loads((reference / source / 'manifest.json').read_text())
        out = output / source
        (out / 'imgs').mkdir(parents=True)
        command = manifest['command'][:]
        command[:2] = [sys.executable, str(root / 'analyze.py')]
        command[command.index('-v') + 1] = ','.join(row['original_video'] for row in manifest['selection'])
        command[command.index('--cross-experiment-sli-export') + 1] = str(out / 'sli.csv')
        command += ['--tracking-variant', 'body-core' if source == 'offline' else 'original',
                    '--tracking-manifest', str(root / 'test/fixtures/flat-ar-control-body-core.tracking.json'),
                    '--no-tracking-fallback']
        (out / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
        env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(out / 'mpl-config'))
        with (out / 'process.log').open('w') as stream:
            subprocess.run(command, cwd=out, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        aliases = {row['video']: row['original_video'] for row in manifest['selection']}
        report[source] = {}
        for name in ('learning_stats.csv', 'sli.csv'):
            expected = normalized_csv(reference / source / name, aliases)
            actual = normalized_csv(out / name, aliases)
            if actual != expected:
                raise AssertionError(f'{source}/{name} differs; inspect {out}')
            report[source][name] = dict(exact_match=True, rows=len(actual))
        print(f'{source}: both exports match', flush=True)
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
