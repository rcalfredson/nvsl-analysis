"""Run manuscript recipes with explicit, isolated trajectory sources."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from src.exporting.manuscript_data import archive_data_stage
from src.utils.tracking_selection import file_identity, select_tracking
from src.utils.util import AVI_X, fileList

VARIANT = 'nov20-two-rule'


def comparison_recipe_id(recipe):
    panels = '+'.join(recipe['panels'])
    return panels + '-data' if recipe.get('output_type') == 'data' else panels


def _tokens(command):
    return shlex.split(command) if isinstance(command, str) else list(command)


def freeze_analysis(command, excluded):
    """Resolve the notebook's grouped selectors, preserving fly overrides."""
    argv = _tokens(command)
    if len(argv) < 2 or argv[0] != 'python' or Path(argv[1]).name != 'analyze.py':
        raise ValueError(f'Expected an analyze.py recipe: {command}')
    index = argv.index('-v') + 1
    groups, recordings = [], []
    for group in argv[index].split('|'):
        entries = []
        for entry in group.split(','):
            pattern, colon, flies = entry.partition(':')
            paths = fileList(pattern, pattern=AVI_X)
            if not paths:
                raise ValueError(f'No recordings match {pattern!r}')
            for path in sorted(paths):
                video = str(Path(path).resolve(strict=True))
                if video in excluded:
                    continue
                entries.append(video + colon + flies)
                recordings.append(video)
        if not entries:
            raise ValueError(f'No recordings remain in group {group!r}')
        groups.append(','.join(entries))
    argv[index] = '|'.join(groups)
    return argv, list(dict.fromkeys(recordings))


def _identity(video, variant, manifest):
    p = Path(video)
    original, data = p.with_suffix('.trx'), p.with_suffix('.data')
    note = p.with_name(p.stem + '__note.txt')
    return dict(video=file_identity(p), original=file_identity(original),
                data=file_identity(data), note=file_identity(note) if note.is_file() else None,
                tracking=select_tracking(p, original, variant=variant, manifest=manifest))


def _analysis_argv(root, frozen, variant, manifest):
    if any(token.split('=', 1)[0] in (
        '--tracking-variant', '--tracking-manifest', '--tracking-fallback',
        '--no-tracking-fallback',
    ) for token in frozen):
        raise ValueError('Set tracking in the comparison configuration')
    argv = [sys.executable, str(root / 'analyze.py'), *frozen[2:], '--tracking-variant', variant]
    if variant != 'original':
        argv += ['--tracking-manifest', str(manifest), '--no-tracking-fallback']
    return argv


def _run_command(root, directory, argv, label):
    env = {**os.environ, 'MPLBACKEND': 'Agg',
           'PYTHONPATH': str(root) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    with (directory / (label + '.log')).open('w') as log:
        log.write(shlex.join(argv) + '\n')
        log.flush()
        subprocess.run(argv, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True)


def _python_argv(root, command):
    argv = _tokens(command)
    if not argv or argv[0] != 'python':
        raise ValueError(f'Expected a notebook Python command: {command}')
    argv[0] = sys.executable
    if argv[1].endswith('.py'):
        argv[1] = str(root / argv[1])
    return argv


def run_comparison(root, recipe, command, image_format, config, plot_style, *, run):
    """Preflight a recipe, then run each requested variant through its existing exporters.

    The optional wall_cache_dirs mapping points variants to existing read-only caches.
    """
    root = Path(root).resolve()
    variants = tuple(config.get('variants', ('original', VARIANT)))
    if not variants or len(set(variants)) != len(variants) or set(variants) - {'original', VARIANT}:
        raise ValueError(f'Invalid comparison variants: {variants}')
    excluded = {str(Path(p).resolve()) for p in config.get('excluded_recordings', ())}
    manifest = Path(config['manifest']).resolve() if config.get('manifest') else None
    data = recipe.get('output_type') == 'data'
    heatmap = recipe.get('plot_type') == 'heatmap'
    if data:
        upstream = recipe.get('analysis_stages', [recipe])
        render = recipe['export_command']
    else:
        upstream = recipe.get('export_commands', [])
        if recipe.get('export_command'):
            upstream = [*upstream, {'command': recipe['export_command']}]
        if Path(_tokens(command)[1]).name == 'analyze.py':
            upstream = [*upstream, {'command': command}]
            render = None
        else:
            render = command
    if not upstream:
        raise ValueError('The recipe has no upstream recording analysis')
    frozen = [freeze_analysis(stage['command'], excluded) for stage in upstream]
    videos = list(dict.fromkeys(video for _, paths in frozen for video in paths))
    identities = {variant: [_identity(v, variant, manifest) for v in videos] for variant in variants}
    base = Path(config['directory']).resolve() / comparison_recipe_id(recipe)
    existing_wall_caches = config.get('wall_cache_dirs', {})
    if set(existing_wall_caches) - set(variants):
        raise ValueError('Wall-cache directories must correspond to requested tracking variants')
    results = {}
    for variant in variants:
        argv = [_analysis_argv(root, tokens, variant, manifest) for tokens, _ in frozen]
        directory = base / variant
        if not heatmap:
            cache_dir = directory / 'wall-cache'
            cache_options = []
            if variant in existing_wall_caches:
                cache_dir = Path(existing_wall_caches[variant]).resolve(strict=True)
                if not cache_dir.is_dir():
                    raise NotADirectoryError(cache_dir)
                cache_options = ['--wall-cache-readonly']
            for args in argv:
                if any(token.split('=', 1)[0] == '--wall-cache-dir' for token in args):
                    raise ValueError('The comparison assigns its own wall-cache directory')
                args += ['--wall-cache-dir', str(cache_dir), *cache_options]
        results[variant] = dict(directory=directory, analysis=argv,
                                render=_python_argv(root, render) if render else None)
    if not run:
        return results

    def execute(variant):
        plan = results[variant]
        directory = plan['directory']
        directory.mkdir(parents=True, exist_ok=heatmap)
        (directory / 'imgs').mkdir(exist_ok=True)
        (directory / 'exports').mkdir(exist_ok=True)
        provenance = dict(
            recipe=recipe['title'], variant=variant, analysis=plan['analysis'],
            render=plan['render'], cwd=str(directory), inputs=identities[variant],
        )
        (directory / 'run.json').write_text(json.dumps(provenance, indent=2) + '\n')
        if heatmap:
            from src.plotting.heatmap_workflow import ensure_heatmap_cache

            return ensure_heatmap_cache(
                root, plan['analysis'][0], directory / 'probability_maps.npz',
                rebuild=config.get('rebuild_heatmaps', False),
                input_identities=identities[variant],
            )
        local = root / '.analyze.local.env'
        options = local.read_text() if local.is_file() else ''
        if recipe.get('requires_default_between_reward_sli_plots'):
            options = '\n'.join(line for line in options.splitlines()
                                if line.partition('=')[0].strip() != 'ENABLE_DEFAULT_BETWEEN_REWARD_SLI_PLOTS')
            options += '\nENABLE_DEFAULT_BETWEEN_REWARD_SLI_PLOTS=true\n'
        (directory / '.analyze.local.env').write_text(options)
        for i, (stage, argv) in enumerate(zip(upstream, plan['analysis'], strict=True), 1):
            _run_command(root, directory, argv, f'analysis-{i}')
            if data:
                archive_data_stage({**stage, 'command': shlex.join(argv)}, directory)
        if plan['render']:
            _run_command(root, directory, plan['render'], 'render' if not data else 'conversion')
        if data:
            for artifact in recipe['artifacts']:
                if not (directory / artifact).is_file():
                    raise FileNotFoundError(directory / artifact)
        else:
            for panel in recipe['panels'].values():
                source, dest = (directory / Path(panel[key]).with_suffix('.' + image_format)
                                for key in ('source', 'output'))
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)

    with ThreadPoolExecutor(max_workers=len(variants)) as executor:
        outputs = list(executor.map(execute, variants))
    if heatmap:
        for variant, cache in zip(variants, outputs, strict=True):
            plan = results[variant]
            _render_heatmap(cache, plan['directory'], recipe, plan['analysis'][0],
                            image_format, plot_style)
    return results


def _render_heatmap(cache, directory, recipe, argv, image_format, style):
    import matplotlib.pyplot as plt
    from src.plotting.heatmap_cache import render_heatmap_cache

    vmin = float(argv[argv.index('--pltHmVmin') + 1])
    vmax = float(argv[argv.index('--pltHmVmax') + 1])
    figure = render_heatmap_cache(cache, vmin, vmax, font_family=style['font_family'],
                                  font_size=style['font_size_heatmap'])
    try:
        for panel in recipe['panels'].values():
            filename = Path(panel['output']).with_suffix('.' + image_format)
            tag = directory.name
            dest = directory / f'{filename.stem}_{tag}{filename.suffix}'
            figure.savefig(dest, bbox_inches='tight', format=image_format)
    finally:
        plt.close(figure)
