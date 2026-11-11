"""Conference figures from CSV sources; no synthetic or imputed measurements.

Run: python src/make_figures.py --results results
Write three 200-dpi PNGs and their plotted-data CSVs to results/figures/.
Figure 2 pools finite per-joint errors after excluding head. Misses cannot
enter pixel-error medians; gross finite errors remain included. Human
self-disagreement and model-vs-label error are distinct measurements.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'physics_residual_aqa_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

MODELS = ('vitpose', 'rtmpose', 'nsaqa')
NAMES = {'vitpose': 'ViTPose', 'rtmpose': 'RTMPose', 'nsaqa': 'NSAQA'}
# Okabe–Ito colours; human uses a distinct bluish green.
COLOURS = {'vitpose': '#0072B2', 'rtmpose': '#E69F00', 'nsaqa': '#CC79A7', 'human': '#009E73'}
PHASES = ('takeoff', 'early', 'peak', 'rot1', 'rot2')
SIGMAS = (1, 2, 3, 5, 8, 10)
BREAKS = (5, 10, 20, 40)
REQUIRED_PX = 3.0  # User-specified requirement; not a fitted threshold.
DPI = 200


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    """Fail explicitly on missing sources or required columns."""
    if not path.is_file():
        raise FileNotFoundError(f'Required CSV missing: {path}')
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path}: empty CSV')
    return rows


def number(value: str, context: str) -> float:
    """Parse a finite nonnegative measurement without filling missing values."""
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f'Invalid measurement in {context}: {value}')
    return result


def load_data(root: Path) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Validate all data before creating any figures; do not average medians."""
    paths = [root / name for name in (
        'detection_threshold.csv', 'pose_error.csv', 'pose_error_nsaqa.csv',
        'human_consistency.csv', 'model_medians.csv')]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Required CSV missing: ' + ', '.join(missing))
    threshold = read_csv(paths[0], {'break_size_percent', *(f'sigma_{s}_px' for s in SIGMAS)})
    by_break = {}
    for row in threshold:
        size = number(row['break_size_percent'], str(paths[0]))
        if size in by_break:
            raise ValueError(f'Duplicate break size: {size}')
        by_break[size] = row
    if set(by_break) != set(BREAKS):
        raise ValueError('Expected break sizes 5, 10, 20, 40 percent')
    heatmap = np.array([[number(by_break[b][f'sigma_{s}_px'], 'detection rate')
                         for s in SIGMAS] for b in BREAKS])
    if np.any(heatmap > 100):
        raise ValueError('Detection-rate cells must be percentages in [0,100]')

    rows = []
    for path in paths[1:3]:
        rows.extend(read_csv(path, {'model', 'image', 'joint', 'phase', 'error_px'}))
    seen = set()
    groups = {(model, phase): [] for model in MODELS for phase in PHASES}
    for row in rows:
        if row.get('row_type', 'joint') != 'joint' or row['model'] not in MODELS:
            continue
        key = (row['model'], row['image'], row['joint'])
        if key in seen:
            raise ValueError(f'Duplicate per-joint observation: {key}')
        seen.add(key)
        if row['joint'] == 'head' or row['phase'] not in PHASES:
            continue
        if row['error_px'].strip() == '':
            continue  # Missing predictions have no finite pixel-error measurement.
        groups[(row['model'], row['phase'])].append(number(row['error_px'], str(key)))
    phase_data = []
    for phase in PHASES:
        for model in MODELS:
            values = groups[(model, phase)]
            if not values:
                raise ValueError(f'Missing head-excluded errors: {model}/{phase}')
            phase_data.append(dict(model=model, phase=phase, n_joint_errors=len(values),
                                   median_px=float(np.median(values))))

    humans = read_csv(paths[3], {'measure', 'median_px', 'paired_points', 'frames'})
    overall = [row for row in humans if row['measure'] == 'overall']
    if len(overall) != 1:
        raise ValueError('Expected exactly one overall human-consistency row')
    model_rows = read_csv(paths[4], {'model', 'median_px', 'head_excluded', 'source'})
    if len(model_rows) != 3 or {r['model'] for r in model_rows} != set(MODELS):
        raise ValueError('Expected exactly one supplied overall median per model')
    limit_data = []
    for model in MODELS:
        row = next(r for r in model_rows if r['model'] == model)
        if row['head_excluded'].lower() != 'true':
            raise ValueError(f'{model}: Figure 3 requires head-excluded median')
        limit_data.append(dict(model=model, median_px=number(row['median_px'], model), source=row['source']))
    human = overall[0]
    limit_data.append(dict(model='human', median_px=number(human['median_px'], 'human overall'),
                           source=f"independent re-annotation: {human['paired_points']} paired points, {human['frames']} frames"))
    return heatmap, phase_data, limit_data


def style() -> None:
    """Large slide fonts, clean axes, and consistent 16:9 canvases."""
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 23, 'axes.labelsize': 26,
        'xtick.labelsize': 23, 'ytick.labelsize': 23, 'legend.fontsize': 22,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 1.1, 'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })


def save(fig: plt.Figure, path: Path) -> None:
    """Keep a 2560x1440 canvas, at 200 dpi, without embedded titles."""
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def figure_requirement(data: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.2), layout='constrained')
    im = ax.imshow(data, cmap='cividis', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(SIGMAS)), [str(s) for s in SIGMAS])
    ax.set_yticks(range(len(BREAKS)), [f'{b}%' for b in BREAKS])
    ax.set_xlabel('Keypoint noise σ (px)', labelpad=15)
    ax.set_ylabel('Angular-momentum break', labelpad=15)
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            colour = 'white' if value < 55 else '#101010'
            ax.text(j, i, f'{value:g}%', ha='center', va='center',
                    color=colour, fontsize=24, fontweight='bold')
    # Highlight the requested 3 px column without implying every break is detectable.
    ax.add_patch(Rectangle((SIGMAS.index(3)-.5, -.5), 1, len(BREAKS),
                           fill=False, edgecolor='#D55E00', linewidth=4, clip_on=False))
    ax.get_xticklabels()[SIGMAS.index(3)].set_fontweight('bold')
    colourbar = fig.colorbar(im, ax=ax, pad=.03, fraction=.05, ticks=[0, 50, 100])
    colourbar.set_label('Detection rate (%)', labelpad=12)
    save(fig, output)


def figure_reality(rows: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.2), layout='constrained')
    x = np.arange(len(PHASES))
    width = .24
    maximum = max(r['median_px'] for r in rows)
    for index, model in enumerate(MODELS):
        values = [next(r['median_px'] for r in rows if r['model']==model and r['phase']==p) for p in PHASES]
        ax.bar(x + (index-1)*width, values, width=width, color=COLOURS[model], label=NAMES[model])
    ax.axhline(REQUIRED_PX, color='#333333', linestyle='--', linewidth=2)
    ax.text(.99, REQUIRED_PX+.45, 'required precision', transform=ax.get_yaxis_transform(),
            ha='right', va='bottom', fontsize=21,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9, 'pad': 2})
    ax.set_xticks(x, PHASES)
    ax.set_xlabel('Dive phase', labelpad=12)
    ax.set_ylabel('Median pixel error\n(head excluded)', labelpad=12)
    ax.set_ylim(0, maximum*1.23)
    ax.set_yticks(np.arange(0, maximum*1.23, 5))
    ax.legend(loc='upper left', ncols=3, frameon=False, handlelength=1.2, columnspacing=1.1)
    ax.spines['bottom'].set_color('#777777')
    save(fig, output)


def figure_limit(rows: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.2), layout='constrained')
    labels = [NAMES.get(r['model'], 'Human re-annotation') for r in rows]
    values = [r['median_px'] for r in rows]
    bars = ax.barh(range(4), values, height=.60, color=[COLOURS[r['model']] for r in rows])
    ax.set_yticks(range(4), labels)
    ax.invert_yaxis()
    ax.set_ylim(3.6, -.8)  # Keep the requirement label above the top bar.
    ax.set_xlim(0, max(values)*1.2)
    ax.set_xlabel('Median pixel discrepancy (px)', labelpad=15)
    ax.axvline(REQUIRED_PX, color='#333333', linestyle='--', linewidth=2)
    ax.text(REQUIRED_PX+.12, .98, 'required precision', transform=ax.get_xaxis_transform(),
            ha='left', va='top', fontsize=20)
    for bar, value in zip(bars, values):
        ax.text(value+.12, bar.get_y()+bar.get_height()/2, f'{value:g}', va='center', fontsize=25)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    save(fig, output)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[1]/'results')
    args = parser.parse_args()
    heatmap, phases, limits = load_data(args.results)
    style()
    out = args.results/'figures'
    out.mkdir(parents=True, exist_ok=True)
    figure_requirement(heatmap, out/'figure_1_requirement.png')
    figure_reality(phases, out/'figure_2_reality.png')
    figure_limit(limits, out/'figure_3_limit.png')
    write_csv(out/'figure_2_phase_medians.csv', phases)
    write_csv(out/'figure_3_medians.csv', limits)
    write_csv(out/'figure_1_detection_rates.csv', [dict(break_size_percent=b,
        **{f'sigma_{s}_px': float(heatmap[i,j]) for j,s in enumerate(SIGMAS)}) for i,b in enumerate(BREAKS)])
    for path in sorted(out.glob('figure_*.png')):
        print(path)


if __name__ == '__main__':
    main()
