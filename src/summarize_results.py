"""Regenerate README evidence from committed errors, labels and frame index.

Run from repository root: python src/summarize_results.py
Uses standard library only. Pools the same head-excluded phases as Figure 2.
Does not run models, regenerate detections or re-evaluate synthetic trials.
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASES = ('takeoff', 'early', 'peak', 'rot1', 'rot2')
MODELS = ('vitpose', 'rtmpose', 'nsaqa')


def read(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def quantile(values, q):
    """Linear-interpolated quantile, matching NumPy's default percentile."""
    values = sorted(values)
    if not values:
        raise ValueError('No measured errors')
    position = (len(values)-1)*q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi]-values[lo])*(position-lo)


def torso_labelled(labels):
    return (any(labels.get(j) is not None for j in ('l_shoulder', 'r_shoulder'))
            and any(labels.get(j) is not None for j in ('l_hip', 'r_hip')))


def write(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    results = ROOT/'results'
    rows = read(results/'pose_error.csv') + read(results/'pose_error_nsaqa.csv')
    output = []
    for model in MODELS:
        selected = [r for r in rows if r['model']==model and r.get('row_type','joint')=='joint'
                    and r['joint']!='head' and r['phase'] in PHASES]
        if not selected:
            raise ValueError(f'Missing model: {model}')
        keys = [(r['image'], r['joint']) for r in selected]
        if len(keys)!=len(set(keys)):
            raise ValueError(f'Duplicate observations: {model}')
        values = [float(r['error_px']) for r in selected if r['error_px'].strip()]
        if any(not math.isfinite(v) or v<0 for v in values):
            raise ValueError(f'Invalid errors: {model}')
        gross = sum(r['gross_failure'].lower()=='true' for r in selected)
        missed = sum(r['missed'].lower()=='true' for r in selected)
        output.append(dict(model=model, median_px=quantile(values,.5), p90_px=quantile(values,.9),
                           gross_pct=100*gross/len(selected), missed_pct=100*missed/len(selected),
                           visible_joint_labels=len(selected), measured_joint_errors=len(values),
                           gross_count=gross, missed_count=missed, head_excluded=True,
                           phases=';'.join(PHASES)))
    write(results/'readme_results.csv', output)
    payload = json.loads((ROOT/'data/labels/hand_labels_round1.json').read_text())
    pngs = {k:v for k,v in payload['images'].items() if Path(k).suffix.lower()=='.png'}
    index = read(ROOT/'label_frames_index.csv')
    entry = [r for r in index if r['phase']=='entry' and Path(r['image']).suffix.lower()=='.png']
    if any(r['image'] not in pngs for r in entry):
        raise ValueError('Entry frame absent from hand labels')
    counts = [dict(measure='labelled_png_frames', count=len(pngs), source='data/labels/hand_labels_round1.json'),
              dict(measure='torso_labelled_frames', count=sum(torso_labelled(v) for v in pngs.values()), source='data/labels/hand_labels_round1.json'),
              dict(measure='entry_frames', count=len(entry), source='label_frames_index.csv'),
              dict(measure='entry_torso_labelled_frames', count=sum(torso_labelled(pngs[r['image']]) for r in entry), source='hand labels + frame index')]
    for model in MODELS:
        counts.append(dict(measure=f'{model}_analysed_frames',
                           count=len({r['image'] for r in rows if r['model']==model and r.get('row_type','joint')=='joint'}),
                           source='per-joint error CSVs'))
    write(results/'readme_counts.csv', counts)
    for row in output:
        print(f"{row['model']}: median={row['median_px']:.2f} px; p90={row['p90_px']:.2f} px; gross={row['gross_pct']:.2f}%; measured={row['measured_joint_errors']}")
    for row in counts:
        print(f"{row['measure']}: {row['count']}")


if __name__=='__main__':
    main()
