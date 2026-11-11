# Key results

All numbers measured, not estimated. Sources are in `results/`.

## The requirement (synthetic detection study)

Detection rate for an angular-momentum break, by break size and keypoint
noise, on a 100 px torso. 100 seeded trials per cell. Known-frame detection
against matched no-break controls, so these are an upper bound on what a
whole-sequence search would achieve.

| break | sigma=1 | 2 | 3 | 5 | 8 | 10 |
|---|---|---|---|---|---|---|
| 5%  | 84%  | 30%  | 15%  | 10%  | 4%  | 4%  |
| 10% | 100% | 85%  | 46%  | 17%  | 11% | 10% |
| 20% | 100% | 100% | 98%  | 57%  | 24% | 15% |
| 40% | 100% | 100% | 100% | 100% | 70% | 51% |

Required precision for reliable detection: about 3 px.

## The reality (81 hand-labelled broadcast frames)

Median Euclidean keypoint error, head excluded.

| model | median px | p90 px | gross % | missed % |
|---|---|---|---|---|
| ViTPose (generic) | 10.08 | 78.48 | 13.79 | 7.71 |
| RTMPose (generic) | 10.41 | 66.16 | 12.17 | 7.71 |
| NSAQA (diving-finetuned) | 9.36 | 33.72 | 8.11 | 7.71 |

Domain finetuning cuts the tail (p90 down 57%, gross failures down 41%) but
barely moves the median (down 7%).

## Error by phase (median px)

| phase | ViTPose | RTMPose | NSAQA |
|---|---|---|---|
| takeoff | 9.50 | 8.94 | 8.30 |
| early | 9.62 | 9.94 | 8.06 |
| peak | 13.22 | 12.12 | 9.19 |
| rot1 | 15.08 | 15.33 | 11.06 |
| rot2 | 17.59 | 20.67 | 18.44 |

Gross-failure rate during rotation: 29–31% (ViTPose), 21–31% (RTMPose),
16–33% (NSAQA). Error is worst in exactly the phase where angular momentum
is computed.

## The limit is the imagery, not the model

Independent relabelling of the same frames by the same annotator, with the
first-round labels hidden. 47 paired points across 6 frames.

- Median human self-disagreement: 7.8 px
- Excluding probable left/right side swaps (>30 px, 13% of points): 6.5 px
- Hips specifically: 20-22 px, the worst joint by a factor of two
- Visibility decisions agreed 99% of the time

Models (9.4 px) perform at roughly the level of a careful human annotator
(7.8 px) on this footage. The hips, which are the most ambiguous joint,
define the torso midpoint and therefore the torso angle that angular
momentum is derived from.

## Caveats to state

- Self-consistency is n=47 from 6 frames. Small.
- Errors are single-frame with no temporal filtering. Smoothing across a
  sequence would reduce random error but cannot fix gross failures, and may
  suppress the residual signal being measured.
- `head` is excluded from headline numbers: the hand label is not the same
  anatomical point as COCO `nose` or MPII `head_top`.
- Entry-phase frames are unusable: 0 of 20 had a labellable torso.
