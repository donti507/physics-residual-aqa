"""
Pulls a stratified sample of frames out of the dive clips for hand labelling.

Put this in C:\\Users\\yada7\\aqa next to the clips\\ folder and clips_index.csv.
Run:  py extract_label_frames.py

Output: label_frames\\ containing 160 jpgs named <clip>_p<phase>.jpg
        plus label_frames_index.csv recording which clip and phase each came from.

Phases sampled, as a fraction through the dive:
  0.08 takeoff      0.20 early flight   0.35 rotation 1   0.50 peak
  0.65 rotation 2   0.80 late flight    0.92 pre-entry    0.99 entry
Entry frames are kept as their own phase because pose estimators fail there.
"""

import os
import csv
import random
import subprocess

CLIPS_DIR = "clips"
INDEX = "clips_index.csv"
OUTDIR = "label_frames"
N_CLIPS = 20
PHASES = [0.08, 0.20, 0.35, 0.50, 0.65, 0.80, 0.92, 0.97]
PHASE_NAMES = ["takeoff", "early", "rot1", "peak",
               "rot2", "late", "preentry", "entry"]
FPS = 25.0
SEED = 0


def main():
    if not os.path.exists(INDEX):
        print("Cannot find", INDEX)
        return

    with open(INDEX, newline="") as f:
        rows = list(csv.DictReader(f))

    # Keep mostly non-twisting dives (the main subset), plus a few twisting
    # ones so the noise estimate is not measured on an unusually easy case.
    non_twist = [r for r in rows if int(r["tw_no"]) == 0]
    twist = [r for r in rows if int(r["tw_no"]) > 0]

    rng = random.Random(SEED)

    # Spread across all five source videos so one camera does not dominate.
    chosen = []
    by_video = {}
    for r in non_twist:
        by_video.setdefault(r["video"], []).append(r)
    for vid in sorted(by_video):
        pool = by_video[vid]
        rng.shuffle(pool)
        chosen.extend(pool[:3])          # 3 non-twisting per video = 15
    rng.shuffle(twist)
    chosen.extend(twist[:5])             # plus 5 twisting = 20
    chosen = chosen[:N_CLIPS]

    os.makedirs(OUTDIR, exist_ok=True)

    out_rows = []
    made = 0
    failed = 0

    for r in chosen:
        clip_path = os.path.join(CLIPS_DIR, r["clip"] + ".mp4")
        if not os.path.exists(clip_path):
            print("Missing clip:", clip_path)
            continue

        n_frames = int(r["n_frames"])
        duration = n_frames / FPS

        for frac, pname in zip(PHASES, PHASE_NAMES):
            t = frac * duration
            name = "%s_%s" % (r["clip"], pname)
            out_path = os.path.join(OUTDIR, name + ".png")

            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", "%.3f" % t,
                "-i", clip_path,
                "-frames:v", "1",
                out_path,
            ]
            res = subprocess.run(cmd)
            if res.returncode == 0 and os.path.exists(out_path):
                made += 1
                out_rows.append({
                    "image": name + ".png",
                    "clip": r["clip"],
                    "video": r["video"],
                    "dive": r["dive"],
                    "phase": pname,
                    "phase_frac": frac,
                    "frame_in_clip": int(round(frac * n_frames)),
                    "abs_frame": int(r["start_frame"]) + int(round(frac * n_frames)),
                    "tw_no": r["tw_no"],
                    "ss_no": r["ss_no"],
                    "difficulty": r["difficulty"],
                    "final_score": r["final_score"],
                })
            else:
                failed += 1
                print("FAILED:", name)

    with open("label_frames_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print()
    print("Clips sampled:", len(chosen))
    print("Frames written:", made, " Failed:", failed)
    print("Index: label_frames_index.csv")
    print()
    print("Next: open the labelling tool and load the", OUTDIR, "folder.")


if __name__ == "__main__":
    main()
