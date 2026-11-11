"""
Cuts individual dive clips out of the long MTL-AQA broadcast videos.

Put this file in C:\\Users\\yada7\\aqa next to 01.mkv, 02.mkv, 04.mkv, 17.mkv, 26.mkv
and next to the Ready_2_Use folder. Then run:

    python cut_clips.py

Output: a clips\\ folder with one short .mp4 per dive, named <video>_<dive>.mp4
plus clips_index.csv listing every clip with its labels.
"""

import os
import csv
import pickle
import subprocess

FPS = 25.0
VIDEOS = ["01", "02", "04", "17", "26"]
ANNOT = os.path.join("Ready_2_Use", "MTL-AQA_split_0_data", "final_annotations_dict.pkl")
OUTDIR = "clips"


def main():
    if not os.path.exists(ANNOT):
        print("Cannot find", ANNOT)
        print("Make sure the Ready_2_Use folder is in this directory.")
        return

    with open(ANNOT, "rb") as f:
        annotations = pickle.load(f)

    os.makedirs(OUTDIR, exist_ok=True)

    wanted = set(int(v) for v in VIDEOS)
    jobs = [(k, v) for k, v in annotations.items() if k[0] in wanted]
    jobs.sort()

    print("Found", len(jobs), "dives across videos", VIDEOS)

    rows = []
    made = 0
    skipped = 0
    failed = 0

    for (vid, dive), info in jobs:
        src = "%02d.mkv" % vid
        if not os.path.exists(src):
            continue

        name = "%02d_%03d" % (vid, dive)
        out = os.path.join(OUTDIR, name + ".mp4")

        start = info["start_frame"] / FPS
        duration = (info["end_frame"] - info["start_frame"]) / FPS

        rows.append({
            "clip": name,
            "video": vid,
            "dive": dive,
            "start_frame": info["start_frame"],
            "end_frame": info["end_frame"],
            "n_frames": info["end_frame"] - info["start_frame"],
            "final_score": info["final_score"],
            "difficulty": info["difficulty"],
            "execution_score": round(info["final_score"] / info["difficulty"], 4),
            "ss_no": info["ss_no"],
            "tw_no": info["tw_no"],
            "position": info["position"],
            "rotation_type": info["rotation_type"],
            "armstand": info["armstand"],
        })

        if os.path.exists(out):
            skipped += 1
            continue

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", "%.3f" % start,
            "-i", src,
            "-t", "%.3f" % duration,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-an", out,
        ]
        result = subprocess.run(cmd)
        if result.returncode == 0:
            made += 1
        else:
            failed += 1
            print("FAILED:", name)

        done = made + skipped + failed
        if done % 25 == 0:
            print("  ... %d / %d" % (done, len(rows)))

    with open("clips_index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Done. New clips:", made, " Already existed:", skipped, " Failed:", failed)
    print("Index written to clips_index.csv with", len(rows), "rows")
    non_twist = sum(1 for r in rows if r["tw_no"] == 0)
    print("Non-twisting dives (your main subset):", non_twist)


if __name__ == "__main__":
    main()
