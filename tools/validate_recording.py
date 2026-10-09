#!/usr/bin/env python3
"""Validate a complete demonstration against the requested sensor contract."""

import argparse, json, subprocess
from pathlib import Path
import h5py, numpy as np

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
p.add_argument("--quick", action="store_true")
a = p.parse_args()
d = a.directory
r = json.loads((d / "report.json").read_text())
assert r["complete_sequence"], "Truncated sequence"
if not a.quick:
    assert not r["quick"], "Index-only preview is not full-finger evidence"
ids = [1] if a.quick else list(range(5))
for k in ["depth", "marker", "wrench", "rgb_change"]:
    assert (np.array(r["peaks"][k])[:, ids] > 1e-7).all(), (k, r["peaks"][k])
regions = (
    ["index_mcp", "index_pip", "palm"] if a.quick else list(r["pressure_region_peaks"])
)
for k in regions:
    assert min(r["pressure_region_peaks"][k]) > 1e-5, (k, r["pressure_region_peaks"][k])
assert r["released_depth_max"] < 1e-6 and r["released_pressure_max"] < 1e-6, (
    "No-contact release did not return to zero"
)
with h5py.File(d / "observations.h5") as f:
    n = len(f["time"])
    assert n == r["frames"]
    assert np.allclose(np.diff(f["time"][:]), 1 / r["fps"], atol=1e-8)
    shapes = {
        "rgb": (2, 5, 3, 240, 320),
        "depth": (2, 5, 240, 320),
        "marker": (2, 5, 100, 3),
        "pressure": (2, 247),
        "wrench": (2, 5, 6),
        "contact": (2, 5),
        "pad_pose": (2, 5, 7),
        "joint_pos": (2, 21),
        "object_pose": (7,),
    }
    for key, shape in shapes.items():
        assert f[key].shape == (n, *shape), (key, f[key].shape)
        for start in range(0, n, 8):
            block = f[key][start : start + 8]
            assert np.isfinite(block).all(), key
            if key in ("rgb", "depth", "pressure"):
                assert block.min() >= 0, key
            if key == "rgb":
                assert block.max() <= 1, key
    assert np.max(np.linalg.norm(f["object_pose"][:, :3], axis=1)) < 2, (
        "Object escaped workspace"
    )
    events = [
        json.loads(line) for line in (d / "frames.jsonl").read_text().splitlines()
    ]
    assert len(events) == n
    assert np.allclose([e["time"] for e in events], f["time"][:])
    hops = {}
    for side, hand_index in [("left", 0), ("right", 1)]:
        frame_ids = np.array(
            [
                i
                for i, e in enumerate(events)
                if e["phase"] == f"ballistic hop to {side} palm"
            ]
        )
        assert len(frame_ids) >= 3, "Missing cross-hand hop"
        times = f["time"][frame_ids]
        times -= times[0]
        positions = f["object_pose"][frame_ids, :3]
        free = times <= 0.20 + 1e-8
        fit = np.polyfit(times[free], positions[free], 2)
        acceleration = 2 * fit[0]
        assert (
            np.linalg.norm(acceleration[:2]) < 0.5 and -11 < acceleration[2] < -8.5
        ), ("Hop is not ballistic", side, acceleration)
        landing_response = float(f["pressure"][frame_ids, hand_index, 130:247].max())
        assert landing_response > 1e-5, ("Hop missed destination palm", side)
        hops[side] = {
            "fitted_free_flight_acceleration_m_s2": acceleration.tolist(),
            "destination_palm_peak": landing_response,
        }
layout = json.loads((d / "sensor_layout.json").read_text())
for side in ["left", "right"]:
    regions = layout["hands"][side]["pressure_regions"]
    assert sum(len(x["points_local_m"]) for x in regions) == 247
    assert len(layout["hands"][side]["joint_names"]) == 21
import imageio_ffmpeg

reader = imageio_ffmpeg.read_frames(str(d / "dashboard.mp4"), pix_fmt="rgb24")
meta = next(reader)
reader.close()
assert abs(meta["fps"] - r["fps"]) < 1e-5
assert meta["size"] == (2560, 1600), meta
decoded = subprocess.run(
    [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "error",
        "-threads",
        "4",
        "-i",
        str(d / "dashboard.mp4"),
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
        "-progress",
        "pipe:1",
    ],
    check=True,
    capture_output=True,
    text=True,
)
decoded_frames = [
    int(line.split("=", 1)[1])
    for line in decoded.stdout.splitlines()
    if line.startswith("frame=")
]
assert decoded_frames[-1] == n, "Video and numeric data frame counts differ"
result = {
    "passed": True,
    "full_finger_coverage": not a.quick,
    "shape": r["shape"],
    "frames": r["frames"],
    "fps": r["fps"],
    "hops": hops,
    "checks": [
        "complete sequence",
        "both hands",
        "each optical modality",
        "all requested pressure regions",
        "finite raw data",
        "array shapes",
        "uniform sampling",
        "bounded physical trajectory",
        "ballistic flight and both palm landings",
        "sensor coordinates",
        "full video decode and frame synchronization",
        "release to zero",
    ],
}
(d / "validation.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
