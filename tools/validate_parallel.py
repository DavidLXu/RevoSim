#!/usr/bin/env python3
"""Validate the simultaneous demo's coverage, telemetry and encoded video."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("directory", type=Path)
a = p.parse_args()
report = json.loads((a.directory / "report.json").read_text())
assert report["complete_sequence"], "Truncated sequence"
assert report["max_targets_per_optical_pad"] == 1
assert len(report["coverage"]) == 32
for name, peak in report["coverage"].items():
    assert peak > (1e-5 if name.endswith("/tip") else 1e-3), (name, peak)
for key, peaks in report["peaks"].items():
    assert (np.asarray(peaks) > 0).all(), (key, peaks)

with np.load(a.directory / "telemetry.npz") as trace:
    n = report["frames"]
    for name in trace.files:
        assert len(trace[name]) == n, name
        assert np.isfinite(trace[name]).all(), name
    np.testing.assert_allclose(np.diff(trace["time"]), 1 / report["fps"], atol=1e-9)
    assert trace["wrench"].shape == (n, 2, 5, 6)
    assert trace["pressure"].shape == (n, 2, 247)
    assert trace["object_poses"].shape == (n, 3, 7)
    owners = trace["optical_object_index"]
    assert set(np.unique(owners)) == {-1, 0, 1, 2}
    assert (
        np.linalg.norm(np.ptp(trace["object_poses"][..., :3], axis=0), axis=-1) > 0.05
    ).all()
    # A contact-free pad must not retain another link's force or moment.
    assert np.abs(trace["wrench"][~trace["contact"].astype(bool)]).max(initial=0) == 0
    for key in ("depth_peak", "pressure", "wrench", "marker"):
        assert np.abs(trace[key][-12:]).max() < 1e-5, (key, "release residual")

probe = json.loads(
    subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_frames,duration",
            "-of",
            "json",
            str(a.directory / "dashboard.mp4"),
        ]
    )
)["streams"][0]
assert int(probe["nb_read_frames"]) == report["frames"]
assert probe["r_frame_rate"] == f"{report['fps']}/1"
assert (probe["width"], probe["height"]) == (2560, 1600)
subprocess.run(
    [
        "ffmpeg",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(a.directory / "dashboard.mp4"),
        "-f",
        "null",
        "-",
    ],
    check=True,
)
result = {
    "passed": True,
    "covered_regions": 32,
    "video": probe,
    "checks": [
        "finite telemetry",
        "60 Hz sampling" if report["fps"] == 60 else "30 Hz sampling",
        "all ten fingertip modalities",
        "all pressure regions",
        "all three objects moving",
        "all three objects produce optical contact",
        "no wrench without pad contact",
        "released sensors return to zero",
        "full video decode",
    ],
}
(a.directory / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
