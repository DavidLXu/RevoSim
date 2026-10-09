#!/usr/bin/env python3
"""Three simultaneous tactile probes; seeded coverage of both hands in a short clip."""

import argparse
import json
import os
from pathlib import Path
from isaaclab.app import AppLauncher
from revosim.launch import configure

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, default=Path("outputs/parallel"))
p.add_argument("--seed", type=int, default=42)
p.add_argument("--fps", type=int, choices=(30, 60), default=60)
p.add_argument("--max-frames", type=int, default=0)
p.add_argument("--no-data", action="store_true")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
app = AppLauncher(configure(a), multi_gpu=False, renderer="RaytracedLighting").app
try:
    import numpy as np
    from PIL import Image
    from revosim.world import World
    from revosim.parallel_demo import ParallelDemo
    from revosim.dashboard import Dashboard
    from revosim.recording import Recorder

    w = World(
        ("sphere", "cube", "polyhedron"),
        device=a.device,
        observation_hz=a.fps,
        hand_spacing=0.20,
    )
    demo = ParallelDemo(w, seed=a.seed)
    dashboard = Dashboard(w, live=not a.headless)
    recorder = Recorder(a.output, fps=a.fps, save_data=not a.no_data)
    (a.output / "sensor_layout.json").write_text(
        json.dumps(w.sensor_layout(), indent=2)
    )
    (a.output / "physics.json").write_text(json.dumps(w.physics_manifest, indent=2))
    np.savez_compressed(
        a.output / "calibration.npz",
        **{f"{s}_{k}": v for s, d in w.calibration.items() for k, v in d.items()},
    )
    start = w.time
    baseline = w.observe()
    metadata = json.loads((a.output / "metadata.json").read_text())
    metadata.update(
        object_names=list(w.objects),
        random_seed=a.seed,
        optical_object_index="-1 means no optical indentation; otherwise indexes object_names",
        pressure_fusion="Per-taxel maximum of raw target responses, followed by one diffusion",
        optical_scope="Disjoint target contacts per fingertip; RGB/Marker/Depth from the same target",
    )
    (a.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    trace = {
        k: []
        for k in (
            "time",
            "wrench",
            "contact",
            "pressure",
            "marker",
            "joint_pos",
            "object_poses",
            "optical_object_index",
            "depth_peak",
        )
    }
    best_score = -1
    peaks = {
        key: np.zeros((2, 5)) for key in ("depth", "marker", "wrench", "rgb_change")
    }
    frames = 0
    while app.is_running():
        phase = demo.update(w.time - start)
        if phase is None:
            break
        w.step()
        values = w.observe()
        demo.record(values)
        pixels = dashboard.draw(values, phase)
        recorder.append(values, pixels, phase)
        for key in trace:
            trace[key].append(
                values["depth"].max(axis=(-2, -1))
                if key == "depth_peak"
                else np.asarray(values[key]).copy()
            )
        score = (
            100
            * int(
                ((values["pressure"].max(1) > 0.001) | values["contact"].any(1)).sum()
            )
            + 10 * int(values["contact"].sum())
            + float(values["pressure"].sum())
        )
        if score > best_score:
            best_score = score
            Image.fromarray(pixels).save(a.output / "best.png")
        for key in peaks:
            raw = (
                values["rgb"] - baseline["rgb"] if key == "rgb_change" else values[key]
            )
            peaks[key] = np.maximum(peaks[key], np.abs(raw).reshape(2, 5, -1).max(-1))
        if frames % 60 == 0:
            Image.fromarray(pixels).save(a.output / "latest.png")
            print("FRAME", frames, "/", round(demo.duration * a.fps), phase, flush=True)
        frames += 1
        if a.max_frames and frames >= a.max_frames:
            break
    recorder.close()
    np.savez_compressed(
        a.output / "telemetry.npz", **{k: np.asarray(v) for k, v in trace.items()}
    )
    dashboard.close()
    report = dict(
        frames=frames,
        fps=a.fps,
        duration_s=frames / a.fps,
        complete_sequence=not a.max_frames,
        seed=a.seed,
        object_names=list(w.objects),
        wrist_spacing_m=w.hand_spacing,
        wrench_scope="Only DIP_rubber_link contacts with all three objects; normal + friction",
        optical_scope="At most one object per fingertip; independently computed RGB/depth/marker",
        pressure_fusion="Per-taxel maximum before a single diffusion pass",
        max_targets_per_optical_pad=max(
            s.max_simultaneous_targets_per_pad for s in w.sensors.values()
        ),
        coverage=demo.coverage,
        peaks={k: v.tolist() for k, v in peaks.items()},
        released_depth_max=float(values["depth"].max()),
        released_pressure_max=float(values["pressure"].max()),
        released_wrench_max=float(np.abs(values["wrench"]).max()),
    )
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("DEMO_COMPLETE", json.dumps(report), flush=True)
except BaseException:
    import traceback

    traceback.print_exc()
    os._exit(1)
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True)
