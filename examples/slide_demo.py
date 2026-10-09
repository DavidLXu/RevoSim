#!/usr/bin/env python3
"""Run the sphere/cube/polyhedron demos, with all ten fingertip data streams."""

import argparse, json, os, subprocess, sys
from pathlib import Path


def main():
    from isaaclab.app import AppLauncher
    from revosim.launch import configure

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shape", choices=["all", "sphere", "cube", "polyhedron"], default="all"
    )
    p.add_argument("--output", type=Path, default=Path("outputs/demo"))
    p.add_argument("--fps", type=int, choices=[15, 30, 60], default=30)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Short development run on both index fingers; not full coverage",
    )
    p.add_argument(
        "--no-data",
        action="store_true",
        help="Skip lossless HDF5; retain dashboard and validation report",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Development limit; marks the report incomplete",
    )
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    if a.shape == "all":
        a.output.mkdir(parents=True, exist_ok=True)
        for shape in ["sphere", "cube", "polyhedron"]:
            args = [
                sys.executable,
                __file__,
                "--shape",
                shape,
                "--output",
                str(a.output / shape),
                "--fps",
                str(a.fps),
                "--device",
                a.device,
            ]
            for flag in ["headless", "quick", "no_data"]:
                if getattr(a, flag):
                    args.append("--" + flag.replace("_", "-"))
            if a.max_frames:
                args += ["--max-frames", str(a.max_frames)]
            if a.kit_args:
                args.append("--kit_args=" + a.kit_args)
            if a.experience:
                args += ["--experience", a.experience]
            subprocess.run(args, check=True)
        return
    a.enable_cameras = True
    app = AppLauncher(configure(a), multi_gpu=False, renderer="RaytracedLighting").app
    try:
        import numpy as np
        from revosim.world import World
        from revosim.demo import Demo
        from revosim.dashboard import Dashboard
        from revosim.recording import Recorder
        from revosim.resources import pressure_regions

        w = World(a.shape, device=a.device, observation_hz=a.fps)
        demo = Demo(w, quick=a.quick)
        dashboard = Dashboard(w, live=not a.headless)
        rec = Recorder(a.output, fps=a.fps, save_data=not a.no_data)
        # Calibration/masks are immutable and saved once, separately from observations.
        np.savez_compressed(
            a.output / "calibration.npz",
            **{f"{s}_{k}": v for s, d in w.calibration.items() for k, v in d.items()},
        )
        (a.output / "physics.json").write_text(json.dumps(w.physics_manifest, indent=2))
        (a.output / "sensor_layout.json").write_text(
            json.dumps(w.sensor_layout(), indent=2)
        )
        baseline = w.observe()
        peaks = {
            k: np.zeros((2, 5)) for k in ["depth", "marker", "wrench", "rgb_change"]
        }
        pp = np.zeros((2, 247))
        frame = 0
        maxima = {}
        snapshots = a.output / "snapshots"
        snapshots.mkdir(exist_ok=True)
        start = w.time
        while app.is_running():
            phase = demo.update(w.time - start)
            if phase is None:
                break
            w.step()
            v = w.observe()
            im = dashboard.draw(v, phase)
            rec.append(v, im, phase)
            for key, raw in [
                ("depth", v["depth"]),
                ("marker", v["marker"]),
                ("wrench", v["wrench"]),
                ("rgb_change", v["rgb"] - baseline["rgb"]),
            ]:
                peak = np.abs(raw).reshape(2, 5, -1).max(-1)
                peaks[key] = np.maximum(peaks[key], peak)
            pp = np.maximum(pp, v["pressure"])
            if (
                phase not in maxima
                or float(v["depth"].max() + v["pressure"].max()) > maxima[phase]
            ):
                maxima[phase] = float(v["depth"].max() + v["pressure"].max())
                if frame % 5 == 0:
                    from PIL import Image
                    import re

                    Image.fromarray(im).save(a.output / "latest.png")
                    name = re.sub(r"[^a-zA-Z0-9]+", "_", phase).strip("_")
                    Image.fromarray(im).save(snapshots / (name + ".png"))
            if frame % 60 == 0:
                print(
                    "FRAME",
                    frame,
                    "TIME",
                    round(w.time - start, 2),
                    "PHASE",
                    phase,
                    "DEPTH_MM",
                    round(float(v["depth"].max() * 1000), 3),
                    "PRESSURE",
                    round(float(v["pressure"].max()), 3),
                    flush=True,
                )
            frame += 1
            if a.max_frames and frame >= a.max_frames:
                break
        report = {
            "shape": a.shape,
            "frames": frame,
            "fps": a.fps,
            "duration_s": frame / a.fps,
            "complete_sequence": not a.max_frames
            and w.time - start >= demo.duration - 1 / a.fps,
            "quick": a.quick,
            "peaks": {k: v.tolist() for k, v in peaks.items()},
            "pressure_region_peaks": {
                name: pp[:, sl].max(-1).tolist()
                for name, sl in pressure_regions().items()
            },
            "released_depth_max": float(v["depth"].max()),
            "released_pressure_max": float(v["pressure"].max()),
            "physics_hz": 240,
        }
        (a.output / "report.json").write_text(json.dumps(report, indent=2))
        rec.close()
        dashboard.close()
        print("DEMO_COMPLETE", json.dumps(report), flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        os._exit(1)
    finally:
        app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
