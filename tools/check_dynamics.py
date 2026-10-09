"""Exercise every sensor region without rendering or writing large recordings."""

import argparse, json, os
from pathlib import Path
from isaaclab.app import AppLauncher
from revosim.launch import configure

p = argparse.ArgumentParser()
p.add_argument("--shape", choices=["sphere", "cube", "polyhedron"], default="sphere")
p.add_argument("--output", type=Path, default=Path("outputs/dynamics.json"))
p.add_argument("--quick", action="store_true")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(configure(a), multi_gpu=False).app
try:
    import numpy as np
    from revosim.world import World
    from revosim.demo import Demo
    from revosim.resources import pressure_regions

    w = World(a.shape, device=a.device, observation_hz=15)
    demo = Demo(w, quick=a.quick)
    start = w.time
    base = w.observe()
    phases = {}
    peaks = {k: np.zeros((2, 5)) for k in ["depth", "marker", "wrench", "rgb_change"]}
    pp = np.zeros((2, 247))
    last = None
    while app.is_running():
        phase = demo.update(w.time - start)
        if phase is None:
            break
        w.step()
        v = w.observe()
        assert np.linalg.norm(v["object_pose"][:3]) < 2, (
            "Object escaped workspace",
            v["object_pose"],
        )
        d = phases.setdefault(
            phase,
            dict(
                depth=0.0, marker=0.0, wrench=0.0, pressure=0.0, max_tracking_error=0.0
            ),
        )
        for k in ["depth", "marker", "wrench", "pressure"]:
            d[k] = max(d[k], float(np.abs(v[k]).max()))
        if w.target is not None:
            d["max_tracking_error"] = max(
                d["max_tracking_error"],
                float(np.linalg.norm(v["object_pose"][:3] - w.target)),
            )
        if phase != last:
            if last:
                print("PHASE", last, phases[last], flush=True)
            last = phase
        for k in peaks:
            raw = v["rgb"] - base["rgb"] if k == "rgb_change" else v[k]
            peaks[k] = np.maximum(peaks[k], np.abs(raw).reshape(2, 5, -1).max(-1))
        pp = np.maximum(pp, v["pressure"])
    report = dict(
        shape=a.shape,
        phases=phases,
        peaks={k: v.tolist() for k, v in peaks.items()},
        pressure_region_peaks={
            k: pp[:, sl].max(-1).tolist() for k, sl in pressure_regions().items()
        },
        released_depth_max=float(v["depth"].max()),
        released_pressure_max=float(v["pressure"].max()),
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print(
        "DYNAMICS_COMPLETE",
        json.dumps({k: v for k, v in report.items() if k != "phases"}),
        flush=True,
    )
except BaseException:
    import traceback

    traceback.print_exc()
    os._exit(1)
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True)
