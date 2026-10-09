"""Small physical contact check for both hands and all sensor modalities."""

import argparse, json, os
from pathlib import Path
from isaaclab.app import AppLauncher
from revosim.launch import configure

p = argparse.ArgumentParser()
p.add_argument("--shape", choices=["sphere", "cube", "polyhedron"], default="sphere")
p.add_argument("--output", type=Path, default=Path("outputs/smoke"))
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(configure(a), multi_gpu=False).app
try:
    import numpy as np
    from revosim.world import World

    w = World(shape=a.shape, device=a.device)
    a.output.mkdir(parents=True, exist_ok=True)
    w.reset_object((0, 0, 0.8))
    w.drive_object((0, 0, 0.8))
    w.step()
    base = w.observe()
    print("BASELINE", base["depth"].max(), base["pressure"].max(), flush=True)
    report = {"shape": a.shape, "phases": []}
    for side in ["left", "right"]:
        for region in ["tip", "mcp", "palm"]:
            point, normal = w.surface_point(side, "index", region)
            print("PROBE", side, region, point, normal, flush=True)
            w.reset_object(point + normal * 0.04)
            w.drive_object(point + normal * 0.006)
            peak = {
                k: 0.0 for k in ["depth", "pressure", "marker", "wrench", "rgb_change"]
            }
            for frame in range(45):
                w.step()
                v = w.observe()
                for k in ["depth", "pressure", "marker", "wrench"]:
                    peak[k] = max(peak[k], float(np.abs(v[k]).max()))
                peak["rgb_change"] = max(
                    peak["rgb_change"], float(np.abs(v["rgb"] - base["rgb"]).max())
                )
            print("PEAK", side, region, peak, flush=True)
            report["phases"].append(
                dict(
                    side=side,
                    region=region,
                    peak=peak,
                    object_pose=v["object_pose"].tolist(),
                )
            )
            np.savez_compressed(a.output / f"{side}_{region}.npz", **v)
    (a.output / "report.json").write_text(json.dumps(report, indent=2))
    print("SMOKE_COMPLETE", flush=True)
except BaseException:
    import traceback

    traceback.print_exc()
    os._exit(1)
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True)
