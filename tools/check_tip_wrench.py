"""PhysX regression: proximal pressure must not appear as fingertip contact force."""

import argparse, json, os
from pathlib import Path
from isaaclab.app import AppLauncher
from revosim.launch import configure

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, default=Path("outputs/tip_wrench_check.json"))
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(configure(a), multi_gpu=False).app
try:
    import numpy as np
    from revosim.world import World
    from revosim.resources import FINGERS, pressure_regions

    w = World("sphere", device=a.device, hand_spacing=0.20)
    result = []
    for side, region in [
        ("left", "mcp"),
        ("right", "mcp"),
        ("left", "tip"),
        ("right", "tip"),
    ]:
        point, normal = w.surface_point(side, "index", region)
        w.reset_object(point + normal * 0.035)
        w.drive_object(point + normal * 0.005)
        samples = []
        for i in range(60):
            w.step()
            if i >= 35:
                v = w.observe()
                h = ("left", "right").index(side)
                f = FINGERS.index("index")
                samples.append(
                    [
                        float(v["pressure"][h, pressure_regions()["index_mcp"]].max()),
                        float(v["depth"][h, f].max()),
                        float(np.abs(v["wrench"][h, f]).max()),
                    ]
                )
        peak = np.max(samples, axis=0)
        row = dict(
            side=side,
            region=region,
            pressure_peak=float(peak[0]),
            depth_peak_m=float(peak[1]),
            tip_wrench_abs_max=float(peak[2]),
        )
        if region == "mcp":
            assert peak[0] > 1e-3, row
            assert peak[2] < 1e-5, row
        else:
            assert peak[1] > 1e-5 and peak[2] > 1e-4, row
        result.append(row)
        print("CHECK", json.dumps(row), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(passed=True, checks=result), indent=2) + "\n")
except BaseException:
    import traceback

    traceback.print_exc()
    os._exit(1)
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True)
