"""Exercise the actual Omni UI image provider even on a headless render worker."""

import argparse, os
from isaaclab.app import AppLauncher
from revosim.launch import configure

p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
app = AppLauncher(configure(a), multi_gpu=False).app
try:
    from revosim.world import World
    from revosim.dashboard import Dashboard

    w = World(device=a.device)
    d = Dashboard(w, live=True)
    for _ in range(3):
        w.step()
        d.draw(w.observe(), "UI smoke check")
    d.close()
    print("UI_PROVIDER_PASS", flush=True)
except BaseException:
    import traceback

    traceback.print_exc()
    os._exit(1)
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True)
