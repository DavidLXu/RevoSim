"""Smallest program: create two hands, press one finger, directly get arrays."""

import argparse
from isaaclab.app import AppLauncher
from revosim.launch import configure

p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(configure(args), multi_gpu=False).app
try:
    from revosim.world import World

    world = World(device=args.device)
    point, normal = world.surface_point("left", "index", "tip")
    world.reset_object(point + normal * 0.025)
    world.drive_object(point + normal * 0.005)
    for step in range(60):
        world.step()
        data = world.observe()
        if step % 10 == 0:
            print(
                {
                    key: tuple(value.shape)
                    for key, value in data.items()
                    if hasattr(value, "shape")
                }
            )
            print(
                "Index depth (m):",
                data["depth"][0, 1].max(),
                "wrench (N,Nm):",
                data["wrench"][0, 1],
            )
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True)
