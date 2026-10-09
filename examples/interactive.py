#!/usr/bin/env python3
"""Move a physical probe with the keyboard and read the built-in hand sensors."""

import argparse, json, os
from pathlib import Path
from isaaclab.app import AppLauncher
from revosim.launch import configure

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--shape", choices=["sphere", "cube", "polyhedron"], default="sphere")
p.add_argument("--output", type=Path, default=Path("outputs/interactive"))
p.add_argument("--steps", type=int, default=0)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
app = AppLauncher(configure(a), multi_gpu=False).app
try:
    import numpy as np
    from revosim.world import World
    from revosim.dashboard import Dashboard
    from revosim.recording import Recorder

    world = World(a.shape, device=a.device, observation_hz=30)
    dashboard = Dashboard(world, live=not a.headless)
    record = Recorder(a.output, fps=30)
    (a.output / "sensor_layout.json").write_text(
        json.dumps(world.sensor_layout(), indent=2)
    )
    (a.output / "physics.json").write_text(json.dumps(world.physics_manifest, indent=2))
    point, normal = world.surface_point("left", "index", "tip")
    target = point + normal * 0.025
    world.reset_object(target)
    keys = set()
    if not a.headless:
        import carb.input, omni.appwindow

        interface = carb.input.acquire_input_interface()
        keyboard = omni.appwindow.get_default_app_window().get_keyboard()

        def event(e, *_):
            name = e.input.name
            if e.type in (
                carb.input.KeyboardEventType.KEY_PRESS,
                carb.input.KeyboardEventType.KEY_REPEAT,
            ):
                keys.add(name)
            elif e.type == carb.input.KeyboardEventType.KEY_RELEASE:
                keys.discard(name)
            return True

        subscription = interface.subscribe_to_keyboard_events(keyboard, event)
    frame = 0
    attached = True
    print(
        "W/S: forward/back; A/D: left/right; Q/E: up/down; SPACE: release; R: reset to left index; ESC: exit",
        flush=True,
    )
    while app.is_running() and (not a.steps or frame < a.steps):
        if "ESCAPE" in keys:
            break
        if "R" in keys:
            target = point + normal * 0.025
            world.reset_object(target)
            attached = True
            keys.discard("R")
        movement = np.array(
            [
                ("D" in keys) - ("A" in keys),
                ("W" in keys) - ("S" in keys),
                ("Q" in keys) - ("E" in keys),
            ]
        )
        target += movement * 0.0008
        if "SPACE" in keys:
            attached = False
            keys.discard("SPACE")
        if attached:
            world.drive_object(target)
        else:
            world.release_object()
        world.step()
        data = world.observe()
        image = dashboard.draw(data, "WASD move / Q,E height / Space release / R reset")
        record.append(data, image, "interactive")
        frame += 1
    record.close()
    dashboard.close()
    if not a.headless:
        interface.unsubscribe_to_keyboard_events(keyboard, subscription)
except BaseException:
    import traceback

    traceback.print_exc()
    os._exit(1)
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True)
