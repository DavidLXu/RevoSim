import argparse, os
from isaaclab.app import AppLauncher
from revosim.launch import configure

p = argparse.ArgumentParser()
p.add_argument("--side", default="left")
p.add_argument("--finger", default="index")
p.add_argument("--region", default="palm")
p.add_argument("--shape", default="sphere", choices=["sphere", "cube", "polyhedron"])
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(configure(a), multi_gpu=False).app
try:
    import numpy as np
    from revosim.world import World

    w = World(shape=a.shape, device=a.device)
    import omni.physics.tensors.impl.api as physx

    view = physx.create_simulation_view("torch")
    view.set_subspace_roots("/")
    bodies = view.create_rigid_body_view("/World/*/*")
    contacts = view.create_rigid_contact_view(
        "/World/*/*", filter_patterns=["/World/Object"], max_contact_data_count=8192
    )
    p, n = w.surface_point(a.side, a.finger, a.region)
    print("SURFACE", p, n, flush=True)
    w.reset_object(p + n * 0.04)
    w.drive_object(p + n * 0.004)
    for i in range(60):
        w.step()
        v = w.observe()
        if i % 15 == 14:
            forces = contacts.get_contact_force_matrix(w.dt).cpu().numpy()[:, 0]
            print(
                "DIAG",
                i,
                "point",
                p,
                "object",
                v["object_pose"][:3],
                "pressure",
                v["pressure"].max(),
                flush=True,
            )
            print(
                [
                    (path, f.tolist())
                    for path, f in zip(bodies.prim_paths, forces)
                    if np.linalg.norm(f) > 0.001
                ],
                flush=True,
            )
except BaseException:
    import traceback

    traceback.print_exc()
    os._exit(1)
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True)
