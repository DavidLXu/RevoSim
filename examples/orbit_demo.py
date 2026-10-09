#!/usr/bin/env python3
"""Render a full studio orbit of two nearby hands driven by PhysX joint motors.

The simulation records link poses first; the path tracer then renders those exact
poses without interpolation. The contact dashboard remains a separate example.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def joint_target(names, side, fraction, limits, peace_sway_deg=12.0):
    import numpy as np

    ids = {name: i for i, name in enumerate(names)}

    def smoothstep(u):
        u = np.clip(u, 0, 1)
        return u**3 * (10 + u * (-15 + 6 * u))

    def pose(curled=(), spread=False):
        q = np.zeros(len(names), dtype=np.float32)
        for finger in curled:
            for joint, value in zip(("MCP", "PIP", "DIP"), (1.15, 1.35, 0.95)):
                q[ids[f"{side}_{finger}_{joint}_joint"]] = value
        if curled:
            # Keep the thumb outside the curled fingers for a clear visual gap.
            for joint, value in zip(
                ("CMP", "CMR", "MCP", "PIP", "DIP"),
                (0.40, 0.10, 0.65, 0.80, 0.55),
            ):
                q[ids[f"{side}_thumb_{joint}_joint"]] = value
        if spread:
            q[ids[f"{side}_index_MPR_joint"]] = 0.24
            q[ids[f"{side}_middle_MPR_joint"]] = -0.10
        return q

    opened = pose()
    curled = pose(("index", "middle", "ring", "little"))
    peace = pose(("ring", "little"), spread=True)
    # Times describe the native 12-second showcase. Changing duration/speed scales
    # the physical motor targets, rather than retiming an already rendered video.
    keys = [
        (0, opened),
        (0.5, opened),
        (1.9, curled),
        (2.5, curled),
        (3.9, opened),
        (4.1, opened),
        (5.5, peace),
        (9.0, peace),
        (10.4, opened),
        (12, opened),
    ]
    t = 12 * fraction
    target = opened.copy()
    for (t0, q0), (t1, q1) in zip(keys, keys[1:]):
        if t <= t1:
            target = q0 + (q1 - q0) * smoothstep((t - t0) / (t1 - t0))
            break
    if 5.5 <= t <= 9.0:
        u = (t - 5.5) / 3.5
        envelope = smoothstep(min(u / 0.15, (1 - u) / 0.15, 1))
        sway = math.radians(peace_sway_deg) * envelope * math.sin(4 * math.pi * u)
        # Move the extended fingers together, preserving the V opening angle.
        for finger in ("index", "middle"):
            target[ids[f"{side}_{finger}_MPR_joint"]] += sway
    assert np.all(target >= limits[:, 0]) and np.all(target <= limits[:, 1])
    return target


def simulate(a, target_fn=joint_target, motion_metadata=None):
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from revosim.scene import create_hand

    dt = 1 / 240
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=dt, render_interval=240 // a.fps, device=a.device, use_fabric=True
        )
    )
    hands = {}
    physics = {}
    for side, sign in (("left", -1), ("right", 1)):
        hands[side], physics[side] = create_hand(
            sim.stage,
            side,
            translation=(sign * a.hand_spacing / 2, 0, 0.045),
            orientation=(math.sqrt(0.5), 0, 0, -math.sqrt(0.5)),
        )
    sim.reset()
    for hand in hands.values():
        hand.reset()
        q = hand.data.default_joint_pos.clone()
        hand.write_joint_state_to_sim(q, torch.zeros_like(q))
        hand.set_joint_position_target(q)
    for _ in range(240):
        for hand in hands.values():
            hand.write_data_to_sim()
        sim.step(render=False)
        for hand in hands.values():
            hand.update(dt)

    duration = a.duration / a.motion_speed
    frames = round(duration * a.fps)
    joint_limits = {
        side: hand.data.joint_pos_limits[0].cpu().numpy().copy()
        for side, hand in hands.items()
    }
    data = {
        s: {k: [] for k in ("body_pose", "joint_pos", "joint_vel", "target", "torque")}
        for s in hands
    }
    for frame in range(frames):
        for substep in range(240 // a.fps):
            t = frame / a.fps + (substep + 1) * dt
            targets = {}
            for side, hand in hands.items():
                target = target_fn(
                    hand.joint_names,
                    side,
                    t / duration,
                    joint_limits[side],
                    a.peace_sway_deg,
                )
                targets[side] = target
                hand.set_joint_position_target(
                    torch.as_tensor(target, device=a.device)[None]
                )
                hand.write_data_to_sim()
            sim.step(render=False)
            for hand in hands.values():
                hand.update(dt)
        for side, hand in hands.items():
            for key, value in (
                ("body_pose", hand.data.body_link_pose_w[0]),
                ("joint_pos", hand.data.joint_pos[0]),
                ("joint_vel", hand.data.joint_vel[0]),
                ("torque", hand.data.applied_torque[0]),
            ):
                data[side][key].append(value.cpu().numpy().copy())
            data[side]["target"].append(targets[side].copy())
        if frame % 120 == 0:
            print("SIM_FRAME", frame, "/", frames, flush=True)
    payload = {"time": (np.arange(frames) + 1) / a.fps}
    reports = {}
    for side, hand in hands.items():
        arrays = {k: np.asarray(v) for k, v in data[side].items()}
        assert all(np.isfinite(v).all() for v in arrays.values())
        limits = hand.data.joint_pos_limits[0].cpu().numpy()
        violation = np.maximum(
            limits[:, 0] - arrays["joint_pos"], arrays["joint_pos"] - limits[:, 1]
        ).clip(0)
        payload.update({f"{side}_{k}": v for k, v in arrays.items()})
        payload[f"{side}_body_names"] = hand.body_names
        payload[f"{side}_joint_names"] = hand.joint_names
        payload[f"{side}_joint_limits"] = limits
        reports[side] = {
            "tracking_rms_rad": float(
                np.sqrt(np.mean((arrays["joint_pos"] - arrays["target"]) ** 2))
            ),
            "max_speed_rad_s": float(np.abs(arrays["joint_vel"]).max()),
            "max_joint_limit_violation_rad": float(violation.max()),
            "joint_ranges_rad": np.ptp(arrays["joint_pos"], axis=0).tolist(),
            "joint_names": hand.joint_names,
            "joint_min_rad": arrays["joint_pos"].min(axis=0).tolist(),
            "joint_max_rad": arrays["joint_pos"].max(axis=0).tolist(),
            "loop_seam_max_joint_error_rad": float(
                np.abs(arrays["joint_pos"][-1] - arrays["joint_pos"][0]).max()
            ),
        }
    np.savez_compressed(a.output / "trajectory.npz", **payload)
    report = {
        "physics_hz": 240,
        "fps": a.fps,
        "frames": frames,
        "duration_s": frames / a.fps,
        "reference_duration_s": a.duration,
        "motion_speed": a.motion_speed,
        "post_render_retiming": False,
        "peace_sway_amplitude_deg": a.peace_sway_deg,
        "finger_flexion_targets_rad": {"MCP": 1.15, "PIP": 1.35, "DIP": 0.95},
        "wrist_center_spacing_m": a.hand_spacing,
        "hands": reports,
        "fixed_wrist": True,
        "self_collision": False,
        "source_mass_inertia_preserved": True,
        "motion": "PhysX PD-driven relaxed fist, open hand, V-sign with two lateral sway cycles; quintic target transitions",
        "scope": "Free-space appearance and articulation showcase, not a physical-hand calibration test",
    }
    if motion_metadata:
        report.update(motion_metadata)
    (a.output / "simulation.json").write_text(json.dumps(report, indent=2) + "\n")
    (a.output / "physics.json").write_text(json.dumps(physics, indent=2) + "\n")
    assert all(r["max_joint_limit_violation_rad"] < 0.01 for r in reports.values()), (
        reports
    )
    print("SIM_COMPLETE", json.dumps(report), flush=True)


def render(a, app, *, preview_poses=None, video_name=None, camera_offset_deg=0):
    import numpy as np
    import imageio.v2 as imageio
    from PIL import Image
    import carb
    import omni.usd
    import omni.timeline
    import omni.replicator.core as rep
    from pxr import Usd, UsdGeom, UsdLux, Gf
    from revosim.resources import ASSETS
    import isaaclab.sim as sim_utils

    trajectory = np.load(a.output / "trajectory.npz")
    simulation = json.loads((a.output / "simulation.json").read_text())
    # Re-rendering must preserve the recorded timing even when CLI defaults differ.
    a.fps = simulation["fps"]
    a.hand_spacing = simulation["wrist_center_spacing_m"]
    count = len(trajectory["time"])
    omni.timeline.get_timeline_interface().stop()
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    settings = carb.settings.get_settings()
    settings.set_int("/rtx/pathtracing/spp", a.samples)
    settings.set_int("/rtx/pathtracing/totalSpp", a.samples * 2)
    settings.set_int("/rtx/pathtracing/maxBounces", 6)
    settings.set_bool("/rtx/pathtracing/optixDenoiser/enabled", True)
    settings.set_bool("/rtx/post/motionblur/enabled", False)
    ops = {}
    models = []
    for side in ("left", "right"):
        model = stage.DefinePrim(f"/World/{side.capitalize()}", "Xform")
        model.GetReferences().AddReference(
            str(ASSETS / f"hands/revo3_{side}_bionic_silver.usda")
        )
        UsdGeom.Xformable(model).ClearXformOpOrder()
        models.append(model)
        ops[side] = []
        for name in trajectory[f"{side}_body_names"]:
            prim = stage.GetPrimAtPath(str(model.GetPath()) + "/" + str(name))
            assert prim.IsValid(), str(name)
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            ops[side].append(xf.AddTransformOp(opSuffix="measuredPhysX"))

    def set_frame(frame):
        for side in ("left", "right"):
            for op, state in zip(ops[side], trajectory[f"{side}_body_pose"][frame]):
                transform = Gf.Matrix4d().SetRotate(
                    Gf.Quatd(float(state[3]), Gf.Vec3d(*state[4:7].astype(float)))
                )
                transform.SetTranslateOnly(Gf.Vec3d(*state[:3].astype(float)))
                op.Set(transform)

    set_frame(0)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    boxes = [cache.ComputeWorldBound(model).ComputeAlignedBox() for model in models]
    lo = np.min([list(b.GetMin()) for b in boxes], axis=0)
    hi = np.max([list(b.GetMax()) for b in boxes], axis=0)
    center = (lo + hi) / 2
    center[0] = 0
    size = hi - lo
    ground = sim_utils.CuboidCfg(
        size=(100, 100, 0.01),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.035, 0.045, 0.060), roughness=0.86
        ),
    )
    ground.func("/World/Ground", ground, translation=(0, 0, float(lo[2] - 0.018)))
    dome = UsdLux.DomeLight.Define(stage, "/World/Ambient")
    dome.CreateIntensityAttr(300)
    dome.CreateColorAttr(Gf.Vec3f(1, 1, 1))

    def look_at(prim, eye, target):
        xf = UsdGeom.Xformable(prim)
        transforms = xf.GetOrderedXformOps()
        op = transforms[0] if transforms else xf.AddTransformOp()
        op.Set(
            Gf.Matrix4d()
            .SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
            .GetInverse()
        )

    for name, offset, width, height, power in (
        ("Key", (-0.48, -0.42, 0.5), 0.38, 0.60, 4000),
        ("Rim", (0.48, 0.32, 0.32), 0.18, 0.65, 4800),
        ("Fill", (0.50, -0.30, 0.12), 0.45, 0.50, 1600),
        ("Back", (-0.40, 0.42, 0.16), 0.35, 0.5, 1800),
    ):
        light = UsdLux.RectLight.Define(stage, "/World/" + name)
        light.CreateWidthAttr(width)
        light.CreateHeightAttr(height)
        light.CreateIntensityAttr(power)
        light.CreateNormalizeAttr(False)
        light.CreateColorAttr(Gf.Vec3f(1, 1, 1))
        look_at(light.GetPrim(), center + offset, center)
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(48)
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.005, 20))
    radius = max(
        size[0] / (36 / 48) * 1.4, size[2] / (36 / 48 * a.height / a.width) * 1.45
    )
    elevation = radius * 0.32
    product = rep.create.render_product(str(camera.GetPath()), (a.width, a.height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    rgb.attach(product)

    def capture(frame, warm=False, angle=None, distance_scale=1.0):
        set_frame(frame)
        if angle is None:
            angle = 2 * math.pi * frame / count + math.radians(camera_offset_deg)
        eye = center + distance_scale * np.array(
            (radius * math.sin(angle), -radius * math.cos(angle), elevation)
        )
        look_at(camera.GetPrim(), eye, center)
        for _ in range(12 if warm else 2):
            app.update()
        pixels = rgb.get_data()[..., :3]
        assert pixels.shape == (a.height, a.width, 3) and pixels.std() > 2
        return pixels.copy()

    previews = a.output / "previews"
    previews.mkdir(exist_ok=True)
    for fraction in (0, 0.125, 0.25, 0.5, 0.625, 0.75, 0.875):
        frame = round(fraction * count)
        Image.fromarray(capture(frame, warm=True)).save(
            previews / f"orbit_{round(360 * fraction):03d}.png"
        )
        print("PREVIEW", round(360 * fraction), flush=True)
    pose_previews = preview_poses if preview_poses is not None else (
        ("relaxed_fist", 2.2 / 12),
        ("peace_sway_left", (5.5 + 3.5 * 0.125) / 12),
        ("peace_sway_right", (5.5 + 3.5 * 0.375) / 12),
    )
    for label, fraction in pose_previews:
        Image.fromarray(capture(round(fraction * count), warm=True, angle=0)).save(
            previews / f"gesture_{label}.png"
        )
        print("PREVIEW", label, flush=True)
        if a.preview and preview_poses is not None:
            for angle_deg in (60, 180, 300):
                Image.fromarray(
                    capture(
                        round(fraction * count), warm=True,
                        angle=math.radians(angle_deg), distance_scale=0.85,
                    )
                ).save(previews / f"gesture_{label}_{angle_deg:03d}.png")
    if a.preview and preview_poses is None:
        for angle_deg in (0, 60, 120, 180, 240, 300):
            Image.fromarray(
                capture(
                    round(2.2 / 12 * count),
                    warm=True,
                    angle=math.radians(angle_deg),
                    distance_scale=0.70,
                )
            ).save(previews / f"fist_closeup_{angle_deg:03d}.png")
    if not a.preview:
        writer = imageio.get_writer(
            a.output / (video_name or f"revosim-orbit-{a.fps}fps.mp4"),
            fps=a.fps,
            codec="libx264",
            macro_block_size=1,
            ffmpeg_params=[
                "-threads",
                "4",
                "-crf",
                "17",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ],
        )
        try:
            for frame in range(count):
                writer.append_data(capture(frame, warm=frame == 0))
                if frame % 60 == 0:
                    print("RENDER_FRAME", frame, "/", count, flush=True)
        finally:
            writer.close()
        report = {
            "frames": count,
            "fps": a.fps,
            "duration_s": count / a.fps,
            "motion_speed": simulation.get("motion_speed", 1.0),
            "post_render_retiming": False,
            "self_collision": simulation["self_collision"],
            "resolution": [a.width, a.height],
            "renderer": "PathTracing",
            "samples_per_update": a.samples,
            "updates_per_frame": 2,
            "orbit_degrees": 360,
            "camera_offset_deg": camera_offset_deg,
            "frame_angles": "2*pi*frame/frame_count + radians(camera_offset_deg); seamless periodic camera",
            "orbit_center_world_m": center.tolist(),
            "orbit_radius_m": float(radius),
            "wrist_center_spacing_m": a.hand_spacing,
            "model_material_changes": False,
            "poses": "Recorded PhysX rigid-body link transforms",
            "source_assets": "Bundled RevoSim left and right hand USDs",
        }
        (a.output / "render.json").write_text(json.dumps(report, indent=2) + "\n")
    rgb.detach(product)
    product.destroy()
    print("RENDER_COMPLETE", flush=True)


def main():
    from isaaclab.app import AppLauncher
    from revosim.launch import configure

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "simulate", "render"), default="all")
    parser.add_argument("--output", type=Path, default=Path("outputs/orbit"))
    parser.add_argument("--duration", type=float, default=18)
    parser.add_argument(
        "--motion-speed",
        type=float,
        default=1.5,
        help="Scale the motor sequence before simulation; 18 / 1.5 gives 12 seconds",
    )
    parser.add_argument("--peace-sway-deg", type=float, default=12.0)
    parser.add_argument("--fps", type=int, choices=(30, 60), default=60)
    parser.add_argument("--hand-spacing", type=float, default=0.16)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--preview", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    a = parser.parse_args()
    if a.duration <= 0 or a.motion_speed <= 0 or a.hand_spacing < 0.14:
        parser.error("duration/speed must be positive and hand spacing at least 0.14 m")
    if not 0 <= a.peace_sway_deg <= 13:
        parser.error(
            "peace sway must be between 0 and 13 degrees to respect joint limits"
        )
    a.output.mkdir(parents=True, exist_ok=True)
    if a.phase == "all":
        args = sys.argv[1:]
        if "--phase" in args:
            i = args.index("--phase")
            args = args[:i] + args[i + 2 :]
        args = [arg for arg in args if not arg.startswith("--phase=")]
        for phase in ("simulate", "render"):
            subprocess.run(
                [sys.executable, __file__, *args, "--phase", phase], check=True
            )
        return
    a.enable_cameras = a.phase == "render"
    app = AppLauncher(
        configure(a),
        multi_gpu=False,
        renderer="PathTracing" if a.phase == "render" else "RaytracedLighting",
    ).app
    try:
        simulate(a) if a.phase == "simulate" else render(a, app)
    except BaseException:
        import traceback

        traceback.print_exc()
        os._exit(1)
    finally:
        app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
