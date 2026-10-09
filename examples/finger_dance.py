#!/usr/bin/env python3
"""A 15-second PhysX-driven finger dance with a complete studio camera orbit."""

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import orbit_demo


FINGERS = ("index", "middle", "ring", "little")
FLEXION = (1.05, 1.12, 0.82)


def ease(x):
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (10 + x * (-15 + 6 * x))


def pulse(t, start, duration):
    """A smooth curl and release, with zero velocity and acceleration at its ends."""
    u = (t - start) / duration
    return float(ease(2 * u) if u <= 0.5 else 1 - ease(2 * u - 1))


def dance_target(names, side, fraction, limits, peace_sway_deg=12.0):
    ids = {name: i for i, name in enumerate(names)}
    t = 15.0 * fraction

    def pose(curled=(), thumb=0.0, spread=0.0):
        q = np.zeros(len(names), dtype=np.float32)
        for finger in curled:
            for joint, value in zip(("MCP", "PIP", "DIP"), FLEXION):
                q[ids[f"{side}_{finger}_{joint}_joint"]] = value
        for joint, value in zip(
            ("CMP", "CMR", "MCP", "PIP", "DIP"), (0.4, 0.1, 0.65, 0.8, 0.55)
        ):
            q[ids[f"{side}_thumb_{joint}_joint"]] = thumb * value
        q[ids[f"{side}_index_MPR_joint"]] = 0.24 * spread
        q[ids[f"{side}_middle_MPR_joint"]] = -0.10 * spread
        return q

    opened = pose()
    point = pose(("middle", "ring", "little"), thumb=0.72)
    peace = pose(("ring", "little"), thumb=0.85, spread=1)
    three = pose(("little",), thumb=0.65)
    horns = pose(("middle", "ring"), thumb=0.7)
    fist = pose(FINGERS, thumb=1)
    thumbs_up = pose(FINGERS, thumb=0)
    # A different ordering on each hand provides call-and-response at every beat.
    poses = (point, peace, three, fist) if side == "left" else (three, horns, point, peace)
    keys = [
        (0.0, opened), (3.0, opened),
        (3.72, poses[0]), (4.44, poses[1]),
        (5.16, poses[2]), (5.88, poses[3]),
        (6.60, peace), (8.12, peace), (8.90, opened),
        (11.65, opened), (12.38, thumbs_up),
        (13.12, peace if side == "left" else horns),
        (13.86, opened), (15.0, opened),
    ]
    target = opened.copy()
    for (t0, q0), (t1, q1) in zip(keys, keys[1:]):
        if t <= t1:
            target = q0 + (q1 - q0) * ease((t - t0) / (t1 - t0))
            break

    for i, finger in enumerate(FINGERS):
        rank = i if side == "left" else 3 - i
        lag = 0 if side == "left" else 0.12
        # Two waves travel in opposite directions, with the right hand off beat.
        curl = 0.75 * (
            pulse(t, 0.05 + rank * 0.10 + lag, 1.20)
            + pulse(t, 1.45 + (3 - rank) * 0.05 + lag, 1.20)
        )
        # Piano phrase: individual notes, then alternating pairs.
        curl += 0.63 * pulse(t, 8.95 + rank * 0.17 + lag, 0.98)
        pair_rank = i % 2 if side == "left" else 1 - i % 2
        curl += 0.63 * pulse(t, 10.32 + pair_rank * 0.30, 0.98)
        # Short, smaller ripples provide a continuous release into the loop seam.
        curl += 0.28 * pulse(t, 13.90 + rank * 0.065, 0.68)
        for joint, value in zip(("MCP", "PIP", "DIP"), FLEXION):
            target[ids[f"{side}_{finger}_{joint}_joint"]] += curl * value

    if 6.60 <= t <= 8.12:
        u = (t - 6.60) / 1.52
        envelope = float(ease(min(u / 0.20, (1 - u) / 0.20, 1)))
        direction = 1 if side == "left" else -1
        sway = direction * math.radians(peace_sway_deg) * envelope * math.sin(4 * math.pi * u)
        for finger in ("index", "middle"):
            target[ids[f"{side}_{finger}_MPR_joint"]] += sway
        # Keep the V sign readable while the two curled fingers articulate in turn.
        for i, finger in enumerate(("ring", "little")):
            release = 0.22 * envelope * (0.5 + 0.5 * math.sin(4 * math.pi * u + i * math.pi))
            for joint, value in zip(("MCP", "PIP", "DIP"), FLEXION):
                target[ids[f"{side}_{finger}_{joint}_joint"]] -= release * value

    assert np.all(target >= limits[:, 0]) and np.all(target <= limits[:, 1]), (side, t)
    return target


PREVIEW_POSES = [
    ("wave", 0.75 / 15),
    ("point_and_three", 3.75 / 15),
    ("peace_and_horns", 4.48 / 15),
    ("relaxed_fist", 5.90 / 15),
    ("peace_sway", 7.20 / 15),
    ("piano", 9.60 / 15),
    ("thumbs_up", 12.40 / 15),
]


def main():
    from isaaclab.app import AppLauncher
    from revosim.launch import configure

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "simulate", "render"), default="all")
    parser.add_argument("--output", type=Path, default=Path("outputs/finger-dance"))
    parser.add_argument("--fps", type=int, choices=(30, 60), default=60)
    parser.add_argument("--hand-spacing", type=float, default=0.16)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--preview", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    a = parser.parse_args()
    a.duration, a.motion_speed, a.peace_sway_deg = 15.0, 1.0, 12.0
    a.output.mkdir(parents=True, exist_ok=True)
    if a.phase == "all":
        args = sys.argv[1:]
        if "--phase" in args:
            i = args.index("--phase")
            args = args[:i] + args[i + 2:]
        args = [arg for arg in args if not arg.startswith("--phase=")]
        for phase in ("simulate", "render"):
            subprocess.run([sys.executable, __file__, *args, "--phase", phase], check=True)
        return
    a.enable_cameras = a.phase == "render"
    app = AppLauncher(
        configure(a), multi_gpu=False,
        renderer="PathTracing" if a.phase == "render" else "RaytracedLighting",
    ).app
    try:
        if a.phase == "simulate":
            orbit_demo.simulate(a, target_fn=dance_target, motion_metadata={
                "motion": "Opposing finger waves, asymmetric gesture changes, V-sign sway, piano notes and pairs, thumbs-up and closing ripple",
                "finger_flexion_targets_rad": dict(zip(("MCP", "PIP", "DIP"), FLEXION)),
                "target_speed_limit_rad_s": 3.0,
                "sequence_duration_s": 15,
                "camera_phase_offset_deg": -70,
            })
        else:
            orbit_demo.render(
                a, app, preview_poses=PREVIEW_POSES,
                video_name=f"revosim-finger-dance-{a.fps}fps.mp4", camera_offset_deg=-70,
            )
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)
    finally:
        app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
