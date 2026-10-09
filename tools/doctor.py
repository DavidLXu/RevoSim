#!/usr/bin/env python3
"""Check the environment and bundled resources without starting Isaac Sim."""

import hashlib, importlib.metadata, importlib.util, json, sys, tomllib
from pathlib import Path
from revosim.resources import ASSETS, sensor_paths

assert sys.version_info[:2] == (3, 11), "RevoSim targets Python 3.11 / Isaac Sim 5.1"
for name in [
    "isaacsim",
    "isaaclab",
    "torch",
    "warp-lang",
    "numpy",
    "scipy",
    "trimesh",
    "h5py",
    "imageio-ffmpeg",
]:
    print(name, importlib.metadata.version(name))
lab_source = Path(importlib.util.find_spec("isaaclab").origin).parents[1]
lab_config = tomllib.loads((lab_source / "config/extension.toml").read_text())
print(
    "Isaac Lab imported source:",
    lab_source,
    "version:",
    lab_config["package"]["version"],
)
manifest = json.loads((ASSETS / "provenance.json").read_text())
for rel, sha in manifest["files"].items():
    path = ASSETS / rel
    assert path.is_file(), f"Missing bundled asset {rel}"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, f"Asset changed: {rel}"
print("RevoSim bundled assets:", ASSETS)
for side in ["left", "right"]:
    for key, path in sensor_paths(side).items():
        assert path.exists(), f"{key}: {path}"
import torch

assert torch.cuda.is_available(), "CUDA GPU unavailable"
print(
    "GPUs:",
    [(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())],
)
print("PASS: bundled assets intact; CUDA and dependencies available")
