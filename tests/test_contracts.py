import ast
import numpy as np
from revosim.resources import ASSETS, pressure_regions, sensor_paths


def test_pressure_layout_covers_exactly_247_channels():
    import xml.etree.ElementTree as E

    regions = pressure_regions()
    assert sum(s.stop - s.start for s in regions.values()) == 247
    assert regions["palm"].stop - regions["palm"].start == 117
    for side in ["left", "right"]:
        p = sensor_paths(side)["pressure"]
        pads = E.parse(p).getroot().findall("link/pressure_pad")
        assert [int(x.get("taxel_count")) for x in pads] == [
            s.stop - s.start for s in regions.values()
        ]
        for pad in pads:
            points = np.load(p.parent / pad.get("points_npy"))
            normals = np.load(p.parent / pad.get("normals_npy"))
            assert points.shape == normals.shape == (int(pad.get("taxel_count")), 3)
            assert np.isfinite(points).all()
            assert np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-5)


def test_package_has_no_runtime_revobench_imports():
    root = ASSETS.parent
    for file in root.rglob("*.py"):
        tree = ast.parse(file.read_text())
        for node in ast.walk(tree):
            modules = (
                [x.name for x in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert not any(
                m.split(".")[0]
                in [
                    "revobench_assets",
                    "brainco_benchmark",
                    "envs",
                    "policy",
                    "scripts",
                ]
                for m in modules
            ), (file, modules)


def test_optical_resources_have_masks_and_original_marker_order():
    for side in ["left", "right"]:
        with np.load(sensor_paths(side)["vtac"] / "marker_positions.npz") as d:
            for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                assert d[finger + "_points_link_m"].shape == (100, 3)
                valid = (
                    d["distortion_valid"]
                    & (d[finger + "_method"] == "ray_hit")
                    & d[finger + "_marker_ray_valid"]
                )
                assert valid.sum() > 20
