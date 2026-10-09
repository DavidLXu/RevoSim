"""Paths and sensor contracts contained entirely in the installed package."""

from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"
SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "little")
BACKEND_ORDER = (4, 1, 0, 2, 3)
PRESSURE_ORDER = (
    "middle_mcp",
    "middle_pip",
    "index_mcp",
    "index_pip",
    "ring_mcp",
    "ring_pip",
    "little_mcp",
    "little_pip",
    "thumb_mcp",
    "thumb_pip",
    "palm",
)


def tactile_root():
    return ASSETS / "sensors"


def sensor_paths(side, root=None):
    if side not in SIDES:
        raise ValueError(side)
    root = Path(root) if root else tactile_root()
    return dict(
        mapping=root / "bindings/revo3.json",
        pressure=root / f"pressure/revo3/{side}/layout.xml",
        vtac=root / f"vtac/registered/{side}",
        defaults=root / "vtac/revo3/v1/surface_defaults.json",
        background=root / "vtac/revo3/v1/reference_median.png",
        surface=root / f"vtac/revo3/v1/{side}",
    )


def pressure_regions():
    result = {}
    offset = 0
    for name, count in zip(PRESSURE_ORDER, (23, 4) * 4 + (18, 4, 117)):
        result[name] = slice(offset, offset + count)
        offset += count
    return result


def model_manifest():
    return {
        "pressure_presets": {
            "sample_offset_m": 0.001,
            "diffusion": {
                "sigma_m": 0.003,
                "blend": 0.7,
                "radius_sigma": 3.0,
                "normal_power": 1.0,
            },
        },
        "vtac_presets": {
            "depth_ray_shape": [32, 24],
            "reference_shape": [240, 320],
            "marker_shape": [10, 10],
            "local_shape": [25, 40],
            "contact_threshold_m": 2e-5,
            "roi_margin_m": 0.001,
            "penetration_deadband_m": 1e-6,
            "max_distance_m": 0.05,
            "object_sample_roi_count": 168,
        },
    }
