from pathlib import Path

from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils import configclass

from .tactile_sensor import TactileSensor

from revosim.resources import tactile_root
ASSET_ROOT = tactile_root()


@configclass
class TactileSensorCfg(SensorBaseCfg):
    """Native InteractiveScene sensor config. Values are copied for each instance.

    Enable ``robot.spawn.activate_contact_sensors`` before spawning the robot.
    Targets must be one rigid object per environment with identical mesh topology.
    Resource creation happens on PLAY; getters alone trigger per-step evaluation.
    """
    class_type: type = TactileSensor
    target_prim_paths: tuple[str, ...] = ('{ENV_REGEX_NS}/Object',)
    hands: tuple[str, ...] = ('left', 'right')
    pressure: bool = True
    depth: bool = True
    marker: bool = True
    rgb: bool = True
    asset_root: str = str(ASSET_ROOT)
    marker_history: str = 'on_demand'  # or 'continuous': updates marker each update()
    pressure_diffusion: bool = True
    object_sample_count: int = 32768
    object_sample_mode: str = 'poisson'
    update_period: float = 0.0


def make_tactile_cfg(*, robot_prim_path='{ENV_REGEX_NS}/Robot', target_prim_paths=('{ENV_REGEX_NS}/Object',),
                     hands=('left', 'right'), pressure=True, depth=True, marker=True, rgb=True, **kwargs):
    """Build a sensor config to assign to ``scene_cfg.tactile``."""
    return TactileSensorCfg(prim_path=robot_prim_path, target_prim_paths=tuple(target_prim_paths),
                           hands=tuple(hands), pressure=pressure, depth=depth, marker=marker, rgb=rgb, **kwargs)
