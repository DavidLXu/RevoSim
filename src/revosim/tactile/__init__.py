"""Four independently readable, lazy Revo tactile observations."""
from .tactile_cfg import TactileSensorCfg, make_tactile_cfg
from .tactile_sensor import TactileSensor

__all__ = ['TactileSensor', 'TactileSensorCfg', 'make_tactile_cfg']
