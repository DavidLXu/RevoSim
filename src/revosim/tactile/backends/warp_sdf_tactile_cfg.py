# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab.sensors.sensor_base_cfg import SensorBaseCfg
from .warp_sdf_tactile_sensor import WarpSdfTactileSensor


@configclass
class WarpSdfTactileSensorCfg(SensorBaseCfg):
    """Configuration for :class:`WarpSdfTactileSensor`.

    This is a minimal Warp-based tactile sensor intended for testing a VT-refine-like
    taxel grid pipeline with an analytic oriented-box SDF.

    Output matches VT-refine's `tactile_points_w` contract: (E, S*P, 4) per step.
    """

    class_type: type = WarpSdfTactileSensor

    # Sensor attachment
    elastomer_prim_paths: list[str] = list()

    # Taxel grid
    num_rows: int = 12
    num_cols: int = 32
    point_distance: float = 0.002
    row_distance: float | None = None
    col_distance: float | None = None
    normal_axis: int = 0
    normal_offset: float = -0.0032
    normal_sign: float | None = None

    # Optional explicit taxels in the attached link frame. When provided, these
    # replace the regular grid, ignore patch placement, and use num_rows*num_cols as map shape.
    taxel_points_l: list[tuple[float, float, float]] | None = None
    taxel_normals_l: list[tuple[float, float, float]] | None = None

    # Optional patch pose in body frame
    patch_offset_pos_b: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Quaternion is wxyz.
    patch_offset_quat_b: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    patch_offset_pos_b_per_elastomer: list[tuple[float, float, float]] | None = None
    patch_offset_quat_b_per_elastomer: list[tuple[float, float, float, float]] | None = None

    # Analytic target SDF: oriented box in world frame
    box_pos_w: tuple[float, float, float] = (0.6, 0.0, 0.55)
    box_quat_w: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    box_half_extents: tuple[float, float, float] = (0.04, 0.04, 0.04)

    # If `target_mesh_prim_path` is provided, the sensor will query distance to the mesh surface using Warp.
    # - Signed distance requires a watertight mesh (Warp provides `sign` for closed meshes).
    # - If `mesh_use_signed_distance` is False, `mesh_unsigned_shell_as_contact=True` makes a surface-gap model:
    #   pressure = max(mesh_shell_thickness - gap_to_surface, 0). The default gap is measured along each
    #   taxel normal so side-near geometry does not trigger pressure as easily as nearest-point distance.
    target_mesh_prim_path: str | None = None
    mesh_max_dist: float = 0.20
    mesh_use_signed_distance: bool = False
    mesh_signed_distance_method: str = "winding"
    # When `mesh_signed_distance_method == "normal"`, use vertex-normal interpolation (Phong-like)
    # instead of flat triangle normals.
    mesh_smooth_normals: bool = True
    mesh_shell_thickness: float = 0.0015
    mesh_unsigned_shell_as_contact: bool = True
    mesh_unsigned_contact_mode: str = "normal_ray"  # "normal_ray" or legacy "nearest"
    mesh_normal_ray_start_offset: float = 1.0e-6
    penetration_deadband: float = 0.0
    # "penetration_kv" keeps the historical spring/damper law. "gap_fraction"
    # uses surface-gap closure / mesh_shell_thickness as pressure directly.
    pressure_response_model: str = "auto"

    # Kelvin-Voigt-compatible law: fn = clamp(k * penetration + c * penetration_velocity, 0, max_force)
    stiffness: float = 5_000.0
    damping: float = 0.0
    max_force: float = 2.4
    pressure_damping_dt: float = 1.0 / 120.0

    # Calibrated pressure response applied after penetration is computed.
    pressure_gain: float = 1.0
    pressure_bias: float = 0.0
    pressure_gamma: float = 1.0
    pressure_threshold: float = 0.0
    taxel_area: float = 1.0

    # Normalize output fn into [0,1] by dividing by max_force
    normalize_forces: bool = True

    # Keep intermediate tensors such as SDF, penetration, raw force, world taxel points, and normals.
    # Disable for RL training when only the final pressure_force_map observation is needed.
    store_debug_fields: bool = True

    # Debug visualization (only used when SensorBaseCfg.debug_vis=True)
    debug_vis_env_id: int = 0
    debug_vis_point_radius: float = 0.002
    debug_vis_force_threshold: float = 1.0e-6
    debug_vis_show_all_taxels: bool = False
    debug_vis_show_axes: bool = False
    debug_vis_axes_scale: float = 0.05
