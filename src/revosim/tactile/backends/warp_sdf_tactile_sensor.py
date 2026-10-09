# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: ignore

from collections.abc import Sequence

import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.sensors.contact_sensor import ContactSensor, ContactSensorCfg
from isaaclab.sensors.sensor_base import SensorBase
from .pressure_taxel_map import (
    PressureCalibration,
    PressureTaxelMap,
    calibrate_gap_fraction,
    calibrate_penetration,
    coerce_taxel_calibration_value,
)
from .warp_sdf_tactile_data import WarpSdfTactileSensorData

import warp as wp  # type: ignore

@wp.func
def _query_mesh_distance(
    q: wp.vec3,
    mesh: wp.uint64,
    tri_indices: wp.array(dtype=wp.int32),
    vertex_normals: wp.array(dtype=wp.vec3),
    max_dist: float,
    signed_mode: int,
    smooth_normals: int,
) -> float:
    sign = float(0.0)
    face_idx = int(0)
    face_u = float(0.0)
    face_v = float(0.0)

    if signed_mode == 1:
        hit = wp.mesh_query_point_sign_winding_number(mesh, q, max_dist, sign, face_idx, face_u, face_v)
    else:
        hit = wp.mesh_query_point(mesh, q, max_dist, sign, face_idx, face_u, face_v)
    if hit:
        p = wp.mesh_eval_position(mesh, face_idx, face_u, face_v)
        delta = q - p
        d = wp.length(delta)

        if signed_mode == 0:
            # unsigned distance
            return d
        elif signed_mode == 1:
            # Warp's winding-number based sign (requires watertight mesh and winding support on wp.Mesh).
            return sign * d
        else:
            # Normal-based sign (works for open meshes but depends on consistent normals)
            # Compute triangle normal (flat)
            p0 = wp.mesh_eval_position(mesh, face_idx, 0.0, 0.0)
            p1 = wp.mesh_eval_position(mesh, face_idx, 1.0, 0.0)
            p2 = wp.mesh_eval_position(mesh, face_idx, 0.0, 1.0)
            n_face = wp.cross(p1 - p0, p2 - p0)
            n_face_len = wp.length(n_face)
            if n_face_len > 1.0e-12:
                n_face = n_face / n_face_len
            else:
                n_face = wp.vec3(0.0, 0.0, 1.0)

            n = n_face
            if smooth_normals == 1:
                # Interpolate vertex normals using barycentric coords
                base = face_idx * 3
                i0 = tri_indices[base + 0]
                i1 = tri_indices[base + 1]
                i2 = tri_indices[base + 2]
                w0 = 1.0 - face_u - face_v
                w1 = face_u
                w2 = face_v
                n_interp = w0 * vertex_normals[i0] + w1 * vertex_normals[i1] + w2 * vertex_normals[i2]
                n_len = wp.length(n_interp)
                if n_len > 1.0e-12:
                    n_interp = n_interp / n_len
                else:
                    n_interp = n_face
                # Align interpolated normal with the triangle's orientation
                if wp.dot(n_interp, n_face) < 0.0:
                    n_interp = -n_interp
                n = n_interp

            sd = wp.dot(delta, n)
            s = float(1.0)
            if sd < 0.0:
                s = float(-1.0)
            return s * d
    else:
        return max_dist
    return max_dist


@wp.kernel(enable_backward=False)
def mesh_distance_kernel(
    queries_l: wp.array(dtype=wp.vec3),
    mesh: wp.uint64,
    tri_indices: wp.array(dtype=wp.int32),
    vertex_normals: wp.array(dtype=wp.vec3),
    max_dist: float,
    signed_mode: int,
    smooth_normals: int,
    dist_out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    dist_out[tid] = _query_mesh_distance(
        queries_l[tid],
        mesh,
        tri_indices,
        vertex_normals,
        max_dist,
        signed_mode,
        smooth_normals,
    )


@wp.func
def _quat_rotate_inv(q: wp.vec4, v: wp.vec3) -> wp.vec3:
    # q is (w, x, y, z). Compute inverse rotation by using conjugate.
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    # conjugate
    cx = -qx
    cy = -qy
    cz = -qz
    # quat * v
    # treat v as pure quaternion (0, v)
    tx = qw * v[0] + cy * v[2] - cz * v[1]
    ty = qw * v[1] + cz * v[0] - cx * v[2]
    tz = qw * v[2] + cx * v[1] - cy * v[0]
    tw = -cx * v[0] - cy * v[1] - cz * v[2]
    # result = (conj(q) * v) * q
    rx = tw * qx + tx * qw + ty * qz - tz * qy
    ry = tw * qy - tx * qz + ty * qw + tz * qx
    rz = tw * qz + tx * qy - ty * qx + tz * qw
    return wp.vec3(rx, ry, rz)


@wp.func
def _quat_rotate(q: wp.vec4, v: wp.vec3) -> wp.vec3:
    # q is (w, x, y, z).
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    # quat * v
    tx = qw * v[0] + qy * v[2] - qz * v[1]
    ty = qw * v[1] + qz * v[0] - qx * v[2]
    tz = qw * v[2] + qx * v[1] - qy * v[0]
    tw = -qx * v[0] - qy * v[1] - qz * v[2]
    # result = (q * v) * conj(q)
    rx = -tw * qx + tx * qw - ty * qz + tz * qy
    ry = -tw * qy + tx * qz + ty * qw - tz * qx
    rz = -tw * qz - tx * qy + ty * qx + tz * qw
    return wp.vec3(rx, ry, rz)


@wp.kernel(enable_backward=False)
def mesh_distance_batched_world_kernel(
    points_w: wp.array(dtype=wp.vec3),
    mesh_pos_w: wp.array(dtype=wp.vec3),
    mesh_quat_w: wp.array(dtype=wp.vec4),
    mesh_scale_w: wp.array(dtype=wp.vec3),
    mesh: wp.uint64,
    tri_indices: wp.array(dtype=wp.int32),
    vertex_normals: wp.array(dtype=wp.vec3),
    num_points: int,
    use_mesh_scale: int,
    max_dist: float,
    signed_mode: int,
    smooth_normals: int,
    dist_out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    env_i = tid // num_points
    q_l = _quat_rotate_inv(mesh_quat_w[env_i], points_w[tid] - mesh_pos_w[env_i])
    if use_mesh_scale == 1:
        scale = mesh_scale_w[0]
        q_l = wp.vec3(q_l[0] / scale[0], q_l[1] / scale[1], q_l[2] / scale[2])
    dist_out[tid] = _query_mesh_distance(
        q_l,
        mesh,
        tri_indices,
        vertex_normals,
        max_dist,
        signed_mode,
        smooth_normals,
    )


@wp.kernel(enable_backward=False)
def mesh_normal_gap_batched_world_kernel(
    points_w: wp.array(dtype=wp.vec3),
    normals_w: wp.array(dtype=wp.vec3),
    mesh_pos_w: wp.array(dtype=wp.vec3),
    mesh_quat_w: wp.array(dtype=wp.vec4),
    mesh_scale_w: wp.array(dtype=wp.vec3),
    mesh: wp.uint64,
    num_points: int,
    use_mesh_scale: int,
    max_dist: float,
    ray_start_offset: float,
    gap_out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    env_i = tid // num_points

    n_w = normals_w[tid]
    n_len = wp.length(n_w)
    if n_len <= 1.0e-12:
        gap_out[tid] = max_dist
        return
    n_w = n_w / n_len

    start_w = points_w[tid] - ray_start_offset * n_w
    start_l = _quat_rotate_inv(mesh_quat_w[env_i], start_w - mesh_pos_w[env_i])
    dir_l = _quat_rotate_inv(mesh_quat_w[env_i], n_w)
    if use_mesh_scale == 1:
        scale = mesh_scale_w[0]
        start_l = wp.vec3(start_l[0] / scale[0], start_l[1] / scale[1], start_l[2] / scale[2])
        dir_l = wp.vec3(dir_l[0] / scale[0], dir_l[1] / scale[1], dir_l[2] / scale[2])

    dir_len = wp.length(dir_l)
    if dir_len <= 1.0e-12:
        gap_out[tid] = max_dist
        return
    dir_l = dir_l / dir_len

    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3()
    face = int(0)
    hit = wp.mesh_query_ray(mesh, start_l, dir_l, max_dist * dir_len, t, u, v, sign, normal, face)
    if hit:
        delta_l = t * dir_l
        if use_mesh_scale == 1:
            scale = mesh_scale_w[0]
            delta_l = wp.vec3(delta_l[0] * scale[0], delta_l[1] * scale[1], delta_l[2] * scale[2])
        delta_w = _quat_rotate(mesh_quat_w[env_i], delta_l)
        gap = wp.dot(delta_w, n_w) - ray_start_offset
        gap_out[tid] = wp.max(gap, 0.0)
    else:
        gap_out[tid] = max_dist


@wp.kernel(enable_backward=False)
def box_sdf_kernel(
    points_w: wp.array(dtype=wp.vec3),
    box_pos_w: wp.vec3,
    box_quat_w: wp.vec4,
    half_extents: wp.vec3,
    sdf_out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    p_w = points_w[tid]
    # transform point into box local frame
    p_l = _quat_rotate_inv(box_quat_w, p_w - box_pos_w)

    qx = wp.abs(p_l.x) - half_extents.x
    qy = wp.abs(p_l.y) - half_extents.y
    qz = wp.abs(p_l.z) - half_extents.z

    # outside distance
    ox = wp.max(qx, 0.0)
    oy = wp.max(qy, 0.0)
    oz = wp.max(qz, 0.0)
    outside = wp.sqrt(ox * ox + oy * oy + oz * oz)

    # inside distance (negative)
    m = wp.max(qx, qy)
    m = wp.max(m, qz)
    inside = wp.min(m, 0.0)

    sdf_out[tid] = outside + inside


def _calibration_all(value, *, op: str) -> bool:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all().item()):
        return False
    if op == "positive":
        return bool((tensor > 0.0).all().item())
    if op == "nonnegative":
        return bool((tensor >= 0.0).all().item())
    raise ValueError(f"unknown calibration check {op!r}")


class WarpSdfTactileSensor(SensorBase):
    """Warp-based tactile sensor using an analytic oriented-box SDF.

    This is a test-oriented sensor to validate a VT-refine-like taxel pipeline in Isaac Lab
    without relying on PhysX contact forces.

    Output tensors match the VT-refine contract: `tactile_points_w` with [x,y,z,fn].
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        if not getattr(self.cfg, "elastomer_prim_paths", None):
            raise ValueError("'elastomer_prim_paths' must be a non-empty list")

        if self.cfg.num_rows <= 0 or self.cfg.num_cols <= 0:
            raise ValueError("'num_rows' and 'num_cols' must be positive")
        explicit_points = getattr(self.cfg, "taxel_points_l", None)
        explicit_normals = getattr(self.cfg, "taxel_normals_l", None)
        if (explicit_points is None) != (explicit_normals is None):
            raise ValueError("'taxel_points_l' and 'taxel_normals_l' must be provided together")
        if explicit_points is None:
            self._row_distance, self._col_distance = self._resolve_grid_distances()
        else:
            self._row_distance = self._col_distance = 0.0
        self._normal_sign = self._resolve_normal_sign()
        if self.cfg.normal_axis not in (0, 1, 2):
            raise ValueError("'normal_axis' must be one of 0, 1, 2")

        if not _calibration_all(self.cfg.stiffness, op="positive"):
            raise ValueError("'stiffness' must be positive")
        if not _calibration_all(getattr(self.cfg, "damping", 0.0), op="nonnegative"):
            raise ValueError("'damping' must be non-negative")
        if not _calibration_all(self.cfg.max_force, op="positive"):
            raise ValueError("'max_force' must be positive")

        self._data = WarpSdfTactileSensorData()

        self._points_local_per_sensor: torch.Tensor | None = None
        self._pressure_taxel_maps: list[PressureTaxelMap] = []
        self._num_points: int = 0
        self._prev_penetration: torch.Tensor | None = None
        self._has_prev_penetration: torch.Tensor | None = None

        # Track pose via lightweight contact sensors (pose only).
        self._pose_sensors: list[ContactSensor] = []
        for elastomer_prim_path in self.cfg.elastomer_prim_paths:
            pose_cfg = ContactSensorCfg(
                prim_path=elastomer_prim_path,
                update_period=self.cfg.update_period,
                history_length=0,
                debug_vis=False,
                track_pose=True,
                track_contact_points=False,
                filter_prim_paths_expr=[],
            )
            self._pose_sensors.append(ContactSensor(pose_cfg))

        self._num_sensors = len(self._pose_sensors)

        # Box pose (world). Initialized on simulator PLAY when device is available.
        self._box_pos_w: torch.Tensor | None = None
        self._box_quat_w: torch.Tensor | None = None
        self._box_half_extents: torch.Tensor | None = None

        # Optional mesh target (loaded from USD).
        self._target_mesh_prim_path: str | None = getattr(self.cfg, "target_mesh_prim_path", None)
        self._resolved_target_mesh_prim_path: str | None = None
        self._wp_mesh: wp.Mesh | None = None
        self._wp_mesh_tri_indices: wp.array | None = None
        self._wp_mesh_vertex_normals: wp.array | None = None
        self._mesh_pos_w: torch.Tensor | None = None
        self._mesh_quat_w: torch.Tensor | None = None
        self._mesh_scale_w: torch.Tensor | None = None

        # Warp device (initialized on PLAY).
        self._wp_device: str | None = None

        # Reusable buffers for warp outputs
        self._sdf_out: torch.Tensor | None = None

        # Debug visualization
        self._debug_markers: VisualizationMarkers | None = None
        self._debug_axes: VisualizationMarkers | None = None

    @property
    def data(self) -> WarpSdfTactileSensorData:
        self._update_outdated_buffers()
        return self._data

    @property
    def pressure_taxel_maps(self) -> tuple[PressureTaxelMap, ...]:
        """Calibrated taxel layouts used to produce pressure force maps."""

        return tuple(self._pressure_taxel_maps)

    def set_box_pose(
        self,
        pos_w: tuple[float, float, float] | torch.Tensor,
        quat_w: tuple[float, float, float, float] | torch.Tensor,
    ):
        """Update the target box pose in world frame."""
        device = self.device
        if isinstance(pos_w, torch.Tensor):
            self._box_pos_w = pos_w.to(device=device, dtype=torch.float32)
        else:
            self._box_pos_w = torch.tensor(pos_w, device=device, dtype=torch.float32)

        if isinstance(quat_w, torch.Tensor):
            self._box_quat_w = quat_w.to(device=device, dtype=torch.float32)
        else:
            self._box_quat_w = torch.tensor(quat_w, device=device, dtype=torch.float32)

    def set_target_pose(
        self,
        pos_w: tuple[float, float, float] | torch.Tensor,
        quat_w: tuple[float, float, float, float] | torch.Tensor,
    ):
        """Update the target mesh pose in world frame (used when `target_mesh_prim_path` is set)."""
        device = self.device
        if isinstance(pos_w, torch.Tensor):
            self._mesh_pos_w = pos_w.to(device=device, dtype=torch.float32)
        else:
            self._mesh_pos_w = torch.tensor(pos_w, device=device, dtype=torch.float32)

        if isinstance(quat_w, torch.Tensor):
            self._mesh_quat_w = quat_w.to(device=device, dtype=torch.float32)
        else:
            self._mesh_quat_w = torch.tensor(quat_w, device=device, dtype=torch.float32)

        # Cache mesh scale once if not available. This is important for scaled USD meshes
        # (e.g., unit cube mesh with a scale xform to represent size), otherwise distances are wrong.
        mesh_prim_path = self._resolved_target_mesh_prim_path or self._target_mesh_prim_path
        if self._mesh_scale_w is None and mesh_prim_path is not None:
            try:
                prim = self.stage.GetPrimAtPath(mesh_prim_path)
                if prim.IsValid():
                    sx, sy, sz = sim_utils.resolve_prim_scale(prim)
                    self._mesh_scale_w = torch.tensor((sx, sy, sz), device=device, dtype=torch.float32)
            except (ValueError, RuntimeError):
                pass

    def query_sdf_world(self, points_w: torch.Tensor) -> torch.Tensor:
        """Query the current target SDF at world-frame points.

        Args:
            points_w: World-frame query points with shape ``(P, 3)`` or ``(E, P, 3)``.

        Returns:
            Signed or configured distance values with shape ``(E, P)``.
        """
        if self._wp_device is None:
            raise RuntimeError("WarpSdfTactileSensor is not initialized yet.")

        points_w = points_w.to(device=self._device, dtype=torch.float32)
        if points_w.ndim == 2:
            points_w = points_w.unsqueeze(0)
        if points_w.ndim != 3 or points_w.shape[-1] != 3:
            raise ValueError(f"Expected points_w shape (P, 3) or (E, P, 3), got {tuple(points_w.shape)}")

        num_envs = int(points_w.shape[0])
        num_points = int(points_w.shape[1])
        sdf = torch.empty((num_envs, num_points), device=self._device, dtype=torch.float32)

        use_mesh = self._target_mesh_prim_path is not None and self._wp_mesh is not None
        if use_mesh:
            if self._mesh_pos_w is None or self._mesh_quat_w is None:
                raise RuntimeError("Target mesh pose is not initialized.")
            if self._wp_mesh_tri_indices is None or self._wp_mesh_vertex_normals is None:
                raise RuntimeError("Target mesh auxiliary buffers are not initialized.")

            max_dist = float(getattr(self.cfg, "mesh_max_dist", 0.20))
            use_signed = bool(getattr(self.cfg, "mesh_use_signed_distance", False))
            signed_method = str(getattr(self.cfg, "mesh_signed_distance_method", "winding")).lower()
            if not use_signed:
                signed_mode = 0
            else:
                signed_mode = 2 if signed_method == "normal" else 1
            smooth_normals = int(bool(getattr(self.cfg, "mesh_smooth_normals", True)))

            mesh_pos_w = self._expand_box_param(self._mesh_pos_w, num_envs, 3, "mesh_pos_w")
            mesh_quat_w = self._expand_box_param(self._mesh_quat_w, num_envs, 4, "mesh_quat_w")

            for env_i in range(num_envs):
                pts_l = quat_apply_inverse(
                    mesh_quat_w[env_i].unsqueeze(0).expand(num_points, -1),
                    points_w[env_i] - mesh_pos_w[env_i],
                )
                if self._mesh_scale_w is not None:
                    scale = self._mesh_scale_w.to(device=pts_l.device, dtype=pts_l.dtype).clamp(min=1.0e-12)
                    pts_l = pts_l / scale.unsqueeze(0)
                pts_l = pts_l.contiguous()
                sdf_out = sdf[env_i].contiguous()

                wp.launch(
                    kernel=mesh_distance_kernel,
                    dim=num_points,
                    inputs=[
                        wp.from_torch(pts_l, dtype=wp.vec3),
                        self._wp_mesh.id,
                        self._wp_mesh_tri_indices,
                        self._wp_mesh_vertex_normals,
                        float(max_dist),
                        int(signed_mode),
                        int(smooth_normals),
                        wp.from_torch(sdf_out, dtype=wp.float32),
                    ],
                    device=self._wp_device,
                )
        else:
            if self._box_pos_w is None or self._box_quat_w is None or self._box_half_extents is None:
                raise RuntimeError("Analytic box target pose is not initialized.")

            sdf[:] = self._box_sdf_torch(points_w, self._box_pos_w, self._box_quat_w, self._box_half_extents)

        return sdf

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        for s in self._pose_sensors:
            s.reset(env_ids)

        resolved_env_ids = slice(None) if env_ids is None else env_ids
        if self._data.tactile_points_w is not None:
            self._data.tactile_points_w[resolved_env_ids] = 0.0
        if self._data.tactile_points_w_per_sensor is not None:
            self._data.tactile_points_w_per_sensor[resolved_env_ids] = 0.0
        if self._data.pressure_force_raw_per_sensor is not None:
            self._data.pressure_force_raw_per_sensor[resolved_env_ids] = 0.0
        if self._data.pressure_force_map_raw is not None:
            self._data.pressure_force_map_raw[resolved_env_ids] = 0.0
        if self._data.pressure_force_map is not None:
            self._data.pressure_force_map[resolved_env_ids] = 0.0
        if self._data.signed_distance_map is not None:
            self._data.signed_distance_map[resolved_env_ids] = 0.0
        if self._data.penetration_map is not None:
            self._data.penetration_map[resolved_env_ids] = 0.0
        if self._data.penetration_velocity_map is not None:
            self._data.penetration_velocity_map[resolved_env_ids] = 0.0
        if self._data.taxel_normals_w_per_sensor is not None:
            self._data.taxel_normals_w_per_sensor[resolved_env_ids] = 0.0
        if self._prev_penetration is not None:
            self._prev_penetration[resolved_env_ids] = 0.0
        if self._has_prev_penetration is not None:
            self._has_prev_penetration[resolved_env_ids] = False

    def _initialize_impl(self):
        super()._initialize_impl()

        # Initialize box tensors and warp device now that SensorBase has a valid device.
        if self._box_pos_w is None:
            self._box_pos_w = torch.tensor(self.cfg.box_pos_w, device=self.device, dtype=torch.float32)
        if self._box_quat_w is None:
            self._box_quat_w = torch.tensor(self.cfg.box_quat_w, device=self.device, dtype=torch.float32)
        if self._box_half_extents is None:
            self._box_half_extents = torch.tensor(self.cfg.box_half_extents, device=self.device, dtype=torch.float32)
        if self._wp_device is None:
            dev = self.device
            # Isaac Lab uses device strings like "cuda:0"/"cpu".
            if dev.startswith("cuda"):
                self._wp_device = dev if ":" in dev else "cuda:0"
            else:
                self._wp_device = "cpu"

        # If a target mesh prim path is provided, load the USD mesh and build a Warp mesh (triangles).
        if self._target_mesh_prim_path is not None and self._wp_mesh is None:
            assert self._wp_device is not None
            mesh_prim_path = self._resolve_target_mesh_prim_path(self._target_mesh_prim_path)
            self._resolved_target_mesh_prim_path = mesh_prim_path
            (
                self._wp_mesh,
                self._wp_mesh_tri_indices,
                self._wp_mesh_vertex_normals,
            ) = self._load_warp_mesh_from_usd(mesh_prim_path, device=self._wp_device)

            # Initialize target pose from the USD prim.
            try:
                prim = self.stage.GetPrimAtPath(mesh_prim_path)
                if prim.IsValid():
                    pos, quat = sim_utils.resolve_prim_pose(prim)
                    self.set_target_pose(pos, quat)
                    # cache scale for correct world->local transform when querying distances
                    sx, sy, sz = sim_utils.resolve_prim_scale(prim)
                    self._mesh_scale_w = torch.tensor((sx, sy, sz), device=self.device, dtype=torch.float32)
            except (ValueError, RuntimeError):
                # If pose resolution fails, user can still drive pose via `set_target_pose`.
                pass

        explicit_points = getattr(self.cfg, "taxel_points_l", None)
        explicit_normals = getattr(self.cfg, "taxel_normals_l", None)
        if (explicit_points is None) != (explicit_normals is None):
            raise ValueError("'taxel_points_l' and 'taxel_normals_l' must be provided together")
        if explicit_points is None:
            base_points_local = self._create_local_grid_points(
                num_rows=self.cfg.num_rows,
                num_cols=self.cfg.num_cols,
                point_distance=self.cfg.point_distance,
                row_distance=self._row_distance,
                col_distance=self._col_distance,
                normal_axis=self.cfg.normal_axis,
                normal_offset=self.cfg.normal_offset,
                device=self._device,
            )
            base_normal = torch.zeros_like(base_points_local)
            base_normal[:, int(self.cfg.normal_axis)] = self._normal_sign
        else:
            base_points_local = torch.as_tensor(explicit_points, device=self._device, dtype=torch.float32)
            base_normal = torch.as_tensor(explicit_normals, device=self._device, dtype=torch.float32)
            if base_points_local.ndim != 2 or base_points_local.shape[-1] != 3:
                raise ValueError(f"'taxel_points_l' must have shape (P, 3), got {base_points_local.shape}")
            if base_normal.shape != base_points_local.shape:
                raise ValueError(
                    f"'taxel_normals_l' must match taxel points shape, got {base_normal.shape} vs "
                    f"{base_points_local.shape}"
                )
            if not bool(torch.isfinite(base_points_local).all()) or not bool(torch.isfinite(base_normal).all()):
                raise ValueError("explicit taxel points and normals must be finite")
            normal_lengths = torch.linalg.norm(base_normal, dim=-1, keepdim=True)
            if bool(torch.any(normal_lengths <= 1.0e-8)):
                raise ValueError("explicit taxel normals must be non-zero")
            base_normal = base_normal / normal_lengths
        self._num_points = int(base_points_local.shape[0])
        expected_points = int(self.cfg.num_rows) * int(self.cfg.num_cols)
        if self._num_points != expected_points:
            raise ValueError(
                f"taxel layout has {self._num_points} points, but num_rows*num_cols is {expected_points}"
            )

        if explicit_points is None:
            offset_pos_list = self._resolve_patch_offset_pos_list()
            offset_quat_list = self._resolve_patch_offset_quat_list()
        else:
            offset_pos_list = [(0.0, 0.0, 0.0)] * len(self.cfg.elastomer_prim_paths)
            offset_quat_list = [(1.0, 0.0, 0.0, 0.0)] * len(self.cfg.elastomer_prim_paths)

        points_local_per_sensor = []
        taxel_maps = []
        image_shape = (int(self.cfg.num_rows), int(self.cfg.num_cols))
        calib = lambda name, default: coerce_taxel_calibration_value(
            getattr(self.cfg, name, default), image_shape, backend_like=base_points_local, field_name=name
        )
        pressure_calibration = PressureCalibration(
            gain=calib("pressure_gain", 1.0),
            bias=calib("pressure_bias", 0.0),
            stiffness=calib("stiffness", 5_000.0),
            damping=calib("damping", 0.0),
            max_force=calib("max_force", 10.0),
            gamma=calib("pressure_gamma", 1.0),
            threshold=calib("pressure_threshold", 0.0),
            area=calib("taxel_area", 1.0),
        )
        for elastomer_path, offset_pos, offset_quat in zip(
            self.cfg.elastomer_prim_paths, offset_pos_list, offset_quat_list, strict=True
        ):
            pos_b = torch.tensor(offset_pos, device=self._device, dtype=torch.float32)
            quat_b = torch.tensor(offset_quat, device=self._device, dtype=torch.float32)
            pts = quat_apply(quat_b.unsqueeze(0).expand(self._num_points, -1), base_points_local) + pos_b
            points_local_per_sensor.append(pts)
            normals = quat_apply(quat_b.unsqueeze(0).expand(self._num_points, -1), base_normal)
            taxel_maps.append(
                PressureTaxelMap(
                    link_name=str(elastomer_path),
                    points_l=pts,
                    normals_l=normals,
                    image_shape=(int(self.cfg.num_rows), int(self.cfg.num_cols)),
                    calibration=pressure_calibration,
                )
            )
        self._points_local_per_sensor = torch.stack(points_local_per_sensor, dim=0)  # (S, P, 3)
        self._pressure_taxel_maps = taxel_maps

        rows = int(self.cfg.num_rows)
        cols = int(self.cfg.num_cols)
        map_shape = (self._num_envs, self._num_sensors, rows, cols)
        self._data.pressure_force_map = torch.zeros(map_shape, device=self._device, dtype=torch.float32)

        store_debug_fields = bool(getattr(self.cfg, "store_debug_fields", True)) or bool(self.cfg.debug_vis)
        if store_debug_fields:
            self._data.tactile_points_w_per_sensor = torch.zeros(
                (self._num_envs, self._num_sensors, self._num_points, 4), device=self._device, dtype=torch.float32
            )
            self._data.tactile_points_w = torch.zeros(
                (self._num_envs, self._num_sensors * self._num_points, 4), device=self._device, dtype=torch.float32
            )
            self._data.pressure_force_raw_per_sensor = torch.zeros(
                (self._num_envs, self._num_sensors, self._num_points), device=self._device, dtype=torch.float32
            )
            self._data.pressure_force_map_raw = torch.zeros(map_shape, device=self._device, dtype=torch.float32)
            self._data.signed_distance_map = torch.zeros_like(self._data.pressure_force_map)
            self._data.penetration_map = torch.zeros_like(self._data.pressure_force_map)
            self._data.penetration_velocity_map = torch.zeros_like(self._data.pressure_force_map)
            self._data.taxel_normals_w_per_sensor = torch.zeros(
                (self._num_envs, self._num_sensors, self._num_points, 3), device=self._device, dtype=torch.float32
            )
        self._prev_penetration = torch.zeros(
            (self._num_envs, self._num_sensors, self._num_points), device=self._device, dtype=torch.float32
        )
        self._has_prev_penetration = torch.zeros(
            (self._num_envs, self._num_sensors), device=self._device, dtype=torch.bool
        )

        # Allocate SDF output only for debug/compatibility paths that read it.
        self._sdf_out = (
            torch.empty((self._num_envs, self._num_sensors, self._num_points), device=self._device, dtype=torch.float32)
            if store_debug_fields
            else None
        )

        if self.cfg.debug_vis:
            radius = float(getattr(self.cfg, "debug_vis_point_radius", 0.002))
            show_all = bool(getattr(self.cfg, "debug_vis_show_all_taxels", False))
            # Two prototypes: contact (visible) and no_contact (hidden).
            vis_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/WarpSdfTactile",
                markers={
                    "contact": sim_utils.SphereCfg(
                        radius=radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                    "no_contact": sim_utils.SphereCfg(
                        radius=radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0)),
                        visible=show_all,
                    ),
                },
            )
            self._debug_markers = VisualizationMarkers(vis_cfg)

            # Optional: show XYZ axes for each attached elastomer body.
            if bool(getattr(self.cfg, "debug_vis_show_axes", False)):
                axes_scale = float(getattr(self.cfg, "debug_vis_axes_scale", 0.05))
                axes_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/WarpSdfTactileAxes",
                    markers={
                        "frame": sim_utils.UsdFileCfg(
                            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                            scale=(axes_scale, axes_scale, axes_scale),
                        )
                    },
                )
                self._debug_axes = VisualizationMarkers(axes_cfg)


    def refresh_debug_visualization(self) -> None:
        """Refresh taxel markers from current body poses without touching pressure state."""

        if not self.cfg.debug_vis or self._debug_markers is None or self._points_local_per_sensor is None:
            return
        debug_env_id = int(getattr(self.cfg, "debug_vis_env_id", 0))
        pts_list = []
        fn_list = []
        for sensor_idx, sensor in enumerate(self._pose_sensors):
            if sensor.is_initialized:
                sensor.update(0.0, force_recompute=True)
            if not sensor.is_initialized:
                continue
            data = sensor.data
            if data.pos_w is None or data.quat_w is None or debug_env_id >= data.pos_w.shape[0]:
                continue
            pos_w = data.pos_w[debug_env_id, 0]
            quat_w = data.quat_w[debug_env_id, 0]
            points_l = self._points_local_per_sensor[sensor_idx]
            quat = quat_w.unsqueeze(0).expand(points_l.shape[0], -1)
            pts_list.append(pos_w.unsqueeze(0) + quat_apply(quat, points_l))
            if self._data.tactile_points_w_per_sensor is not None:
                fn_list.append(self._data.tactile_points_w_per_sensor[debug_env_id, sensor_idx, :, 3])
            else:
                fn_list.append(torch.zeros((points_l.shape[0],), device=points_l.device, dtype=torch.float32))
        if not pts_list:
            return

        pts = torch.cat(pts_list, dim=0)
        fn = torch.cat(fn_list, dim=0)
        thr = float(getattr(self.cfg, "debug_vis_force_threshold", 1.0e-6))
        proto = torch.where(
            fn > thr,
            torch.zeros_like(fn, dtype=torch.int64),
            torch.ones_like(fn, dtype=torch.int64),
        )
        self._debug_markers.visualize(translations=pts, marker_indices=proto)





    def _update_buffers_impl(self, env_ids: Sequence[int]):
        assert self._points_local_per_sensor is not None
        assert self._data.pressure_force_map is not None
        use_mesh = self._target_mesh_prim_path is not None and self._wp_mesh is not None
        if use_mesh:
            assert self._mesh_pos_w is not None
            assert self._mesh_quat_w is not None
        else:
            assert self._box_pos_w is not None
            assert self._box_quat_w is not None
            assert self._box_half_extents is not None
        assert self._wp_device is not None

        # Keep internal pose sensors fresh.
        for s in self._pose_sensors:
            if s.is_initialized:
                s.update(0.0, force_recompute=True)

        num_envs = int(env_ids.numel()) if isinstance(env_ids, torch.Tensor) else len(env_ids)
        rows = int(self.cfg.num_rows)
        cols = int(self.cfg.num_cols)
        store_debug_fields = bool(getattr(self.cfg, "store_debug_fields", True)) or bool(self.cfg.debug_vis)
        unsigned_contact_mode = str(getattr(self.cfg, "mesh_unsigned_contact_mode", "normal_ray")).lower()
        if unsigned_contact_mode not in ("normal_ray", "nearest"):
            raise ValueError("mesh_unsigned_contact_mode must be 'normal_ray' or 'nearest'")
        response_model = str(getattr(self.cfg, "pressure_response_model", "auto")).lower()
        if response_model not in ("auto", "penetration_kv", "gap_fraction"):
            raise ValueError("pressure_response_model must be 'auto', 'penetration_kv', or 'gap_fraction'")
        use_unsigned_normal_gap = (
            use_mesh
            and not bool(getattr(self.cfg, "mesh_use_signed_distance", False))
            and bool(getattr(self.cfg, "mesh_unsigned_shell_as_contact", False))
            and unsigned_contact_mode == "normal_ray"
        )

        tactile_per_sensor = []
        force_raw_per_sensor = []
        sdf_per_sensor = []
        penetration_per_sensor = []
        penetration_velocity_per_sensor = []
        normals_w_per_sensor = []

        for sensor_idx, s in enumerate(self._pose_sensors):
            if not s.is_initialized:
                zeros = torch.zeros((num_envs, self._num_points), device=self._device, dtype=torch.float32)
                self._data.pressure_force_map[env_ids, sensor_idx] = zeros.view(num_envs, rows, cols)
                if store_debug_fields:
                    tactile_per_sensor.append(
                        torch.zeros((num_envs, self._num_points, 4), device=self._device, dtype=torch.float32)
                    )
                    force_raw_per_sensor.append(zeros)
                    sdf_per_sensor.append(torch.full((num_envs, self._num_points), float("inf"), device=self._device))
                    penetration_per_sensor.append(zeros)
                    penetration_velocity_per_sensor.append(zeros)
                    normals_w_per_sensor.append(
                        torch.zeros((num_envs, self._num_points, 3), device=self._device, dtype=torch.float32)
                    )
                continue

            if s.num_bodies != 1:
                raise RuntimeError(
                    "WarpSdfTactileSensor expects one rigid body per elastomer prim path. "
                    f"Got {s.num_bodies} bodies for prim_path='{s.cfg.prim_path}'."
                )

            sd = s.data
            assert sd.pos_w is not None
            assert sd.quat_w is not None

            pos_w = sd.pos_w[env_ids, 0]  # (E, 3)
            quat_w = sd.quat_w[env_ids, 0]  # (E, 4)

            points_local = self._points_local_per_sensor[sensor_idx].unsqueeze(0).expand(num_envs, -1, -1)
            quat = quat_w.unsqueeze(1).expand(-1, self._num_points, -1)
            points_w = pos_w.unsqueeze(1) + quat_apply(quat, points_local)  # (E, P, 3)
            normals_w = None
            if store_debug_fields or use_unsigned_normal_gap:
                normals_l = self._pressure_taxel_maps[sensor_idx].normals_l.unsqueeze(0).expand(num_envs, -1, -1)
                normals_w = quat_apply(quat, normals_l)

            # Warp distance query (box analytic SDF or USD mesh).
            sdf_e = torch.empty((num_envs, self._num_points), device=self._device, dtype=torch.float32)

            if use_mesh:
                wp_mesh = self._wp_mesh
                assert wp_mesh is not None
                tri_indices = self._wp_mesh_tri_indices
                vertex_normals = self._wp_mesh_vertex_normals
                if tri_indices is None or vertex_normals is None:
                    raise RuntimeError("Mesh query requested but mesh auxiliary buffers are not initialized.")

                mesh_pos = self._select_env_box_param(self._mesh_pos_w, env_ids)
                mesh_quat = self._select_env_box_param(self._mesh_quat_w, env_ids)
                assert mesh_pos is not None and mesh_quat is not None
                mesh_pos = self._expand_box_param(mesh_pos, num_envs, 3, "mesh_pos_w")
                mesh_quat = self._expand_box_param(mesh_quat, num_envs, 4, "mesh_quat_w")
                mesh_scale = self._mesh_scale_w

                max_dist = float(getattr(self.cfg, "mesh_max_dist", 0.20))

                use_signed = bool(getattr(self.cfg, "mesh_use_signed_distance", False))
                signed_method = str(getattr(self.cfg, "mesh_signed_distance_method", "winding")).lower()
                if not use_signed:
                    signed_mode = 0
                else:
                    signed_mode = 2 if signed_method == "normal" else 1
                smooth_normals = int(bool(getattr(self.cfg, "mesh_smooth_normals", True)))

                mesh_scale = (
                    mesh_scale.to(device=points_w.device, dtype=points_w.dtype).clamp(min=1.0e-12)
                    if mesh_scale is not None
                    else None
                )
                points_flat = points_w.contiguous().view(-1, 3)
                mesh_pos_flat = mesh_pos.contiguous()
                mesh_quat_flat = mesh_quat.contiguous()
                mesh_scale_flat = (
                    mesh_scale.view(1, 3).contiguous()
                    if mesh_scale is not None
                    else torch.ones((1, 3), device=points_w.device, dtype=points_w.dtype)
                )
                sdf_flat = sdf_e.view(-1)
                wp.launch(
                    kernel=mesh_distance_batched_world_kernel,
                    dim=num_envs * self._num_points,
                    inputs=[
                        wp.from_torch(points_flat, dtype=wp.vec3),
                        wp.from_torch(mesh_pos_flat, dtype=wp.vec3),
                        wp.from_torch(mesh_quat_flat, dtype=wp.vec4),
                        wp.from_torch(mesh_scale_flat, dtype=wp.vec3),
                        wp_mesh.id,
                        tri_indices,
                        vertex_normals,
                        int(self._num_points),
                        int(mesh_scale is not None),
                        float(max_dist),
                        int(signed_mode),
                        int(smooth_normals),
                        wp.from_torch(sdf_flat, dtype=wp.float32),
                    ],
                    device=self._wp_device,
                )
            else:
                box_pos_w = self._box_pos_w
                box_quat_w = self._box_quat_w
                box_half_extents = self._box_half_extents
                assert box_pos_w is not None and box_quat_w is not None and box_half_extents is not None
                sdf_e = self._box_sdf_torch(
                    points_w,
                    self._select_env_box_param(box_pos_w, env_ids),
                    self._select_env_box_param(box_quat_w, env_ids),
                    self._select_env_box_param(box_half_extents, env_ids),
                )

            # Contact depth proxy: signed penetration or unsigned surface-gap closure.
            if use_mesh and not bool(getattr(self.cfg, "mesh_use_signed_distance", False)):
                if bool(getattr(self.cfg, "mesh_unsigned_shell_as_contact", False)):
                    if use_unsigned_normal_gap:
                        assert normals_w is not None
                        gap_flat = sdf_e.view(-1)
                        ray_offset = max(0.0, float(getattr(self.cfg, "mesh_normal_ray_start_offset", 1.0e-6)))
                        wp.launch(
                            kernel=mesh_normal_gap_batched_world_kernel,
                            dim=num_envs * self._num_points,
                            inputs=[
                                wp.from_torch(points_w.contiguous().view(-1, 3), dtype=wp.vec3),
                                wp.from_torch(normals_w.contiguous().view(-1, 3), dtype=wp.vec3),
                                wp.from_torch(mesh_pos.contiguous(), dtype=wp.vec3),
                                wp.from_torch(mesh_quat.contiguous(), dtype=wp.vec4),
                                wp.from_torch(mesh_scale_flat, dtype=wp.vec3),
                                wp_mesh.id,
                                int(self._num_points),
                                int(mesh_scale is not None),
                                float(max_dist),
                                float(ray_offset),
                                wp.from_torch(gap_flat, dtype=wp.float32),
                            ],
                            device=self._wp_device,
                        )
                    shell = float(getattr(self.cfg, "mesh_shell_thickness", 0.001))
                    penetration = (shell - sdf_e).clamp_min(0.0)
                else:
                    penetration = torch.zeros_like(sdf_e)
            else:
                # Signed distance (or analytic box SDF): negative means inside.
                penetration = (-sdf_e).clamp_min(0.0)
            deadband = max(0.0, float(getattr(self.cfg, "penetration_deadband", 0.0)))
            if deadband > 0.0:
                penetration = (penetration - deadband).clamp_min(0.0)

            assert self._prev_penetration is not None
            assert self._has_prev_penetration is not None
            damping_dt = max(float(getattr(self.cfg, "pressure_damping_dt", 1.0 / 120.0)), 1.0e-12)
            prev_penetration = self._prev_penetration[env_ids, sensor_idx]
            has_prev = self._has_prev_penetration[env_ids, sensor_idx].unsqueeze(-1)
            penetration_velocity = torch.where(
                has_prev,
                (penetration - prev_penetration) / damping_dt,
                torch.zeros_like(penetration),
            )
            self._prev_penetration[env_ids, sensor_idx] = penetration.detach()
            self._has_prev_penetration[env_ids, sensor_idx] = True

            calibration = (
                self._pressure_taxel_maps[sensor_idx].calibration
                if sensor_idx < len(self._pressure_taxel_maps)
                else PressureCalibration(
                    stiffness=coerce_taxel_calibration_value(
                        self.cfg.stiffness, (int(self.cfg.num_rows), int(self.cfg.num_cols)), backend_like=penetration
                    ),
                    damping=coerce_taxel_calibration_value(
                        getattr(self.cfg, "damping", 0.0),
                        (int(self.cfg.num_rows), int(self.cfg.num_cols)),
                        backend_like=penetration,
                    ),
                    max_force=coerce_taxel_calibration_value(
                        self.cfg.max_force, (int(self.cfg.num_rows), int(self.cfg.num_cols)), backend_like=penetration
                    ),
                )
            )
            use_gap_fraction = response_model == "gap_fraction" or (
                response_model == "auto"
                and use_mesh
                and not bool(getattr(self.cfg, "mesh_use_signed_distance", False))
                and bool(getattr(self.cfg, "mesh_unsigned_shell_as_contact", False))
            )
            if use_gap_fraction:
                fn_raw, fn = calibrate_gap_fraction(
                    penetration,
                    float(getattr(self.cfg, "mesh_shell_thickness", 0.001)),
                    calibration,
                    normalize=bool(self.cfg.normalize_forces),
                )
            else:
                fn_raw, fn = calibrate_penetration(
                    penetration,
                    calibration,
                    penetration_velocity=penetration_velocity,
                    normalize=bool(self.cfg.normalize_forces),
                )
            self._data.pressure_force_map[env_ids, sensor_idx] = fn.view(num_envs, rows, cols)

            if store_debug_fields:
                tactile = torch.cat((points_w, fn.unsqueeze(-1)), dim=-1)
                tactile_per_sensor.append(tactile)
                force_raw_per_sensor.append(fn_raw)
                sdf_per_sensor.append(sdf_e)
                penetration_per_sensor.append(penetration)
                penetration_velocity_per_sensor.append(penetration_velocity)
                assert normals_w is not None
                normals_w_per_sensor.append(normals_w)

        if store_debug_fields:
            tactile_stack = torch.stack(tactile_per_sensor, dim=1)  # (E, S, P, 4)
            force_raw_stack = torch.stack(force_raw_per_sensor, dim=1)  # (E, S, P)
            sdf_stack = torch.stack(sdf_per_sensor, dim=1)  # (E, S, P)
            penetration_stack = torch.stack(penetration_per_sensor, dim=1)  # (E, S, P)
            penetration_velocity_stack = torch.stack(penetration_velocity_per_sensor, dim=1)  # (E, S, P)
            normals_w_stack = torch.stack(normals_w_per_sensor, dim=1)  # (E, S, P, 3)

            if self._data.tactile_points_w_per_sensor is not None:
                self._data.tactile_points_w_per_sensor[env_ids] = tactile_stack
            if self._data.tactile_points_w is not None:
                self._data.tactile_points_w[env_ids] = tactile_stack.view(num_envs, self._num_sensors * self._num_points, 4)
            if self._data.pressure_force_raw_per_sensor is not None:
                self._data.pressure_force_raw_per_sensor[env_ids] = force_raw_stack
            if self._data.pressure_force_map_raw is not None:
                self._data.pressure_force_map_raw[env_ids] = force_raw_stack.view(num_envs, self._num_sensors, rows, cols)
            if self._data.signed_distance_map is not None:
                self._data.signed_distance_map[env_ids] = sdf_stack.view(num_envs, self._num_sensors, rows, cols)
            if self._data.penetration_map is not None:
                self._data.penetration_map[env_ids] = penetration_stack.view(num_envs, self._num_sensors, rows, cols)
            if self._data.penetration_velocity_map is not None:
                self._data.penetration_velocity_map[env_ids] = penetration_velocity_stack.view(
                    num_envs, self._num_sensors, rows, cols
                )
            if self._data.taxel_normals_w_per_sensor is not None:
                self._data.taxel_normals_w_per_sensor[env_ids] = normals_w_stack
            if self._sdf_out is not None:
                self._sdf_out[env_ids] = sdf_stack

        # Debug visualization: show taxels with fn > threshold for a chosen env.
        if self.cfg.debug_vis and self._debug_markers is not None and self._data.tactile_points_w_per_sensor is not None:
            debug_env_id = int(getattr(self.cfg, "debug_vis_env_id", 0))
            if isinstance(env_ids, torch.Tensor):
                env_ids_list = env_ids.detach().cpu().tolist()
            else:
                env_ids_list = list(env_ids)
            if debug_env_id in env_ids_list:
                tactile_env = self._data.tactile_points_w_per_sensor[debug_env_id]  # (S, P, 4)
                pts = tactile_env[..., :3].reshape(-1, 3)
                fn = tactile_env[..., 3].reshape(-1)
                thr = float(getattr(self.cfg, "debug_vis_force_threshold", 1.0e-6))
                contact = fn > thr
                # 0 -> contact prototype, 1 -> no_contact prototype (hidden unless show_all enabled)
                proto = torch.where(
                    contact,
                    torch.zeros_like(fn, dtype=torch.int64),
                    torch.ones_like(fn, dtype=torch.int64),
                )
                self._debug_markers.visualize(translations=pts, marker_indices=proto)

                # Also visualize per-sensor body frames (XYZ axes) to inspect orientation.
                if self._debug_axes is not None:
                    trans_list = []
                    quat_list = []
                    # Visualize axes at the tactile patch frame (not the raw rigid-body origin),
                    # so users can validate/adjust patch_offset_{pos,quat} and normal_offset.
                    patch_pos_b = torch.tensor(
                        self._resolve_patch_offset_pos_list(),
                        device=self.device,
                        dtype=torch.float32,
                    )  # (S, 3)
                    patch_quat_b = torch.tensor(
                        self._resolve_patch_offset_quat_list(),
                        device=self.device,
                        dtype=torch.float32,
                    )  # (S, 4)

                    # normal_offset is applied along normal_axis in the patch-local frame when generating taxels.
                    normal_axis = int(self.cfg.normal_axis)
                    n_local = torch.zeros((self._num_sensors, 3), device=self.device, dtype=torch.float32)
                    n_local[:, normal_axis] = float(self.cfg.normal_offset)

                    for sensor_idx, s in enumerate(self._pose_sensors):
                        if not s.is_initialized:
                            continue
                        sd = s.data
                        if sd.pos_w is None or sd.quat_w is None:
                            continue

                        body_pos_w = sd.pos_w[debug_env_id, 0]
                        body_quat_w = sd.quat_w[debug_env_id, 0]

                        # patch orientation in world
                        patch_quat_w = quat_mul(body_quat_w.unsqueeze(0), patch_quat_b[sensor_idx].unsqueeze(0)).squeeze(0)
                        # patch origin in world: body origin + rotated patch offset + rotated normal offset
                        off_w = quat_apply(body_quat_w.unsqueeze(0), patch_pos_b[sensor_idx].unsqueeze(0)).squeeze(0)
                        n_off_w = quat_apply(patch_quat_w.unsqueeze(0), n_local[sensor_idx].unsqueeze(0)).squeeze(0)
                        patch_pos_w = body_pos_w + off_w + n_off_w

                        trans_list.append(patch_pos_w)
                        quat_list.append(patch_quat_w)
                    if trans_list:
                        trans = torch.stack(trans_list, dim=0)
                        quats = torch.stack(quat_list, dim=0)
                        idx = torch.zeros((trans.shape[0],), device=trans.device, dtype=torch.int64)
                        self._debug_axes.visualize(translations=trans, orientations=quats, marker_indices=idx)

    @staticmethod
    def _create_local_grid_points(
        *,
        num_rows: int,
        num_cols: int,
        point_distance: float | None = None,
        row_distance: float | None = None,
        col_distance: float | None = None,
        normal_axis: int,
        normal_offset: float,
        device: str,
    ) -> torch.Tensor:
        if row_distance is None:
            row_distance = point_distance
        if col_distance is None:
            col_distance = point_distance
        if row_distance is None or col_distance is None:
            raise ValueError("point_distance or both row_distance/col_distance must be provided")
        tangential_axes = [0, 1, 2]
        tangential_axes.remove(normal_axis)
        axis_u, axis_v = tangential_axes

        u = (torch.arange(num_rows, device=device, dtype=torch.float32) - (float(num_rows) - 1.0) / 2.0) * float(
            row_distance
        )
        v = (torch.arange(num_cols, device=device, dtype=torch.float32) - (float(num_cols) - 1.0) / 2.0) * float(
            col_distance
        )

        uu, vv = torch.meshgrid(u, v, indexing="ij")
        points = torch.zeros((num_rows * num_cols, 3), device=device, dtype=torch.float32)
        points[:, axis_u] = uu.reshape(-1)
        points[:, axis_v] = vv.reshape(-1)
        points[:, normal_axis] = float(normal_offset)
        return points

    def _resolve_grid_distances(self) -> tuple[float, float]:
        point_distance = getattr(self.cfg, "point_distance", None)
        row_distance = getattr(self.cfg, "row_distance", None)
        col_distance = getattr(self.cfg, "col_distance", None)
        if row_distance is None:
            row_distance = point_distance
        if col_distance is None:
            col_distance = point_distance
        if row_distance is None or col_distance is None:
            raise ValueError("'point_distance' or both 'row_distance'/'col_distance' must be provided")
        row_distance = float(row_distance)
        col_distance = float(col_distance)
        if row_distance <= 0.0:
            raise ValueError("'row_distance' must be positive")
        if col_distance <= 0.0:
            raise ValueError("'col_distance' must be positive")
        return row_distance, col_distance

    def _resolve_normal_sign(self) -> float:
        raw = getattr(self.cfg, "normal_sign", None)
        if raw is None:
            return 1.0 if float(self.cfg.normal_offset) >= 0.0 else -1.0
        return 1.0 if float(raw) >= 0.0 else -1.0

    def _select_env_box_param(self, value: torch.Tensor, env_ids: Sequence[int]) -> torch.Tensor:
        if value.ndim != 2 or value.shape[0] != self._num_envs:
            return value
        return value[env_ids]

    @staticmethod
    def _expand_box_param(value: torch.Tensor, num_envs: int, width: int, name: str) -> torch.Tensor:
        value = value.to(dtype=torch.float32)
        if value.ndim == 1 and value.shape[0] == width:
            return value.unsqueeze(0).expand(num_envs, -1)
        if value.ndim == 2 and value.shape[-1] == width:
            if value.shape[0] == num_envs:
                return value
            if value.shape[0] == 1:
                return value.expand(num_envs, -1)
        raise ValueError(f"Expected {name} shape ({width},) or (num_envs, {width}), got {tuple(value.shape)}")

    @classmethod
    def _box_sdf_torch(
        cls,
        points_w: torch.Tensor,
        box_pos_w: torch.Tensor,
        box_quat_w: torch.Tensor,
        box_half_extents: torch.Tensor,
    ) -> torch.Tensor:
        num_envs, num_points, _ = points_w.shape
        box_pos_w = cls._expand_box_param(box_pos_w, num_envs, 3, "box_pos_w")
        box_quat_w = cls._expand_box_param(box_quat_w, num_envs, 4, "box_quat_w")
        box_half_extents = cls._expand_box_param(box_half_extents, num_envs, 3, "box_half_extents")

        rel_w = points_w - box_pos_w.unsqueeze(1)
        quat = box_quat_w.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, 4)
        points_l = quat_apply_inverse(quat, rel_w.reshape(-1, 3)).reshape(num_envs, num_points, 3)

        q = torch.abs(points_l) - box_half_extents.unsqueeze(1)
        outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
        inside = torch.clamp(torch.amax(q, dim=-1), max=0.0)
        return outside + inside

    def _resolve_patch_offset_pos_list(self) -> list[tuple[float, float, float]]:
        per = getattr(self.cfg, "patch_offset_pos_b_per_elastomer", None)
        if per is not None:
            if len(per) != self._num_sensors:
                raise ValueError(
                    "'patch_offset_pos_b_per_elastomer' must have the same length as 'elastomer_prim_paths'. "
                    f"Got {len(per)} vs {self._num_sensors}."
                )
            return list(per)
        base = getattr(self.cfg, "patch_offset_pos_b", (0.0, 0.0, 0.0))
        base_tuple: tuple[float, float, float] = (float(base[0]), float(base[1]), float(base[2]))
        return [base_tuple for _ in range(self._num_sensors)]

    def _resolve_patch_offset_quat_list(self) -> list[tuple[float, float, float, float]]:
        per = getattr(self.cfg, "patch_offset_quat_b_per_elastomer", None)
        if per is not None:
            if len(per) != self._num_sensors:
                raise ValueError(
                    "'patch_offset_quat_b_per_elastomer' must have the same length as 'elastomer_prim_paths'. "
                    f"Got {len(per)} vs {self._num_sensors}."
                )
            return list(per)
        base = getattr(self.cfg, "patch_offset_quat_b", (1.0, 0.0, 0.0, 0.0))
        base_tuple: tuple[float, float, float, float] = (
            float(base[0]),
            float(base[1]),
            float(base[2]),
            float(base[3]),
        )
        return [base_tuple for _ in range(self._num_sensors)]

    @staticmethod
    def _triangulate_usd_mesh(face_counts, face_indices):
        # Fan triangulation for polygon faces.
        tris = []
        idx = 0
        for n in face_counts:
            n = int(n)
            if n < 3:
                idx += n
                continue
            v0 = int(face_indices[idx])
            for i in range(1, n - 1):
                v1 = int(face_indices[idx + i])
                v2 = int(face_indices[idx + i + 1])
                tris.append((v0, v1, v2))
            idx += n
        return tris

    def _resolve_target_mesh_prim_path(self, prim_path: str | None) -> str:
        if not prim_path:
            raise RuntimeError("target_mesh_prim_path is empty.")
        from pxr import Sdf, Usd
        prim = self.stage.GetPrimAtPath(prim_path) if Sdf.Path.IsValidPathString(prim_path) else Usd.Prim()
        if prim.IsValid():
            if prim.GetTypeName() in ("Mesh", "Cube", "Sphere"):
                return str(prim.GetPath())
            mesh_prim = sim_utils.get_first_matching_child_prim(
                str(prim.GetPath()),
                lambda child: child.GetTypeName() in ("Mesh", "Cube", "Sphere"),
                stage=self.stage,
            )
            if mesh_prim is not None and mesh_prim.IsValid():
                return str(mesh_prim.GetPath())
            return str(prim.GetPath())
        try:
            matches = sim_utils.find_matching_prims(prim_path, stage=self.stage)
        except (RuntimeError, ValueError):
            matches = []
        if matches:
            return self._resolve_target_mesh_prim_path(str(matches[0].GetPath()))
        return prim_path

    def _load_warp_mesh_from_usd(self, prim_path: str, device: str):
        import numpy as np
        from pxr import UsdGeom  # type: ignore[import-not-found]

        stage_prim = self.stage.GetPrimAtPath(prim_path)
        if stage_prim.IsValid() and stage_prim.GetTypeName() == "Mesh":
            mesh_prim = stage_prim
        elif stage_prim.IsValid() and stage_prim.GetTypeName() == "Sphere":
            return self._load_warp_sphere_from_usd(stage_prim, device=device)
        elif stage_prim.IsValid() and stage_prim.GetTypeName() == "Cube":
            return self._load_warp_cube_from_usd(stage_prim, device=device)
        else:
            mesh_prim = sim_utils.get_first_matching_child_prim(
                prim_path,
                lambda prim: prim.GetTypeName() in ("Mesh", "Cube", "Sphere"),
                stage=self.stage,
            )
            if mesh_prim is not None and mesh_prim.GetTypeName() == "Sphere":
                return self._load_warp_sphere_from_usd(mesh_prim, device=device)
            if mesh_prim is not None and mesh_prim.GetTypeName() == "Cube":
                return self._load_warp_cube_from_usd(mesh_prim, device=device)
        if mesh_prim is None or not mesh_prim.IsValid():
            raise RuntimeError(f"Invalid mesh prim path (no UsdGeom.Mesh, UsdGeom.Cube or UsdGeom.Sphere found under): {prim_path}")

        mesh = UsdGeom.Mesh(mesh_prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)

        face_counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        face_indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        if face_counts.size == 0 or face_indices.size == 0:
            raise RuntimeError(f"Mesh prim has no faces: {mesh.GetPath()}")

        if np.all(face_counts == 3) and (face_indices.size % 3 == 0):
            triangles = face_indices.reshape(-1, 3)
        else:
            triangles = np.asarray(self._triangulate_usd_mesh(face_counts, face_indices), dtype=np.int32)
            if triangles.size == 0:
                raise RuntimeError(f"Failed to triangulate mesh prim: {mesh.GetPath()}")

        points, triangles = self._weld_duplicate_vertices(points, triangles)

        # Compute vertex normals (area-weighted)
        v0 = points[triangles[:, 0]]
        v1 = points[triangles[:, 1]]
        v2 = points[triangles[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        vert_normals = np.zeros_like(points, dtype=np.float32)
        np.add.at(vert_normals, triangles[:, 0], face_normals)
        np.add.at(vert_normals, triangles[:, 1], face_normals)
        np.add.at(vert_normals, triangles[:, 2], face_normals)
        nrm = np.linalg.norm(vert_normals, axis=1, keepdims=True)
        vert_normals = np.divide(
            vert_normals,
            np.clip(nrm, 1.0e-12, None),
            out=np.zeros_like(vert_normals, dtype=np.float32),
            where=nrm > 0.0,
        ).astype(np.float32)

        wp_mesh = wp.Mesh(
            points=wp.array(points.astype(np.float32), dtype=wp.vec3, device=device),
            indices=wp.array(triangles.astype(np.int32).flatten(), dtype=wp.int32, device=device),
            support_winding_number=True,
        )
        wp_tri_indices = wp.array(triangles.astype(np.int32).flatten(), dtype=wp.int32, device=device)
        wp_vertex_normals = wp.array(vert_normals.astype(np.float32), dtype=wp.vec3, device=device)
        return wp_mesh, wp_tri_indices, wp_vertex_normals

    def _load_warp_sphere_from_usd(self, sphere_prim, device: str):
        """Triangulate the actual USD sphere for pressure signed-distance queries."""
        import numpy as np
        import trimesh
        from pxr import UsdGeom

        radius = float(UsdGeom.Sphere(sphere_prim).GetRadiusAttr().Get())
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=radius)
        points = np.asarray(mesh.vertices, dtype=np.float32)
        indices = np.asarray(mesh.faces, dtype=np.int32).reshape(-1)
        normals = points / np.linalg.norm(points, axis=1, keepdims=True)
        wp_points = wp.array(points, dtype=wp.vec3, device=device)
        wp_indices = wp.array(indices, dtype=wp.int32, device=device)
        wp_mesh = wp.Mesh(points=wp_points, indices=wp_indices, support_winding_number=True)
        return wp_mesh, wp_indices, wp.array(normals, dtype=wp.vec3, device=device)

    def _load_warp_cube_from_usd(self, cube_prim, device: str):
        import numpy as np
        from pxr import UsdGeom  # type: ignore[import-not-found]

        cube = UsdGeom.Cube(cube_prim)
        size_attr = cube.GetSizeAttr().Get()
        size = float(size_attr) if size_attr is not None else 2.0
        h = 0.5 * size
        points = np.asarray(
            [
                (-h, -h, -h),
                (h, -h, -h),
                (h, h, -h),
                (-h, h, -h),
                (-h, -h, h),
                (h, -h, h),
                (h, h, h),
                (-h, h, h),
            ],
            dtype=np.float32,
        )
        triangles = np.asarray(
            [
                (0, 2, 1),
                (0, 3, 2),
                (4, 5, 6),
                (4, 6, 7),
                (0, 1, 5),
                (0, 5, 4),
                (3, 7, 6),
                (3, 6, 2),
                (0, 4, 7),
                (0, 7, 3),
                (1, 2, 6),
                (1, 6, 5),
            ],
            dtype=np.int32,
        )
        vert_normals = points / np.clip(np.linalg.norm(points, axis=1, keepdims=True), 1.0e-12, None)
        wp_mesh = wp.Mesh(
            points=wp.array(points, dtype=wp.vec3, device=device),
            indices=wp.array(triangles.flatten(), dtype=wp.int32, device=device),
            support_winding_number=True,
        )
        wp_tri_indices = wp.array(triangles.flatten(), dtype=wp.int32, device=device)
        wp_vertex_normals = wp.array(vert_normals.astype(np.float32), dtype=wp.vec3, device=device)
        return wp_mesh, wp_tri_indices, wp_vertex_normals

    @staticmethod
    def _weld_duplicate_vertices(points, triangles, *, decimals: int = 8):
        """Merge duplicated USD vertices so mesh signed-distance queries see a connected surface."""
        import numpy as np

        original_vertices = int(points.shape[0])
        rounded = np.round(points, decimals=int(decimals))
        unique_points, inverse = np.unique(rounded, axis=0, return_inverse=True)
        unique_points = unique_points.astype(np.float32, copy=False)
        welded_triangles = inverse[triangles].astype(np.int32, copy=False)

        valid = (
            (welded_triangles[:, 0] != welded_triangles[:, 1])
            & (welded_triangles[:, 1] != welded_triangles[:, 2])
            & (welded_triangles[:, 2] != welded_triangles[:, 0])
        )
        welded_triangles = welded_triangles[valid]

        if welded_triangles.size:
            canonical = np.sort(welded_triangles, axis=1)
            _, keep = np.unique(canonical, axis=0, return_index=True)
            welded_triangles = welded_triangles[np.sort(keep)]

        if unique_points.shape[0] != original_vertices:
            print(
                f"[INFO] welded USD mesh vertices: {original_vertices} -> {unique_points.shape[0]}, "
                f"triangles={welded_triangles.shape[0]}",
                flush=True,
            )

        return unique_points, welded_triangles
