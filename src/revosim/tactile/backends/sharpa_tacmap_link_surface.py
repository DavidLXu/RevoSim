from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Literal

import numpy as np
import torch

import omni.physics.tensors.impl.api as physx
import isaaclab.sim as sim_utils
from isaaclab.sim.views import XformPrimView
import isaaclab.utils.math as math_utils
from isaaclab.sensors.ray_caster.multi_mesh_ray_caster import MultiMeshRayCaster
from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_data import MultiMeshRayCasterData
from isaaclab.utils import configclass
from isaaclab.utils.warp import raycast_dynamic_meshes
from pxr import Gf, Sdf, Usd, UsdGeom, Vt
import omni.usd as omni_usd

from .sharpa_tacmap_cfg import SharpaTacmapCfg
from .torch_jit_utils import deform_quantize


AxisName = Literal["+x", "-x", "+y", "-y", "+z", "-z"]


def _load_explicit_ray_layout(
    layout_path: str,
    prefix: str,
    *,
    expected_count: int,
    backoff_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(layout_path, allow_pickle=False) as layout:
        starts_key = f"{prefix}_ray_starts_link_m"
        directions_key = f"{prefix}_ray_directions_link"
        uses_camera_plane_starts = starts_key in layout and directions_key in layout
        if uses_camera_plane_starts:
            starts = np.asarray(layout[starts_key], dtype=np.float32).reshape(-1, 3)
            directions = np.asarray(layout[directions_key], dtype=np.float32).reshape(-1, 3)
        else:
            points = np.asarray(
                layout[f"{prefix}_ray_points_link_m"], dtype=np.float32
            ).reshape(-1, 3)
            directions = np.asarray(
                layout[f"{prefix}_ray_normals_link"], dtype=np.float32
            ).reshape(-1, 3)
            starts = points
    if len(starts) != int(expected_count) or len(directions) != int(expected_count):
        raise ValueError(
            f"Explicit ray layout {prefix!r} must contain {expected_count} starts and directions"
        )
    direction_length = np.linalg.norm(directions, axis=1, keepdims=True)
    if (
        not np.isfinite(starts).all()
        or not np.isfinite(directions).all()
        or np.any(direction_length <= 1.0e-9)
    ):
        raise ValueError(f"Explicit ray layout {prefix!r} contains invalid starts or directions")
    directions = directions / direction_length
    if not uses_camera_plane_starts:
        starts = starts - max(0.0, float(backoff_m)) * directions
    return starts.astype(np.float32), directions.astype(np.float32)


def _load_explicit_ray_hit_indices(
    layout_path: str,
    prefix: str,
    *,
    expected_count: int,
) -> np.ndarray:
    with np.load(layout_path, allow_pickle=False) as layout:
        hit_indices = np.asarray(
            layout[f"{prefix}_ray_surface_hit_index"], dtype=np.int64
        ).reshape(-1)
    if len(hit_indices) != int(expected_count) or not np.all(
        (hit_indices == 1) | (hit_indices == 2)
    ):
        raise ValueError(
            f"Explicit ray layout {prefix!r} must contain {expected_count} hit indices in {{1, 2}}"
        )
    return hit_indices


class _UsdXformPoseView:
    def __init__(self, prim_path: str, prims):
        self.prim_path = prim_path
        self.prims = list(prims)
        self.count = len(self.prims)


@configclass
class SharpaTacmapLinkSurfaceCfg(SharpaTacmapCfg):
    """TacMap v2 dense ray grid in the touch-link local frame."""

    image_width: int = 240
    image_height: int = 240
    ray_axis: AxisName = "+x"
    ray_direction: tuple[float, float, float] | None = None
    grid_u_axis: AxisName = "+y"
    grid_v_axis: AxisName = "+z"
    orthonormalize_grid_axes: bool = True
    grid_u_size: float = 0.014
    grid_v_size: float = 0.020
    grid_center: tuple[float, float, float] = (-0.008, 0.0, 0.0012)
    ray_layout_npz: str | None = None
    ray_layout_prefix: str = ""
    ray_backoff_m: float = 0.012
    ray_hit_index: int = 1
    use_ray_hit_index_layout: bool = False
    second_hit_epsilon: float = 1.0e-5
    use_first_hit_fallback: bool = False
    debug_viz_link_surfaces: bool = False
    debug_viz_hits: bool = False
    debug_viz_hits_external_mask: bool = False
    debug_viz_rays: bool = True
    debug_viz_ray_length: float = 0.008
    debug_viz_ray_width: float = 0.00006


class SharpaTacmapLinkSurface(MultiMeshRayCaster):
    """Cast a dense local-frame ray grid against one configured target mesh."""

    cfg: SharpaTacmapLinkSurfaceCfg
    UNSUPPORTED_TYPES: ClassVar[set[str]] = set()

    def __init__(self, cfg: SharpaTacmapLinkSurfaceCfg):
        for name in cfg.data_types:
            if name not in ["distance_along_normal", "distance_along_normal_raw"]:
                raise ValueError(f"Unsupported data type: {name}")
        super().__init__(cfg)
        MultiMeshRayCaster.__init__(self, cfg)
        self._data = MultiMeshRayCasterData()

    def _obtain_trackable_prim_view(self, target_prim_path: str):
        if target_prim_path.rstrip("/").endswith("TacmapAnchor") or "TacmapAnchor" in target_prim_path:
            prims = sim_utils.find_matching_prims(target_prim_path, stage=self.stage)
            if len(prims) == 0:
                raise RuntimeError(f"Failed to find TacmapAnchor prims at path expression: {target_prim_path}")
            view = _UsdXformPoseView(target_prim_path, prims)
            positions = torch.zeros((view.count, 3), dtype=torch.float32, device=self._device)
            orientations = torch.zeros((view.count, 4), dtype=torch.float32, device=self._device)
            orientations[:, 0] = 1.0
            return view, (positions, orientations)
        return super()._obtain_trackable_prim_view(target_prim_path)

    @property
    def data(self) -> MultiMeshRayCasterData:
        self._update_outdated_buffers()
        return self._data

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._frame[env_ids] = 0

    def _initialize_rays_impl(self):
        self._ALL_INDICES = torch.arange(self._view.count, device=self._device, dtype=torch.long)
        self._frame = torch.zeros(self._view.count, device=self._device, dtype=torch.long)

        height = int(self.cfg.image_height)
        width = int(self.cfg.image_width)
        if height <= 0 or width <= 0:
            raise ValueError(f"image_width/image_height must be positive, got {width}x{height}")

        if self.cfg.ray_layout_npz:
            starts_np, directions_np = _load_explicit_ray_layout(
                self.cfg.ray_layout_npz,
                self.cfg.ray_layout_prefix,
                expected_count=height * width,
                backoff_m=self.cfg.ray_backoff_m,
            )
            starts = torch.as_tensor(starts_np, dtype=torch.float32, device=self._device)
            directions = torch.as_tensor(directions_np, dtype=torch.float32, device=self._device)
        else:
            ray_axis = _direction_vec(getattr(self.cfg, "ray_direction", None), self.cfg.ray_axis, self._device)
            u_axis = _axis_vec(self.cfg.grid_u_axis, self._device)
            v_axis = _axis_vec(self.cfg.grid_v_axis, self._device)
            if bool(getattr(self.cfg, "orthonormalize_grid_axes", True)):
                u_axis, v_axis = _orthonormal_grid_axes(ray_axis, u_axis, v_axis)
            center = torch.tensor(self.cfg.grid_center, dtype=torch.float32, device=self._device)

            u = torch.linspace(
                -0.5 * float(self.cfg.grid_u_size),
                0.5 * float(self.cfg.grid_u_size),
                width,
                device=self._device,
            )
            v = torch.linspace(
                -0.5 * float(self.cfg.grid_v_size),
                0.5 * float(self.cfg.grid_v_size),
                height,
                device=self._device,
            )
            grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")
            starts = center + grid_u.reshape(-1, 1) * u_axis + grid_v.reshape(-1, 1) * v_axis
            directions = ray_axis.view(1, 3).repeat(starts.shape[0], 1)

        if bool(getattr(self.cfg, "use_ray_hit_index_layout", False)):
            if not self.cfg.ray_layout_npz:
                raise ValueError("use_ray_hit_index_layout requires ray_layout_npz")
            hit_indices_np = _load_explicit_ray_hit_indices(
                self.cfg.ray_layout_npz,
                self.cfg.ray_layout_prefix,
                expected_count=height * width,
            )
            self._ray_hit_indices_att = torch.as_tensor(
                hit_indices_np,
                dtype=torch.long,
                device=self._device,
            )
        else:
            self._ray_hit_indices_att = None

        self.num_rays = int(starts.shape[0])
        self._create_buffers()

        self.ray_starts_att = starts.unsqueeze(0).repeat(self._view.count, 1, 1)
        self.ray_directions_att = directions.unsqueeze(0).repeat(self._view.count, 1, 1)
        self.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        self.ray_normals_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        self.ray_hit_valid = torch.zeros(self._view.count, self.num_rays, device=self._device, dtype=torch.bool)
        self.first_ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        self.second_ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        self.first_ray_normals_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        self.second_ray_normals_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        self.first_ray_hit_valid = torch.zeros(self._view.count, self.num_rays, device=self._device, dtype=torch.bool)
        self.second_ray_hit_valid = torch.zeros(self._view.count, self.num_rays, device=self._device, dtype=torch.bool)

        quat_w = math_utils.convert_camera_frame_orientation_convention(
            torch.tensor([self.cfg.offset.rot], device=self._device),
            origin=self.cfg.offset.convention,
            target="world",
        )
        self._offset_quat = quat_w.repeat(self._view.count, 1)
        self._offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device).repeat(self._view.count, 1)

        self._ray_starts_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)
        self._ray_directions_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)

        max_viz_points = max(1, int(getattr(self.cfg, "debug_viz_max_points", 5000)))
        self._viz_stride = max(1, (self.num_rays + max_viz_points - 1) // max_viz_points)
        self._viz_count = (self.num_rays + self._viz_stride - 1) // self._viz_stride
        self._viz_indices = torch.arange(
            0, self.num_rays, self._viz_stride, device=self._device, dtype=torch.long
        )[: self._viz_count]

        if bool(getattr(self.cfg, "debug_viz_hits", False)):
            self._setup_hit_points_viz_usd(env_id=int(getattr(self.cfg, "debug_viz_env_id", 0)))
        if bool(getattr(self.cfg, "debug_viz_link_surfaces", False)):
            self._setup_link_surfaces_viz_usd(env_id=int(getattr(self.cfg, "debug_viz_env_id", 0)))
        if bool(getattr(self.cfg, "debug_viz_rays", False)):
            self._setup_rays_viz_usd(env_id=int(getattr(self.cfg, "debug_viz_env_id", 0)))

    def _create_buffers(self):
        self._data.pos_w = torch.zeros((self._view.count, 3), device=self._device)
        self._data.quat_w = torch.zeros((self._view.count, 4), device=self._device)
        self._data.output = {}
        if "distance_along_normal" in self.cfg.data_types:
            self._data.output["distance_along_normal"] = torch.zeros(
                (self._view.count, self.num_rays, 1), device=self._device, dtype=torch.uint8
            )
        if "distance_along_normal_raw" in self.cfg.data_types:
            self._data.output["distance_along_normal_raw"] = torch.zeros(
                (self._view.count, self.num_rays, 1), device=self._device, dtype=torch.float32
            )
        self._data.image_mesh_ids = torch.zeros(self._num_envs, self.num_rays, 1, device=self.device, dtype=torch.int16)

        self.drift = torch.zeros((self._view.count, 3), device=self._device, dtype=torch.float32)
        self.bias = torch.zeros((self._view.count, 3), device=self._device, dtype=torch.float32)
        self.ray_cast_drift = torch.zeros((self._view.count, 3), device=self._device, dtype=torch.float32)
        self.ray_cast_bias = torch.zeros((self._view.count, 3), device=self._device, dtype=torch.float32)

    def _normalize_env_ids(self, env_ids: Sequence[int] | None) -> torch.Tensor:
        if env_ids is None:
            return self._ALL_INDICES
        if isinstance(env_ids, slice):
            if env_ids == slice(None):
                return self._ALL_INDICES
            return self._ALL_INDICES[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self._device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self._device, dtype=torch.long)

    def _update_ray_infos(self, env_ids: Sequence[int]):
        env_ids = self._normalize_env_ids(env_ids)
        pos_w, quat_w = self._compute_view_world_poses(self._view, env_ids)
        pos_w, quat_w = math_utils.combine_frame_transforms(
            pos_w, quat_w, self._offset_pos[env_ids], self._offset_quat[env_ids]
        )
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        ray_starts_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts_att[env_ids])
        ray_starts_w += pos_w.unsqueeze(1)
        ray_directions_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions_att[env_ids])

        self._ray_starts_w[env_ids] = ray_starts_w
        self._ray_directions_w[env_ids] = ray_directions_w

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        env_ids = self._normalize_env_ids(env_ids)
        self._update_ray_infos(env_ids)

        self._frame[env_ids] += 1

        mesh_idx = 0
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]
                continue

            view_ids = torch.arange(view.count, device=self._device, dtype=torch.long)
            pos_w, ori_w = self._compute_view_world_poses(view, view_ids)
            pos_w = pos_w.squeeze(0) if len(pos_w.shape) == 3 else pos_w
            ori_w = ori_w.squeeze(0) if len(ori_w.shape) == 3 else ori_w

            if target_cfg.prim_expr in MultiMeshRayCaster.mesh_offsets:
                pos_offset, ori_offset = MultiMeshRayCaster.mesh_offsets[target_cfg.prim_expr]
                pos_w -= pos_offset
                ori_w = math_utils.quat_mul(ori_offset.expand(ori_w.shape[0], -1), ori_w)

            count = view.count
            if count != 1:
                count = count // self._num_envs
                pos_w = pos_w.view(self._num_envs, count, 3)
                ori_w = ori_w.view(self._num_envs, count, 4)

            self._mesh_positions_w[:, mesh_idx : mesh_idx + count] = pos_w
            self._mesh_orientations_w[:, mesh_idx : mesh_idx + count] = ori_w
            mesh_idx += count

        self.ray_hits_w[env_ids], ray_depth, ray_normal, _, ray_mesh_ids = raycast_dynamic_meshes(
            self._ray_starts_w[env_ids],
            self._ray_directions_w[env_ids],
            mesh_ids_wp=self._mesh_ids_wp,
            max_dist=self.cfg.max_distance,
            mesh_positions_w=self._mesh_positions_w[env_ids],
            mesh_orientations_w=self._mesh_orientations_w[env_ids],
            return_distance=True,
            return_normal=True,
            return_mesh_id=bool(getattr(self.cfg, "update_mesh_ids", False)),
        )
        first_hits = self.ray_hits_w[env_ids].clone()
        first_depth = ray_depth
        first_valid = torch.isfinite(first_depth) & (first_depth > 0.0)
        first_normals = _normalize_hit_normals(ray_normal, first_valid)
        ray_hit_index = max(1, int(getattr(self.cfg, "ray_hit_index", 1)))
        ray_hit_indices = self._ray_hit_indices_att
        self.first_ray_hits_w[env_ids] = torch.where(first_valid.unsqueeze(-1), first_hits, torch.zeros_like(first_hits))
        self.first_ray_normals_w[env_ids] = first_normals
        self.first_ray_hit_valid[env_ids] = first_valid
        self.second_ray_hits_w[env_ids] = torch.zeros_like(first_hits)
        self.second_ray_normals_w[env_ids] = torch.zeros_like(first_hits)
        self.second_ray_hit_valid[env_ids] = torch.zeros_like(first_valid)
        selected_hits = first_hits
        selected_normals = first_normals
        selected_valid = first_valid
        selected_depth = first_depth
        second_hits = None
        second_depth = None
        second_normal = None
        second_valid = None

        needs_second_hit = (
            ray_hit_indices is not None
            or ray_hit_index == 2
            or bool(getattr(self.cfg, "debug_viz_link_surfaces", False))
        )
        if needs_second_hit:
            eps = max(0.0, float(getattr(self.cfg, "second_hit_epsilon", 1.0e-5)))
            first_depth_clean = torch.nan_to_num(first_depth, nan=0.0, posinf=0.0, neginf=0.0)
            second_starts = self._ray_starts_w[env_ids] + self._ray_directions_w[env_ids] * (
                first_depth_clean.unsqueeze(-1) + eps
            )
            second_hits, second_depth, second_normal, _, second_mesh_ids = raycast_dynamic_meshes(
                second_starts,
                self._ray_directions_w[env_ids],
                mesh_ids_wp=self._mesh_ids_wp,
                max_dist=self.cfg.max_distance,
                mesh_positions_w=self._mesh_positions_w[env_ids],
                mesh_orientations_w=self._mesh_orientations_w[env_ids],
                return_distance=True,
                return_normal=True,
                return_mesh_id=bool(getattr(self.cfg, "update_mesh_ids", False)),
            )
            second_total = first_depth + eps + second_depth
            second_valid = first_valid & torch.isfinite(second_depth) & (second_total <= float(self.cfg.max_distance))
            self.second_ray_hits_w[env_ids] = torch.where(
                second_valid.unsqueeze(-1), second_hits, torch.zeros_like(second_hits)
            )
            second_normals = _normalize_hit_normals(second_normal, second_valid)
            self.second_ray_normals_w[env_ids] = second_normals
            self.second_ray_hit_valid[env_ids] = second_valid
            if bool(getattr(self.cfg, "debug_viz_link_surfaces", False)):
                self._update_link_surfaces_viz_usd(env_ids, first_hits, first_valid, second_hits, second_valid)

        if ray_hit_indices is not None:
            if second_hits is None or second_depth is None or second_valid is None:
                raise RuntimeError("Per-ray surface hit selection requires the second raycast.")
            use_second = (ray_hit_indices == 2).view(1, -1).expand_as(first_valid)
            use_first_fallback = bool(getattr(self.cfg, "use_first_hit_fallback", False))
            fallback_valid = first_valid if use_first_fallback else torch.zeros_like(first_valid)
            selected_second_valid = second_valid | fallback_valid
            selected_second_depth = torch.where(
                second_valid,
                second_total,
                torch.where(fallback_valid, first_depth, torch.full_like(first_depth, torch.inf)),
            )
            selected_second_hits = torch.where(
                second_valid.unsqueeze(-1),
                second_hits,
                torch.where(
                    fallback_valid.unsqueeze(-1),
                    first_hits,
                    torch.zeros_like(second_hits),
                ),
            )
            selected_second_normals = torch.where(
                second_valid.unsqueeze(-1),
                second_normals,
                torch.where(
                    fallback_valid.unsqueeze(-1),
                    first_normals,
                    torch.zeros_like(second_normals),
                ),
            )
            selected_valid = torch.where(use_second, selected_second_valid, first_valid)
            selected_depth = torch.where(use_second, selected_second_depth, first_depth)
            selected_hits = torch.where(
                use_second.unsqueeze(-1),
                selected_second_hits,
                torch.where(
                    first_valid.unsqueeze(-1),
                    first_hits,
                    torch.zeros_like(first_hits),
                ),
            )
            selected_normals = torch.where(
                use_second.unsqueeze(-1),
                selected_second_normals,
                first_normals,
            )
            ray_depth = selected_depth
            self.ray_hits_w[env_ids] = selected_hits
            if bool(getattr(self.cfg, "update_mesh_ids", False)) and second_mesh_ids is not None:
                invalid_mesh_ids = torch.full_like(second_mesh_ids, -1)
                selected_second_mesh_ids = torch.where(
                    second_valid.unsqueeze(-1),
                    second_mesh_ids,
                    torch.where(
                        fallback_valid.unsqueeze(-1),
                        ray_mesh_ids,
                        invalid_mesh_ids,
                    ),
                )
                ray_mesh_ids = torch.where(
                    use_second.unsqueeze(-1),
                    selected_second_mesh_ids,
                    ray_mesh_ids,
                )
        elif ray_hit_index == 2:
            if second_hits is None or second_depth is None or second_valid is None:
                raise RuntimeError("Second hit raycast was not computed.")
            use_first_fallback = bool(getattr(self.cfg, "use_first_hit_fallback", False))
            fallback_valid = first_valid if use_first_fallback else torch.zeros_like(first_valid)
            selected_valid = second_valid | fallback_valid
            selected_depth = torch.where(
                second_valid,
                second_total,
                torch.where(fallback_valid, first_depth, torch.full_like(first_depth, torch.inf)),
            )
            selected_hits = torch.where(
                second_valid.unsqueeze(-1),
                second_hits,
                torch.where(
                    fallback_valid.unsqueeze(-1),
                    first_hits,
                    torch.zeros_like(second_hits),
                ),
            )
            selected_normals = torch.where(
                second_valid.unsqueeze(-1),
                second_normals,
                torch.where(
                    fallback_valid.unsqueeze(-1),
                    first_normals,
                    torch.zeros_like(second_normals),
                ),
            )
            ray_depth = selected_depth
            self.ray_hits_w[env_ids] = selected_hits
            if bool(getattr(self.cfg, "update_mesh_ids", False)) and second_mesh_ids is not None:
                ray_mesh_ids = torch.where(second_valid.unsqueeze(-1), second_mesh_ids, torch.full_like(second_mesh_ids, -1))
        elif ray_hit_index != 1:
            raise ValueError(f"SharpaTacmapLinkSurface only supports ray_hit_index 1 or 2, got {ray_hit_index}")
        else:
            self.ray_hits_w[env_ids] = torch.where(first_valid.unsqueeze(-1), first_hits, torch.zeros_like(first_hits))

        self.ray_normals_w[env_ids] = selected_normals
        self.ray_hit_valid[env_ids] = selected_valid

        export_ray_depth = torch.where(torch.isfinite(ray_depth), ray_depth, torch.zeros_like(ray_depth))
        raw_ray_depth = export_ray_depth.clone()

        if "distance_along_normal_raw" in self._data.output:
            self._data.output["distance_along_normal_raw"][env_ids] = raw_ray_depth.view(-1, self.num_rays, 1)

        if "distance_along_normal" in self._data.output:
            export_quantize = deform_quantize(export_ray_depth.clone().view(-1, self.num_rays, 1))
            self._data.output["distance_along_normal"][env_ids] = export_quantize.view(-1, self.num_rays, 1)

        if self.cfg.update_mesh_ids and ray_mesh_ids is not None:
            self._data.image_mesh_ids[env_ids] = ray_mesh_ids.view(-1, self.num_rays, 1)

        if bool(getattr(self.cfg, "debug_viz_hits", False)) and not bool(
            getattr(self.cfg, "debug_viz_hits_external_mask", False)
        ):
            self._update_hit_points_viz_usd(env_ids, raw_ray_depth)
        if bool(getattr(self.cfg, "debug_viz_rays", False)):
            self._update_rays_viz_usd(env_ids, raw_ray_depth)

    def update_hit_points_viz_mask(self, valid_mask, env_id: int = 0):
        """Display only externally selected hit points, e.g. true penetration points."""
        if not hasattr(self, "_hit_viz_instancer"):
            return

        env_id = max(0, min(int(env_id), int(self._view.count) - 1))
        if not isinstance(valid_mask, torch.Tensor):
            valid = torch.as_tensor(valid_mask, device=self._device, dtype=torch.bool)
        else:
            valid = valid_mask.to(device=self._device, dtype=torch.bool)
        valid = valid.reshape(-1)
        if valid.numel() != self.num_rays:
            raise ValueError(f"valid_mask has {valid.numel()} values, expected {self.num_rays}")

        valid = valid[self._viz_indices]
        hits = self.ray_hits_w[env_id, self._viz_indices, :]
        hits = torch.where(valid.unsqueeze(-1), hits, torch.zeros_like(hits))
        pts = hits.detach().cpu().numpy()
        self._hit_viz_instancer.GetPositionsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts])
        )

    def _setup_link_surfaces_viz_usd(self, env_id: int = 0):
        stage = omni_usd.get_context().get_stage()
        self._surface_viz_env_id = max(0, min(int(env_id), int(self._view.count) - 1))
        sensor_name = self.cfg.prim_path.rstrip("/").split("/")[-1]
        root = Sdf.Path(f"/Visuals/TacmapLinkSurfaces/env_{self._surface_viz_env_id}/{sensor_name}")

        self._back_surface_viz_instancer = self._make_point_instancer(
            stage,
            root.AppendPath("BackEntryRed"),
            color=(1.0, 0.0, 0.0),
        )
        self._front_surface_viz_instancer = self._make_point_instancer(
            stage,
            root.AppendPath("FrontExitBlue"),
            color=(0.0, 0.2, 1.0),
        )

    def _make_point_instancer(self, stage, path: Sdf.Path, *, color: tuple[float, float, float]):
        UsdGeom.Xform.Define(stage, path.GetParentPath())
        inst = UsdGeom.PointInstancer.Define(stage, path)
        inst.CreatePositionsAttr()
        inst.CreateProtoIndicesAttr()
        inst.CreateScalesAttr()

        proto_root = path.AppendPath("Prototypes")
        UsdGeom.Xform.Define(stage, proto_root)
        sphere_path = proto_root.AppendPath("Point")
        sphere = UsdGeom.Sphere.Define(stage, sphere_path)
        sphere.GetRadiusAttr().Set(float(getattr(self.cfg, "debug_viz_point_radius", 0.0008)))
        UsdGeom.Gprim(sphere.GetPrim()).CreateDisplayColorAttr(
            Vt.Vec3fArray([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
        )

        k = int(self._viz_count)
        inst.CreatePrototypesRel().SetTargets([sphere_path])
        inst.GetProtoIndicesAttr().Set(Vt.IntArray([0] * k))
        inst.GetPositionsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 0.0)] * k))
        inst.GetScalesAttr().Set(Vt.Vec3fArray([Gf.Vec3f(1.0, 1.0, 1.0)] * k))
        return inst

    def _update_link_surfaces_viz_usd(
        self,
        env_ids: torch.Tensor,
        first_hits: torch.Tensor,
        first_valid: torch.Tensor,
        second_hits: torch.Tensor,
        second_valid: torch.Tensor,
    ):
        if not hasattr(self, "_back_surface_viz_instancer") or not hasattr(self, "_front_surface_viz_instancer"):
            return

        env_id = int(self._surface_viz_env_id)
        matches = torch.nonzero(env_ids == env_id, as_tuple=False).flatten()
        if matches.numel() == 0:
            return
        local_i = int(matches[0].detach().cpu().item())

        back_hits = first_hits[local_i, self._viz_indices, :]
        back_valid = first_valid[local_i, self._viz_indices]
        back_hits = torch.where(back_valid.unsqueeze(-1), back_hits, torch.zeros_like(back_hits))
        back_pts = back_hits.detach().cpu().numpy()
        self._back_surface_viz_instancer.GetPositionsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in back_pts])
        )

        front_hits = second_hits[local_i, self._viz_indices, :]
        front_valid = second_valid[local_i, self._viz_indices]
        front_hits = torch.where(front_valid.unsqueeze(-1), front_hits, torch.zeros_like(front_hits))
        front_pts = front_hits.detach().cpu().numpy()
        self._front_surface_viz_instancer.GetPositionsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in front_pts])
        )

    def _compute_view_world_poses(self, view, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(view, (XformPrimView, _UsdXformPoseView)):
            env_ids = self._normalize_env_ids(env_ids)
            cache = getattr(view, "_tacmap_cached_world_pose", None)
            if cache is None or cache[0].shape[0] != view.count:
                positions = []
                quaternions = []
                for prim in view.prims:
                    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    translation = transform.ExtractTranslation()
                    rotation_quat = transform.ExtractRotation().GetQuat()
                    rotation_imag = rotation_quat.GetImaginary()
                    positions.append((float(translation[0]), float(translation[1]), float(translation[2])))
                    quaternions.append(
                        (
                            float(rotation_quat.GetReal()),
                            float(rotation_imag[0]),
                            float(rotation_imag[1]),
                            float(rotation_imag[2]),
                        )
                    )
                pos_all = torch.tensor(positions, device=self._device, dtype=torch.float32)
                quat_all = torch.tensor(quaternions, device=self._device, dtype=torch.float32)
                cache = (pos_all, quat_all)
                setattr(view, "_tacmap_cached_world_pose", cache)
            pos_w, quat_w = cache[0][env_ids], cache[1][env_ids]
        elif isinstance(view, physx.ArticulationView):
            pos_w, quat_w = view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = math_utils.convert_quat(quat_w, to="wxyz")
        elif isinstance(view, physx.RigidBodyView):
            pos_w, quat_w = view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = math_utils.convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(view)}")
        return pos_w.clone(), quat_w.clone()

    def _setup_hit_points_viz_usd(self, env_id: int = 0):
        stage = omni_usd.get_context().get_stage()
        self._hit_viz_env_id = max(0, min(int(env_id), int(self._view.count) - 1))
        sensor_name = self.cfg.prim_path.rstrip("/").split("/")[-1]
        self._hit_viz_path = Sdf.Path(f"/Visuals/TacmapLinkSurfaceHits/env_{self._hit_viz_env_id}/{sensor_name}")

        UsdGeom.Xform.Define(stage, self._hit_viz_path.GetParentPath())
        inst = UsdGeom.PointInstancer.Define(stage, self._hit_viz_path)
        inst.CreatePositionsAttr()
        inst.CreateProtoIndicesAttr()
        inst.CreateScalesAttr()

        proto_root = self._hit_viz_path.AppendPath("Prototypes")
        UsdGeom.Xform.Define(stage, proto_root)
        sphere_path = proto_root.AppendPath("HitPoint")
        sphere = UsdGeom.Sphere.Define(stage, sphere_path)
        sphere.GetRadiusAttr().Set(float(getattr(self.cfg, "debug_viz_point_radius", 0.0008)))
        UsdGeom.Gprim(sphere.GetPrim()).CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.0, 0.0)]))

        inst.CreatePrototypesRel().SetTargets([sphere_path])
        k = int(self._viz_count)
        inst.GetProtoIndicesAttr().Set(Vt.IntArray([0] * k))
        inst.GetPositionsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 0.0)] * k))
        inst.GetScalesAttr().Set(Vt.Vec3fArray([Gf.Vec3f(1.0, 1.0, 1.0)] * k))
        self._hit_viz_instancer = inst

    def _update_hit_points_viz_usd(self, env_ids: torch.Tensor, ray_depth: torch.Tensor):
        if not hasattr(self, "_hit_viz_instancer"):
            return

        env_id = int(self._hit_viz_env_id)
        matches = torch.nonzero(env_ids == env_id, as_tuple=False).flatten()
        if matches.numel() == 0:
            return
        local_i = int(matches[0].detach().cpu().item())

        hits = self.ray_hits_w[env_id, self._viz_indices, :]
        depth = ray_depth[local_i, self._viz_indices]
        valid = torch.isfinite(depth) & (depth > 0)
        hits = torch.where(valid.unsqueeze(-1), hits, torch.zeros_like(hits))
        pts = hits.detach().cpu().numpy()
        self._hit_viz_instancer.GetPositionsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts])
        )

    def _setup_rays_viz_usd(self, env_id: int = 0):
        stage = omni_usd.get_context().get_stage()
        self._ray_viz_env_id = max(0, min(int(env_id), int(self._view.count) - 1))
        sensor_name = self.cfg.prim_path.rstrip("/").split("/")[-1]
        self._ray_viz_path = Sdf.Path(f"/Visuals/TacmapLinkSurfaceRays/env_{self._ray_viz_env_id}/{sensor_name}")

        UsdGeom.Xform.Define(stage, self._ray_viz_path.GetParentPath())
        curves = UsdGeom.BasisCurves.Define(stage, self._ray_viz_path)
        curves.CreateTypeAttr("linear")
        curves.CreateBasisAttr("bezier")
        curves.CreateCurveVertexCountsAttr(Vt.IntArray([2] * int(self._viz_count)))
        curves.CreateWidthsAttr(Vt.FloatArray([float(getattr(self.cfg, "debug_viz_ray_width", 0.00006))]))
        UsdGeom.Gprim(curves.GetPrim()).CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.85, 0.0)]))
        zero_points = [Gf.Vec3f(0.0, 0.0, 0.0)] * int(self._viz_count * 2)
        curves.CreatePointsAttr(Vt.Vec3fArray(zero_points))
        self._ray_viz_curves = curves

    def _update_rays_viz_usd(self, env_ids: torch.Tensor, ray_depth: torch.Tensor):
        if not hasattr(self, "_ray_viz_curves"):
            return

        env_id = int(self._ray_viz_env_id)
        matches = torch.nonzero(env_ids == env_id, as_tuple=False).flatten()
        if matches.numel() == 0:
            return
        local_i = int(matches[0].detach().cpu().item())

        starts = self._ray_starts_w[env_id, self._viz_indices, :]
        dirs = self._ray_directions_w[env_id, self._viz_indices, :]
        depth = ray_depth[local_i, self._viz_indices]
        fallback_len = min(float(getattr(self.cfg, "debug_viz_ray_length", 0.008)), float(self.cfg.max_distance))
        viz_len = torch.where(torch.isfinite(depth) & (depth > 0), depth, torch.full_like(depth, fallback_len))
        ends = starts + dirs * viz_len.unsqueeze(-1)
        segments = torch.stack((starts, ends), dim=1).reshape(-1, 3)
        pts = segments.detach().cpu().numpy()
        self._ray_viz_curves.GetPointsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts])
        )


def _normalize_hit_normals(normals: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    normals = torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    norm = torch.linalg.norm(normals, dim=-1, keepdim=True)
    normalized = normals / norm.clamp_min(1.0e-8)
    return torch.where(valid.unsqueeze(-1), normalized, torch.zeros_like(normalized))


def _axis_vec(axis: str, device: str) -> torch.Tensor:
    axis = axis.lower()
    sign = -1.0 if axis.startswith("-") else 1.0
    name = axis[-1]
    if name == "x":
        vec = (sign, 0.0, 0.0)
    elif name == "y":
        vec = (0.0, sign, 0.0)
    elif name == "z":
        vec = (0.0, 0.0, sign)
    else:
        raise ValueError(f"Unsupported axis: {axis!r}")
    return torch.tensor(vec, dtype=torch.float32, device=device)


def _direction_vec(direction, fallback_axis: str, device: str) -> torch.Tensor:
    if direction is None:
        return _axis_vec(fallback_axis, device)
    vec = torch.tensor(tuple(float(v) for v in direction), dtype=torch.float32, device=device)
    norm = torch.linalg.norm(vec)
    if float(norm.detach().cpu().item()) < 1.0e-8:
        return _axis_vec(fallback_axis, device)
    return vec / norm


def _orthonormal_grid_axes(ray_axis: torch.Tensor, u_axis: torch.Tensor, v_axis: torch.Tensor):
    ray_axis = ray_axis / torch.clamp(torch.linalg.norm(ray_axis), min=1.0e-8)

    u_axis = u_axis - torch.dot(u_axis, ray_axis) * ray_axis
    u_norm = torch.linalg.norm(u_axis)
    if float(u_norm.detach().cpu().item()) < 1.0e-8:
        candidates = (
            torch.tensor((0.0, 1.0, 0.0), dtype=torch.float32, device=ray_axis.device),
            torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32, device=ray_axis.device),
            torch.tensor((1.0, 0.0, 0.0), dtype=torch.float32, device=ray_axis.device),
        )
        for candidate in candidates:
            projected = candidate - torch.dot(candidate, ray_axis) * ray_axis
            if float(torch.linalg.norm(projected).detach().cpu().item()) >= 1.0e-8:
                u_axis = projected
                u_norm = torch.linalg.norm(u_axis)
                break
    u_axis = u_axis / torch.clamp(u_norm, min=1.0e-8)

    v_axis = v_axis - torch.dot(v_axis, ray_axis) * ray_axis - torch.dot(v_axis, u_axis) * u_axis
    v_norm = torch.linalg.norm(v_axis)
    if float(v_norm.detach().cpu().item()) < 1.0e-8:
        v_axis = torch.linalg.cross(ray_axis, u_axis)
        v_norm = torch.linalg.norm(v_axis)
    v_axis = v_axis / torch.clamp(v_norm, min=1.0e-8)
    return u_axis, v_axis
