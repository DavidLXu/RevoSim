"""Curved-surface HydroShear-style marker visualization.

This adapter keeps the FOTS marker layout, but computes marker motion in 3D on
the curved touch-link surface.  It uses TacMap link-surface penetration as the
normal indentation signal and all ray-visible object hit points as the shear
source points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


_MARKER_GREEN = (0, 255, 0)


@dataclass
class CurvedHydroShearOutput:
    marker_flow: torch.Tensor
    marker_images: np.ndarray
    original_marker_images: np.ndarray
    debug_marker_displacement_images: np.ndarray
    debug_sdf_images: np.ndarray
    debug_projection_images: np.ndarray
    debug_mdilate_images: np.ndarray
    debug_object_sample_points_w: list[np.ndarray]
    debug_all_object_sample_points_w: list[np.ndarray]
    debug_marker_points_w: list[np.ndarray]
    debug_marker_normals_w: list[np.ndarray]
    debug_marker_row_axes_w: list[np.ndarray]
    debug_marker_col_axes_w: list[np.ndarray]
    displacement_m: torch.Tensor
    max_depth_mm: torch.Tensor
    active_markers: torch.Tensor
    active_object_samples: torch.Tensor
    max_marker_motion_px: torch.Tensor


@dataclass
class RevoCurvedHydroShearCfg:
    width: int = 240
    height: int = 240
    marker_cols: int = 11
    marker_rows: int = 9
    marker_margin_x: float = 15.0
    marker_margin_y: float = 26.0
    marker_uv: np.ndarray | None = None
    contact_threshold_mm: float = 0.02
    lambda_dilate: float = 100000.0
    lambda_shear: float = 100000.0
    dilate_scale: float = 30.0
    shear_scale: float = 50.0
    hydrosoft_k: float = 1.0
    hydrosoft_e: float = 1.0
    hydrosoft_area: float = 1.0
    hydrosoft_mu: float = 0.5
    arrow_scale_px_per_m: float = 24000.0
    min_arrow_length_px: float = 0.0
    marker_radius: int = 3
    arrow_thickness: int = 2
    arrow_tip_length: float = 0.24
    show_depth_background: bool = False
    depth_background_max_mm: float = 8.0
    object_chunk_size: int = 8192
    batch_slot_chunk_size: int = 64
    curved_surface_lookup_chunk_size: int = 1024
    object_sample_reference_count: int = 4096
    object_sample_roi_count: int = 256
    object_sample_roi_boundary_padding_cells: float = 1.0
    object_sample_roi_surface_margin_m: float = 0.003
    object_sample_roi_invalid_depth_ratio: float = 0.5
    object_sample_roi_replace_margin_m: float = 0.0002
    object_sample_points_l: np.ndarray | None = None
    shear_reference_starts_l: np.ndarray | torch.Tensor | None = None
    shear_reference_directions_l: np.ndarray | torch.Tensor | None = None
    shear_reference_depth_m: np.ndarray | torch.Tensor | None = None
    shear_reference_valid: np.ndarray | torch.Tensor | None = None
    shear_coarse_xy_camera_m: np.ndarray | torch.Tensor | None = None
    shear_reference_bounds_camera_m: np.ndarray | torch.Tensor | None = None
    shear_reference_row_axis_camera: tuple[int, ...] | None = None
    device: str | None = None


class RevoCurvedHydroShearAdapter:
    """Approximate HydroShear marker motion on the Revo curved tactile pad."""

    def __init__(self, cfg: RevoCurvedHydroShearCfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._marker_uv = self._make_marker_uv(cfg)
        self._marker_uv_t = torch.as_tensor(self._marker_uv, dtype=torch.float32, device=self.device)
        self._object_sample_points_l = (
            None
            if cfg.object_sample_points_l is None
            else torch.as_tensor(cfg.object_sample_points_l, dtype=torch.float32, device=self.device).reshape(-1, 3)
        )
        self._object_sample_area_scale = 1.0
        if self._object_sample_points_l is not None:
            # Match the original fixed-sample behavior when the object cloud is
            # sparse, but avoid over-counting when dense sampling is enabled.
            self._object_sample_area_scale = min(
                1.0,
                float(cfg.object_sample_reference_count) / max(1.0, float(self._object_sample_points_l.shape[0])),
            )
        reference_values = (
            cfg.shear_reference_starts_l,
            cfg.shear_reference_directions_l,
            cfg.shear_reference_depth_m,
            cfg.shear_reference_valid,
            cfg.shear_coarse_xy_camera_m,
            cfg.shear_reference_bounds_camera_m,
            cfg.shear_reference_row_axis_camera,
        )
        reference_count = sum(value is not None for value in reference_values)
        if reference_count not in (0, len(reference_values)):
            raise ValueError("All curved Shear reference fields must be provided together")
        self._shear_reference_starts_l: torch.Tensor | None = None
        self._shear_reference_directions_l: torch.Tensor | None = None
        self._shear_reference_depth_m: torch.Tensor | None = None
        self._shear_reference_valid: torch.Tensor | None = None
        self._shear_coarse_xy_camera_m: torch.Tensor | None = None
        self._shear_reference_bounds_camera_m: torch.Tensor | None = None
        self._shear_reference_row_axis_camera: torch.Tensor | None = None
        if reference_count:
            starts = torch.as_tensor(cfg.shear_reference_starts_l, device=self.device, dtype=torch.float32)
            directions = torch.as_tensor(
                cfg.shear_reference_directions_l,
                device=self.device,
                dtype=torch.float32,
            )
            reference_depth = torch.as_tensor(
                cfg.shear_reference_depth_m,
                device=self.device,
                dtype=torch.float32,
            )
            reference_valid = torch.as_tensor(
                cfg.shear_reference_valid,
                device=self.device,
                dtype=torch.bool,
            )
            coarse_xy = torch.as_tensor(
                cfg.shear_coarse_xy_camera_m,
                device=self.device,
                dtype=torch.float32,
            )
            bounds = torch.as_tensor(
                cfg.shear_reference_bounds_camera_m,
                device=self.device,
                dtype=torch.float32,
            )
            row_axis = torch.as_tensor(
                cfg.shear_reference_row_axis_camera,
                device=self.device,
                dtype=torch.long,
            ).reshape(-1)
            if starts.ndim != 4 or starts.shape[-1] != 3:
                raise ValueError("Curved Shear reference starts must have shape [sensors,H,W,3]")
            if directions.shape != starts.shape:
                raise ValueError("Curved Shear reference directions must match reference starts")
            if reference_depth.shape != starts.shape[:-1] or reference_valid.shape != starts.shape[:-1]:
                raise ValueError("Curved Shear reference depth/valid shapes must match reference starts")
            if coarse_xy.ndim != 4 or coarse_xy.shape[-1] != 2:
                raise ValueError("Curved Shear coarse camera coordinates must have shape [sensors,H,W,2]")
            sensor_count = int(starts.shape[0])
            if (
                int(coarse_xy.shape[0]) != sensor_count
                or tuple(bounds.shape) != (sensor_count, 4)
                or int(row_axis.shape[0]) != sensor_count
            ):
                raise ValueError("Curved Shear reference fields must contain the same sensor count")
            if not torch.all((row_axis == 0) | (row_axis == 1)):
                raise ValueError("Curved Shear reference row axes must be 0 or 1")
            self._shear_reference_starts_l = starts
            self._shear_reference_directions_l = directions
            self._shear_reference_depth_m = reference_depth
            self._shear_reference_valid = reference_valid
            self._shear_coarse_xy_camera_m = coarse_xy
            self._shear_reference_bounds_camera_m = bounds
            self._shear_reference_row_axis_camera = row_axis
        self._hydrosoft_forces: list[torch.Tensor | None] = []
        self._prev_sdf: list[torch.Tensor | None] = []
        self._prev_indenter_points: list[torch.Tensor | None] = []
        self._prev_normals: list[torch.Tensor | None] = []
        self._batch_hydrosoft_forces: torch.Tensor | None = None
        self._batch_prev_sdf: torch.Tensor | None = None
        self._batch_prev_indenter_points: torch.Tensor | None = None
        self._batch_prev_normals: torch.Tensor | None = None
        self._batch_state_valid: torch.Tensor | None = None
        self._batch_prev_sample_ids: torch.Tensor | None = None

    def reset(self) -> None:
        self._hydrosoft_forces.clear()
        self._prev_sdf.clear()
        self._prev_indenter_points.clear()
        self._prev_normals.clear()
        self._batch_hydrosoft_forces = None
        self._batch_prev_sdf = None
        self._batch_prev_indenter_points = None
        self._batch_prev_normals = None
        self._batch_state_valid = None
        self._batch_prev_sample_ids = None

    def step_displacement_only(
        self,
        penetration_depth_m: np.ndarray | torch.Tensor,
        surface_points_w: np.ndarray | torch.Tensor,
        surface_valid: np.ndarray | torch.Tensor,
        object_points_w: np.ndarray | torch.Tensor,
        object_valid: np.ndarray | torch.Tensor,
        object_pose_wxyz: np.ndarray | torch.Tensor | None = None,
        surface_raw_m: np.ndarray | torch.Tensor | None = None,
        *,
        surface_normals_w: np.ndarray | torch.Tensor | None = None,
        ray_directions_w: np.ndarray | torch.Tensor | None = None,
        marker_points_w: np.ndarray | torch.Tensor | None = None,
        marker_normals_w: np.ndarray | torch.Tensor | None = None,
        marker_valid: np.ndarray | torch.Tensor | None = None,
        marker_depth_m: np.ndarray | torch.Tensor | None = None,
        marker_depth_valid: np.ndarray | torch.Tensor | None = None,
        dilation_source_depth_m: np.ndarray | torch.Tensor | None = None,
        dilation_source_roi_norm: np.ndarray | torch.Tensor | None = None,
        dilation_source_active: np.ndarray | torch.Tensor | None = None,
        surface_frame_pose_wxyz: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute marker displacement only, keeping the RL path off image rendering."""

        depth = self._to_tensor(penetration_depth_m)
        if depth.ndim == 2:
            depth = depth[None, ...]

        surface_points = self._to_tensor(surface_points_w)
        if surface_points.ndim == 3:
            surface_points = surface_points[None, ...]
        surface_valid_arr = self._to_tensor(surface_valid, dtype=torch.bool)
        if surface_valid_arr.ndim == 2:
            surface_valid_arr = surface_valid_arr[None, ...]

        object_points = self._to_tensor(object_points_w)
        if object_points.ndim == 3:
            object_points = object_points[None, ...]
        object_valid_arr = self._to_tensor(object_valid, dtype=torch.bool)
        if object_valid_arr.ndim == 2:
            object_valid_arr = object_valid_arr[None, ...]

        surface_raw = None
        if surface_raw_m is not None:
            surface_raw = self._to_tensor(surface_raw_m)
            if surface_raw.ndim == 2:
                surface_raw = surface_raw[None, ...]

        surface_normals = None
        if surface_normals_w is not None:
            surface_normals = self._to_tensor(surface_normals_w)
            if surface_normals.ndim == 3:
                surface_normals = surface_normals[None, ...]

        ray_directions = None
        if ray_directions_w is not None:
            ray_directions = self._to_tensor(ray_directions_w).reshape(-1, 3)

        marker_points = None
        marker_normals = None
        marker_valid_arr = None
        marker_override_count = sum(value is not None for value in (marker_points_w, marker_normals_w, marker_valid))
        if marker_override_count not in (0, 3):
            raise ValueError("marker_points_w, marker_normals_w, and marker_valid must be provided together")
        if marker_override_count:
            marker_points = self._to_tensor(marker_points_w)
            marker_normals = self._to_tensor(marker_normals_w)
            marker_valid_arr = self._to_tensor(marker_valid, dtype=torch.bool)
            if marker_points.ndim == 2:
                marker_points = marker_points[None, ...]
            if marker_normals.ndim == 2:
                marker_normals = marker_normals[None, ...]
            if marker_valid_arr.ndim == 1:
                marker_valid_arr = marker_valid_arr[None, ...]

        marker_depth = None
        marker_depth_valid_arr = None
        marker_depth_override_count = sum(value is not None for value in (marker_depth_m, marker_depth_valid))
        if marker_depth_override_count not in (0, 2):
            raise ValueError("marker_depth_m and marker_depth_valid must be provided together")
        if marker_depth_override_count and marker_override_count != 3:
            raise ValueError("independent marker depth requires calibrated marker point overrides")
        if marker_depth_override_count:
            marker_depth = self._to_tensor(marker_depth_m)
            marker_depth_valid_arr = self._to_tensor(marker_depth_valid, dtype=torch.bool)
            if marker_depth.ndim == 1:
                marker_depth = marker_depth[None, ...]
            if marker_depth_valid_arr.ndim == 1:
                marker_depth_valid_arr = marker_depth_valid_arr[None, ...]

        dilation_source_depth = None
        dilation_source_roi = None
        dilation_source_active_arr = None
        dilation_source_count = sum(
            value is not None
            for value in (dilation_source_depth_m, dilation_source_roi_norm, dilation_source_active)
        )
        if dilation_source_count not in (0, 3):
            raise ValueError(
                "dilation_source_depth_m, dilation_source_roi_norm, and dilation_source_active "
                "must be provided together"
            )
        if dilation_source_count:
            dilation_source_depth = self._to_tensor(dilation_source_depth_m)
            dilation_source_roi = self._to_tensor(dilation_source_roi_norm)
            dilation_source_active_arr = self._to_tensor(dilation_source_active, dtype=torch.bool)
            if dilation_source_depth.ndim == 2:
                dilation_source_depth = dilation_source_depth[None, ...]
            if dilation_source_roi.ndim == 1:
                dilation_source_roi = dilation_source_roi[None, ...]
            if dilation_source_active_arr.ndim == 0:
                dilation_source_active_arr = dilation_source_active_arr[None]

        surface_frame_pose_input = None
        if surface_frame_pose_wxyz is not None:
            surface_frame_pose_input = self._to_tensor(surface_frame_pose_wxyz)
            if surface_frame_pose_input.ndim == 1:
                surface_frame_pose_input = surface_frame_pose_input.reshape(1, -1)
            else:
                surface_frame_pose_input = surface_frame_pose_input.reshape(
                    -1,
                    surface_frame_pose_input.shape[-1],
                )

        counts = [
            depth.shape[0],
            surface_points.shape[0],
            surface_valid_arr.shape[0],
            object_points.shape[0],
            object_valid_arr.shape[0],
        ]
        if surface_raw is not None:
            counts.append(surface_raw.shape[0])
        if surface_normals is not None:
            counts.append(surface_normals.shape[0])
        if ray_directions is not None:
            counts.append(ray_directions.shape[0])
        if marker_points is not None and marker_normals is not None and marker_valid_arr is not None:
            counts.extend((marker_points.shape[0], marker_normals.shape[0], marker_valid_arr.shape[0]))
        if marker_depth is not None and marker_depth_valid_arr is not None:
            counts.extend((marker_depth.shape[0], marker_depth_valid_arr.shape[0]))
        if (
            dilation_source_depth is not None
            and dilation_source_roi is not None
            and dilation_source_active_arr is not None
        ):
            counts.extend(
                (
                    dilation_source_depth.shape[0],
                    dilation_source_roi.shape[0],
                    dilation_source_active_arr.shape[0],
                )
            )
        if surface_frame_pose_input is not None and int(surface_frame_pose_input.shape[0]) > 1:
            counts.append(surface_frame_pose_input.shape[0])
        sensor_count = min(counts)
        self._ensure_prev_buffers(sensor_count)
        surface_frame_pose = (
            None
            if surface_frame_pose_input is None
            else self._object_pose_batch_t(surface_frame_pose_input, sensor_count)
        )

        if self._object_sample_points_l is not None and object_pose_wxyz is not None:
            object_pose = self._object_pose_batch_t(object_pose_wxyz, sensor_count)
            return self._step_displacement_only_object_samples_batched(
                depth[:sensor_count],
                surface_points[:sensor_count],
                surface_valid_arr[:sensor_count],
                object_points[:sensor_count],
                object_valid_arr[:sensor_count],
                None if surface_raw is None else surface_raw[:sensor_count],
                None if surface_normals is None else surface_normals[:sensor_count],
                object_pose,
                None if ray_directions is None else ray_directions[:sensor_count],
                None if marker_points is None else marker_points[:sensor_count],
                None if marker_normals is None else marker_normals[:sensor_count],
                None if marker_valid_arr is None else marker_valid_arr[:sensor_count],
                None if marker_depth is None else marker_depth[:sensor_count],
                None if marker_depth_valid_arr is None else marker_depth_valid_arr[:sensor_count],
                None if dilation_source_depth is None else dilation_source_depth[:sensor_count],
                None if dilation_source_roi is None else dilation_source_roi[:sensor_count],
                None if dilation_source_active_arr is None else dilation_source_active_arr[:sensor_count],
                surface_frame_pose,
            )

        if self._object_sample_points_l is None or marker_points is not None:
            return self._step_displacement_only_batched(
                depth[:sensor_count],
                surface_points[:sensor_count],
                surface_valid_arr[:sensor_count],
                object_points[:sensor_count],
                object_valid_arr[:sensor_count],
                None if surface_raw is None else surface_raw[:sensor_count],
                None if surface_normals is None else surface_normals[:sensor_count],
                None if marker_points is None else marker_points[:sensor_count],
                None if marker_normals is None else marker_normals[:sensor_count],
                None if marker_valid_arr is None else marker_valid_arr[:sensor_count],
                None if marker_depth is None else marker_depth[:sensor_count],
                None if marker_depth_valid_arr is None else marker_depth_valid_arr[:sensor_count],
                None if dilation_source_depth is None else dilation_source_depth[:sensor_count],
                None if dilation_source_roi is None else dilation_source_roi[:sensor_count],
                None if dilation_source_active_arr is None else dilation_source_active_arr[:sensor_count],
                surface_frame_pose,
            )

        displacements = []
        for i in range(sensor_count):
            object_pose_i = self._pose_for_sensor_t(object_pose_wxyz, i)
            displacements.append(
                self._step_one_displacement_only(
                    i,
                    depth[i],
                    surface_points[i],
                    surface_valid_arr[i],
                    object_points[i],
                    object_valid_arr[i],
                    object_pose_i,
                    None if surface_raw is None else surface_raw[i],
                    None if surface_normals is None else surface_normals[i],
                    None if ray_directions is None else ray_directions[i],
                )
            )
        if not displacements:
            return self._empty_tensor(0, self.cfg.marker_rows * self.cfg.marker_cols, 3)
        return torch.stack(displacements, dim=0)

    def step_original_displacement_only(
        self,
        penetration_depth_m: np.ndarray | torch.Tensor,
        surface_points_w: np.ndarray | torch.Tensor,
        surface_valid: np.ndarray | torch.Tensor,
        object_points_w: np.ndarray | torch.Tensor,
        object_valid: np.ndarray | torch.Tensor,
        object_pose_wxyz: np.ndarray | torch.Tensor | None = None,
        surface_raw_m: np.ndarray | torch.Tensor | None = None,
        *,
        surface_normals_w: np.ndarray | torch.Tensor | None = None,
        ray_directions_w: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the original Euclidean HydroShear marker field without rendering."""

        depth = self._to_tensor(penetration_depth_m)
        if depth.ndim == 2:
            depth = depth[None, ...]
        surface_points = self._to_tensor(surface_points_w)
        if surface_points.ndim == 3:
            surface_points = surface_points[None, ...]
        surface_valid_arr = self._to_tensor(surface_valid, dtype=torch.bool)
        if surface_valid_arr.ndim == 2:
            surface_valid_arr = surface_valid_arr[None, ...]
        object_points = self._to_tensor(object_points_w)
        if object_points.ndim == 3:
            object_points = object_points[None, ...]
        object_valid_arr = self._to_tensor(object_valid, dtype=torch.bool)
        if object_valid_arr.ndim == 2:
            object_valid_arr = object_valid_arr[None, ...]

        surface_raw = None
        if surface_raw_m is not None:
            surface_raw = self._to_tensor(surface_raw_m)
            if surface_raw.ndim == 2:
                surface_raw = surface_raw[None, ...]
        surface_normals = None
        if surface_normals_w is not None:
            surface_normals = self._to_tensor(surface_normals_w)
            if surface_normals.ndim == 3:
                surface_normals = surface_normals[None, ...]
        ray_directions = None
        if ray_directions_w is not None:
            ray_directions = self._to_tensor(ray_directions_w).reshape(-1, 3)

        counts = [
            depth.shape[0],
            surface_points.shape[0],
            surface_valid_arr.shape[0],
            object_points.shape[0],
            object_valid_arr.shape[0],
        ]
        if surface_raw is not None:
            counts.append(surface_raw.shape[0])
        if surface_normals is not None:
            counts.append(surface_normals.shape[0])
        if ray_directions is not None:
            counts.append(ray_directions.shape[0])
        sensor_count = min(counts)

        if self._object_sample_points_l is not None and object_pose_wxyz is not None:
            object_pose = self._object_pose_batch_t(object_pose_wxyz, sensor_count)
            return self._step_original_displacement_only_object_samples_batched(
                depth[:sensor_count],
                surface_points[:sensor_count],
                surface_valid_arr[:sensor_count],
                object_points[:sensor_count],
                object_valid_arr[:sensor_count],
                None if surface_raw is None else surface_raw[:sensor_count],
                None if surface_normals is None else surface_normals[:sensor_count],
                object_pose,
                None if ray_directions is None else ray_directions[:sensor_count],
            )

        return self._step_original_displacement_only_grid_batched(
            depth[:sensor_count],
            surface_points[:sensor_count],
            surface_valid_arr[:sensor_count],
            object_points[:sensor_count],
            object_valid_arr[:sensor_count],
            None if surface_normals is None else surface_normals[:sensor_count],
            None if ray_directions is None else ray_directions[:sensor_count],
        )

    def render_displacement_output(
        self,
        displacement_m: np.ndarray | torch.Tensor,
        penetration_depth_m: np.ndarray | torch.Tensor,
        surface_points_w: np.ndarray | torch.Tensor,
        surface_valid: np.ndarray | torch.Tensor,
        *,
        marker_points_w: np.ndarray | torch.Tensor | None = None,
        marker_normals_w: np.ndarray | torch.Tensor | None = None,
        marker_valid: np.ndarray | torch.Tensor | None = None,
        render_marker_images: bool = True,
        debug_batch_index: int | None = None,
    ) -> CurvedHydroShearOutput:
        """Render marker images from an already-computed training displacement.

        This method is intentionally stateless: it does not recompute HydroShear
        forces or update either the batched or per-sensor history buffers.  Callers
        that redraw ``marker_flow`` themselves may disable ``render_marker_images``;
        ``debug_batch_index`` keeps CPU debug geometry only for one flattened batch
        slot while preserving list indices for existing consumers.
        """

        displacement = self._to_tensor(displacement_m)
        if displacement.ndim == 2:
            displacement = displacement[None, ...]
        depth = self._to_tensor(penetration_depth_m)
        if depth.ndim == 2:
            depth = depth[None, ...]
        surface_points = self._to_tensor(surface_points_w)
        if surface_points.ndim == 3:
            surface_points = surface_points[None, ...]
        valid = self._to_tensor(surface_valid, dtype=torch.bool)
        if valid.ndim == 2:
            valid = valid[None, ...]

        batch_count = min(
            int(displacement.shape[0]),
            int(depth.shape[0]),
            int(surface_points.shape[0]),
            int(valid.shape[0]),
        )
        marker_count = int(self.cfg.marker_rows) * int(self.cfg.marker_cols)
        if batch_count <= 0:
            empty_images = np.zeros((0, int(self.cfg.height), int(self.cfg.width), 3), dtype=np.uint8)
            return CurvedHydroShearOutput(
                marker_flow=self._empty_tensor(0, 2, marker_count, 2),
                marker_images=empty_images,
                original_marker_images=empty_images.copy(),
                debug_marker_displacement_images=empty_images.copy(),
                debug_sdf_images=empty_images.copy(),
                debug_projection_images=empty_images.copy(),
                debug_mdilate_images=empty_images.copy(),
                debug_object_sample_points_w=[],
                debug_all_object_sample_points_w=[],
                debug_marker_points_w=[],
                debug_marker_normals_w=[],
                debug_marker_row_axes_w=[],
                debug_marker_col_axes_w=[],
                displacement_m=self._empty_tensor(0, marker_count, 3),
                max_depth_mm=self._empty_tensor(0),
                active_markers=self._empty_tensor(0),
                active_object_samples=self._empty_tensor(0),
                max_marker_motion_px=self._empty_tensor(0),
            )

        displacement = self._finite_tensor(displacement[:batch_count])
        if displacement.ndim != 3 or displacement.shape[-1] != 3:
            raise ValueError(f"Expected displacement shape (batch, markers, 3), got {tuple(displacement.shape)}")
        if int(displacement.shape[1]) != marker_count:
            resized = self._empty_tensor(batch_count, marker_count, 3)
            copy_count = min(marker_count, int(displacement.shape[1]))
            resized[:, :copy_count] = displacement[:, :copy_count]
            displacement = resized

        depth = torch.clamp(self._finite_tensor(depth[:batch_count]), min=0.0)
        surface_points = self._finite_tensor(surface_points[:batch_count])
        valid = valid[:batch_count].to(device=self.device, dtype=torch.bool)
        if (
            depth.ndim != 3
            or surface_points.ndim != 4
            or surface_points.shape[-1] != 3
            or valid.ndim != 3
            or tuple(surface_points.shape[1:3]) != tuple(depth.shape[1:])
            or tuple(valid.shape[1:]) != tuple(depth.shape[1:])
        ):
            raise ValueError(
                "Expected depth/valid (batch, rows, cols) and surface points (batch, rows, cols, 3), got "
                f"{tuple(depth.shape)}, {tuple(valid.shape)}, and {tuple(surface_points.shape)}"
            )

        height, width = int(depth.shape[1]), int(depth.shape[2])
        marker_uv_render = self._marker_uv_t
        marker_points_override = None if marker_points_w is None else self._to_tensor(marker_points_w)
        marker_normals_override = None if marker_normals_w is None else self._to_tensor(marker_normals_w)
        marker_valid_override = None if marker_valid is None else self._to_tensor(marker_valid, dtype=torch.bool)
        (
            _,
            marker_points,
            _,
            marker_valid_arr,
            marker_row_axes,
            marker_col_axes,
            marker_normals,
        ) = self._batched_marker_samples(
            depth,
            surface_points,
            valid,
            None,
            marker_points_override,
            marker_normals_override,
            marker_valid_override,
        )
        row_axes, col_axes = self._batched_uniform_marker_projection_axes(
            surface_points,
            valid,
            marker_row_axes,
            marker_col_axes,
        )
        rendered_displacement = torch.where(marker_valid_arr[..., None], displacement, torch.zeros_like(displacement))

        selected_debug_index = None
        if debug_batch_index is not None:
            selected_debug_index = max(0, min(int(debug_batch_index), batch_count - 1))

        flows: list[torch.Tensor] = []
        images: list[np.ndarray] = []
        for batch_index in range(batch_count):
            flow = self._flow_from_displacement_t(
                marker_uv_render,
                rendered_displacement[batch_index],
                row_axes[batch_index],
                col_axes[batch_index],
            )
            flows.append(flow)
            if bool(render_marker_images):
                images.append(
                    self._render_vector_field_image(
                        self._to_numpy(flow[:, marker_valid_arr[batch_index]]).astype(np.float32),
                        color=_MARKER_GREEN,
                        draw_points=True,
                    )
                )

        marker_flow = torch.stack(flows, dim=0)
        marker_images = (
            np.stack(images, axis=0)
            if images
            else np.zeros((0, int(self.cfg.height), int(self.cfg.width), 3), dtype=np.uint8)
        )
        blank_images = np.zeros_like(marker_images)
        active = marker_valid_arr & (torch.linalg.norm(rendered_displacement, dim=-1) > 1.0e-12)
        max_motion = torch.amax(torch.linalg.norm(marker_flow[:, 1] - marker_flow[:, 0], dim=-1), dim=1)
        empty_points = [np.zeros((0, 3), dtype=np.float32) for _ in range(batch_count)]
        debug_marker_points = [points.copy() for points in empty_points]
        debug_marker_normals = [points.copy() for points in empty_points]
        debug_marker_row_axes = [points.copy() for points in empty_points]
        debug_marker_col_axes = [points.copy() for points in empty_points]
        for index in range(batch_count):
            if selected_debug_index is not None and index != selected_debug_index:
                continue
            valid_index = marker_valid_arr[index]
            packed = torch.cat(
                (
                    marker_points[index],
                    marker_normals[index],
                    marker_row_axes[index],
                    marker_col_axes[index],
                ),
                dim=-1,
            )
            packed_np = self._to_numpy(packed[valid_index]).astype(np.float32)
            debug_marker_points[index] = packed_np[:, 0:3]
            debug_marker_normals[index] = packed_np[:, 3:6]
            debug_marker_row_axes[index] = packed_np[:, 6:9]
            debug_marker_col_axes[index] = packed_np[:, 9:12]
        return CurvedHydroShearOutput(
            marker_flow=marker_flow,
            marker_images=marker_images,
            original_marker_images=blank_images.copy(),
            debug_marker_displacement_images=blank_images.copy(),
            debug_sdf_images=blank_images.copy(),
            debug_projection_images=blank_images.copy(),
            debug_mdilate_images=blank_images.copy(),
            debug_object_sample_points_w=empty_points,
            debug_all_object_sample_points_w=[points.copy() for points in empty_points],
            debug_marker_points_w=debug_marker_points,
            debug_marker_normals_w=debug_marker_normals,
            debug_marker_row_axes_w=debug_marker_row_axes,
            debug_marker_col_axes_w=debug_marker_col_axes,
            displacement_m=displacement,
            max_depth_mm=torch.amax(depth, dim=(1, 2)) * 1000.0,
            active_markers=torch.count_nonzero(active, dim=1).to(dtype=torch.float32),
            active_object_samples=torch.zeros((batch_count,), dtype=torch.float32, device=self.device),
            max_marker_motion_px=max_motion.to(dtype=torch.float32),
        )

    def _step_displacement_only_batched(
        self,
        depth: torch.Tensor,
        surface_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        object_points_w: torch.Tensor,
        object_valid: torch.Tensor,
        surface_raw_m: torch.Tensor | None,
        surface_normals_w: torch.Tensor | None,
        marker_points_w: torch.Tensor | None = None,
        marker_normals_w: torch.Tensor | None = None,
        marker_valid_override: torch.Tensor | None = None,
        marker_depth_override: torch.Tensor | None = None,
        marker_depth_valid_override: torch.Tensor | None = None,
        dilation_source_depth: torch.Tensor | None = None,
        dilation_source_roi: torch.Tensor | None = None,
        dilation_source_active: torch.Tensor | None = None,
        surface_frame_pose_wxyz: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vectorized RL-only HydroShear path over all env/finger slots."""

        if depth.ndim != 3 or surface_points_w.ndim != 4 or object_points_w.ndim != 4:
            return self._empty_tensor(0, self.cfg.marker_rows * self.cfg.marker_cols, 3)

        depth = torch.clamp(self._finite_tensor(depth), min=0.0)
        surface_points_w = self._finite_tensor(surface_points_w)
        object_points_w = self._finite_tensor(object_points_w)
        surface_valid = surface_valid.to(device=self.device, dtype=torch.bool)
        object_valid = object_valid.to(device=self.device, dtype=torch.bool)
        surface_normals = self._finite_tensor(surface_normals_w) if surface_normals_w is not None else None

        batch_count, _, _ = depth.shape
        (
            marker_uv,
            marker_points,
            marker_depth,
            marker_valid,
            row_axes,
            col_axes,
            marker_normals,
        ) = self._batched_marker_samples(
            depth,
            surface_points_w,
            surface_valid,
            surface_normals,
            marker_points_w,
            marker_normals_w,
            marker_valid_override,
            marker_depth_override,
            marker_depth_valid_override,
        )
        marker_count = int(marker_uv.shape[0])
        if batch_count <= 0 or marker_count <= 0:
            return self._empty_tensor(batch_count, marker_count, 3)

        contact_threshold_m = float(self.cfg.contact_threshold_mm) * 1.0e-3
        marker_contact = marker_valid & (marker_depth > contact_threshold_m)

        ray_row_vec, ray_col_vec = self._batched_grid_metric_vectors(surface_points_w, surface_valid, row_axes, col_axes)

        marker_mdilate = self._compute_dilation_batched(
            marker_uv,
            marker_depth,
            marker_contact,
            ray_row_vec,
            ray_col_vec,
        )
        mdilate = self._select_dilation_source_batched(
            marker_mdilate,
            marker_uv,
            dilation_source_depth,
            dilation_source_roi,
            dilation_source_active,
            ray_row_vec,
            ray_col_vec,
            grid_width=int(depth.shape[2]),
            grid_height=int(depth.shape[1]),
        )
        mshear = self._compute_shear_from_grid_batched(
            marker_uv,
            marker_points,
            marker_normals,
            marker_valid,
            depth,
            surface_points_w,
            surface_valid,
            object_points_w,
            object_valid,
            ray_row_vec,
            ray_col_vec,
            surface_frame_pose_wxyz,
        )

        displacement = (mdilate + mshear).to(dtype=torch.float32)
        return torch.where(marker_valid[..., None], displacement, torch.zeros_like(displacement))

    def _step_displacement_only_object_samples_batched(
        self,
        depth: torch.Tensor,
        surface_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        object_points_w: torch.Tensor,
        object_valid: torch.Tensor,
        surface_raw_m: torch.Tensor | None,
        surface_normals_w: torch.Tensor | None,
        object_pose_wxyz: torch.Tensor,
        ray_directions_w: torch.Tensor | None,
        marker_points_w: torch.Tensor | None = None,
        marker_normals_w: torch.Tensor | None = None,
        marker_valid_override: torch.Tensor | None = None,
        marker_depth_override: torch.Tensor | None = None,
        marker_depth_valid_override: torch.Tensor | None = None,
        dilation_source_depth: torch.Tensor | None = None,
        dilation_source_roi: torch.Tensor | None = None,
        dilation_source_active: torch.Tensor | None = None,
        surface_frame_pose_wxyz: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vectorized RL path that preserves dense object-surface HydroShear samples."""

        if self._object_sample_points_l is None or int(self._object_sample_points_l.shape[0]) <= 0:
            return self._step_displacement_only_batched(
                depth,
                surface_points_w,
                surface_valid,
                object_points_w,
                object_valid,
                surface_raw_m,
                surface_normals_w,
                marker_points_w,
                marker_normals_w,
                marker_valid_override,
                marker_depth_override,
                marker_depth_valid_override,
                dilation_source_depth,
                dilation_source_roi,
                dilation_source_active,
                surface_frame_pose_wxyz,
            )
        if depth.ndim != 3 or surface_points_w.ndim != 4 or object_points_w.ndim != 4:
            return self._empty_tensor(0, self.cfg.marker_rows * self.cfg.marker_cols, 3)

        depth = torch.clamp(self._finite_tensor(depth), min=0.0)
        surface_points_w = self._finite_tensor(surface_points_w)
        object_points_w = self._finite_tensor(object_points_w)
        surface_valid = surface_valid.to(device=self.device, dtype=torch.bool)
        object_valid = object_valid.to(device=self.device, dtype=torch.bool)
        surface_raw = self._finite_tensor(surface_raw_m) if surface_raw_m is not None else depth
        surface_normals = self._finite_tensor(surface_normals_w) if surface_normals_w is not None else None

        batch_count, _, _ = depth.shape
        (
            marker_uv,
            marker_points,
            marker_depth,
            marker_valid,
            row_axes,
            col_axes,
            marker_normals,
        ) = self._batched_marker_samples(
            depth,
            surface_points_w,
            surface_valid,
            surface_normals,
            marker_points_w,
            marker_normals_w,
            marker_valid_override,
            marker_depth_override,
            marker_depth_valid_override,
        )
        marker_count = int(marker_uv.shape[0])
        if batch_count <= 0 or marker_count <= 0:
            return self._empty_tensor(batch_count, marker_count, 3)

        contact_threshold_m = float(self.cfg.contact_threshold_mm) * 1.0e-3
        marker_contact = marker_valid & (marker_depth > contact_threshold_m)

        ray_dir = self._batched_ray_direction(
            surface_points_w,
            object_points_w,
            surface_valid,
            object_valid,
            depth,
            ray_directions_w=ray_directions_w,
        )
        ray_row_vec, ray_col_vec = self._batched_ray_grid_metric_vectors(
            surface_points_w,
            surface_valid,
            surface_raw,
            row_axes,
            col_axes,
            ray_dir,
        )

        marker_mdilate = self._compute_dilation_batched(
            marker_uv,
            marker_depth,
            marker_contact,
            ray_row_vec,
            ray_col_vec,
        )
        mdilate = self._select_dilation_source_batched(
            marker_mdilate,
            marker_uv,
            dilation_source_depth,
            dilation_source_roi,
            dilation_source_active,
            ray_row_vec,
            ray_col_vec,
            grid_width=int(depth.shape[2]),
            grid_height=int(depth.shape[1]),
        )
        obj_all = self._pose_apply_batched_t(object_pose_wxyz[:batch_count], self._object_sample_points_l)
        roi_result = self._estimate_object_sample_sdf_batched(
            obj_all,
            depth,
            surface_points_w,
            surface_valid,
            object_points_w,
            object_valid,
            surface_raw,
            ray_dir,
            ray_row_vec,
            ray_col_vec,
        )
        selected_ids, roi_valid, history_valid, sdf, normals, sample_uv, slot_active, force_slot_active = roi_result
        obj = self._gather_roi_batched(obj_all, selected_ids, roi_valid, fill=0.0)
        frame = self._batched_surface_motion_frame(
            surface_points_w.reshape(batch_count, -1, 3),
            surface_valid.reshape(batch_count, -1),
            ray_row_vec,
            ray_col_vec,
            ray_dir,
        )
        origin, basis = frame
        obj_for_history = torch.matmul(obj - origin[:, None, :], basis)
        normals_for_history = torch.matmul(normals, basis)
        fbar_local = self._step_hydrosoft_forces_batched(
            sdf,
            obj_for_history,
            normals_for_history,
            slot_active,
            point_history_valid=history_valid,
            current_sample_ids=selected_ids,
            force_slot_active=force_slot_active,
        )
        fbar = torch.matmul(fbar_local, basis.transpose(1, 2))
        mshear = self._compute_shear_from_samples_batched(
            marker_uv,
            marker_normals,
            marker_valid,
            obj,
            sdf,
            fbar,
            normals,
            sample_uv,
            ray_row_vec,
            ray_col_vec,
            marker_points=marker_points,
            surface_frame_pose_wxyz=surface_frame_pose_wxyz,
        )

        displacement = (mdilate + mshear).to(dtype=torch.float32)
        return torch.where(marker_valid[..., None], displacement, torch.zeros_like(displacement))

    def _step_original_displacement_only_object_samples_batched(
        self,
        depth: torch.Tensor,
        surface_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        object_points_w: torch.Tensor,
        object_valid: torch.Tensor,
        surface_raw_m: torch.Tensor | None,
        surface_normals_w: torch.Tensor | None,
        object_pose_wxyz: torch.Tensor,
        ray_directions_w: torch.Tensor | None,
    ) -> torch.Tensor:
        """GPU-batched original HydroShear over every configured object sample."""

        if self._object_sample_points_l is None or int(self._object_sample_points_l.shape[0]) <= 0:
            return self._step_original_displacement_only_grid_batched(
                depth,
                surface_points_w,
                surface_valid,
                object_points_w,
                object_valid,
                surface_normals_w,
                ray_directions_w,
            )
        if depth.ndim != 3 or surface_points_w.ndim != 4 or object_points_w.ndim != 4:
            return self._empty_tensor(0, self.cfg.marker_rows * self.cfg.marker_cols, 3)

        depth = torch.clamp(self._finite_tensor(depth), min=0.0)
        surface_points_w = self._finite_tensor(surface_points_w)
        object_points_w = self._finite_tensor(object_points_w)
        surface_valid = surface_valid.to(device=self.device, dtype=torch.bool)
        object_valid = object_valid.to(device=self.device, dtype=torch.bool)
        surface_raw = self._finite_tensor(surface_raw_m) if surface_raw_m is not None else depth
        surface_normals = self._finite_tensor(surface_normals_w) if surface_normals_w is not None else None

        batch_count, height, width = depth.shape
        marker_uv = self._scaled_marker_uv_t(int(width), int(height))
        marker_count = int(marker_uv.shape[0])
        if batch_count <= 0 or marker_count <= 0:
            return self._empty_tensor(batch_count, marker_count, 3)

        xs = torch.clamp(torch.round(marker_uv[:, 0]).to(torch.long), 0, int(width) - 1)
        ys = torch.clamp(torch.round(marker_uv[:, 1]).to(torch.long), 0, int(height) - 1)
        marker_points = surface_points_w[:, ys, xs]
        marker_depth = depth[:, ys, xs]
        marker_valid = surface_valid[:, ys, xs] & torch.isfinite(marker_points).all(dim=-1)
        contact_threshold_m = float(self.cfg.contact_threshold_mm) * 1.0e-3
        marker_contact = marker_valid & (marker_depth > contact_threshold_m)
        mdilate = self._compute_original_dilation_batched(
            marker_points,
            marker_depth,
            marker_contact,
        )

        row_axes, col_axes = self._batched_marker_tangent_axes(surface_points_w, surface_valid, xs, ys)
        ray_dir = self._batched_ray_direction(
            surface_points_w,
            object_points_w,
            surface_valid,
            object_valid,
            depth,
            ray_directions_w=ray_directions_w,
        )
        ray_row_vec, ray_col_vec = self._batched_ray_grid_metric_vectors(
            surface_points_w,
            surface_valid,
            surface_raw,
            row_axes,
            col_axes,
            ray_dir,
        )
        obj = self._pose_apply_batched_t(object_pose_wxyz[:batch_count], self._object_sample_points_l)
        sdf, normals, slot_active = self._estimate_all_object_sample_sdf_batched(
            obj,
            depth,
            surface_points_w,
            surface_valid,
            object_points_w,
            object_valid,
            surface_raw,
            ray_dir,
            ray_row_vec,
            ray_col_vec,
        )

        origin, basis = self._batched_surface_motion_frame(
            surface_points_w.reshape(batch_count, -1, 3),
            surface_valid.reshape(batch_count, -1),
            ray_row_vec,
            ray_col_vec,
            ray_dir,
        )
        obj_for_history = torch.matmul(obj - origin[:, None, :], basis)
        normals_for_history = torch.matmul(normals, basis)
        fbar_local = self._step_hydrosoft_forces_batched(
            sdf,
            obj_for_history,
            normals_for_history,
            slot_active,
            force_slot_active=slot_active,
        )
        fbar = torch.matmul(fbar_local, basis.transpose(1, 2))
        mshear = self._compute_original_shear_batched(marker_points, marker_valid, obj, sdf, fbar, normals)
        displacement = (mdilate + mshear).to(dtype=torch.float32)
        return torch.where(marker_valid[..., None], displacement, torch.zeros_like(displacement))

    def _step_original_displacement_only_grid_batched(
        self,
        depth: torch.Tensor,
        surface_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        object_points_w: torch.Tensor,
        object_valid: torch.Tensor,
        surface_normals_w: torch.Tensor | None,
        ray_directions_w: torch.Tensor | None,
    ) -> torch.Tensor:
        """Original HydroShear fallback using the 32x24 TacMap hit grid as samples."""

        if depth.ndim != 3 or surface_points_w.ndim != 4 or object_points_w.ndim != 4:
            return self._empty_tensor(0, self.cfg.marker_rows * self.cfg.marker_cols, 3)
        depth = torch.clamp(self._finite_tensor(depth), min=0.0)
        surface_points_w = self._finite_tensor(surface_points_w)
        object_points_w = self._finite_tensor(object_points_w)
        surface_valid = surface_valid.to(device=self.device, dtype=torch.bool)
        object_valid = object_valid.to(device=self.device, dtype=torch.bool)

        batch_count, height, width = depth.shape
        marker_uv = self._scaled_marker_uv_t(int(width), int(height))
        xs = torch.clamp(torch.round(marker_uv[:, 0]).to(torch.long), 0, int(width) - 1)
        ys = torch.clamp(torch.round(marker_uv[:, 1]).to(torch.long), 0, int(height) - 1)
        marker_points = surface_points_w[:, ys, xs]
        marker_depth = depth[:, ys, xs]
        marker_valid = surface_valid[:, ys, xs] & torch.isfinite(marker_points).all(dim=-1)
        threshold_m = float(self.cfg.contact_threshold_mm) * 1.0e-3
        marker_contact = marker_valid & (marker_depth > threshold_m)
        mdilate = self._compute_original_dilation_batched(marker_points, marker_depth, marker_contact)

        flat_depth = depth.reshape(batch_count, -1)
        surface_flat = surface_points_w.reshape(batch_count, -1, 3)
        object_flat = object_points_w.reshape(batch_count, -1, 3)
        surface_valid_flat = surface_valid.reshape(batch_count, -1)
        object_valid_flat = object_valid.reshape(batch_count, -1)
        active = (
            surface_valid_flat
            & object_valid_flat
            & torch.isfinite(flat_depth)
            & (flat_depth > threshold_m)
            & torch.isfinite(surface_flat).all(dim=-1)
            & torch.isfinite(object_flat).all(dim=-1)
        )
        obj = torch.where(active[..., None], object_flat, surface_flat)
        sdf = torch.where(active, -flat_depth, torch.full_like(flat_depth, max(threshold_m, 1.0e-9)))
        ray_dir = self._batched_ray_direction(
            surface_points_w,
            object_points_w,
            surface_valid,
            object_valid,
            depth,
            ray_directions_w=ray_directions_w,
        )
        normals = ray_dir[:, None, :].expand_as(obj).contiguous()
        row_axes, col_axes = self._batched_marker_tangent_axes(surface_points_w, surface_valid, xs, ys)
        ray_row_vec, ray_col_vec = self._batched_grid_metric_vectors(
            surface_points_w,
            surface_valid,
            row_axes,
            col_axes,
        )
        origin, basis = self._batched_surface_motion_frame(
            surface_flat,
            surface_valid_flat,
            ray_row_vec,
            ray_col_vec,
            ray_dir,
        )
        slot_active = torch.any(active, dim=1)
        fbar_local = self._step_hydrosoft_forces_batched(
            sdf,
            torch.matmul(obj - origin[:, None, :], basis),
            torch.matmul(normals, basis),
            slot_active,
            force_slot_active=slot_active,
        )
        fbar = torch.matmul(fbar_local, basis.transpose(1, 2))
        mshear = self._compute_original_shear_batched(marker_points, marker_valid, obj, sdf, fbar, normals)
        displacement = (mdilate + mshear).to(dtype=torch.float32)
        return torch.where(marker_valid[..., None], displacement, torch.zeros_like(displacement))

    def _to_tensor(self, value, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=self.device, dtype=dtype)
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _finite_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(device=self.device, dtype=torch.float32)
        return torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))

    def _empty_tensor(self, *shape: int) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.float32, device=self.device)

    def _normalize_rows_batched(self, vectors: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        vectors = vectors.to(device=self.device, dtype=torch.float32)
        fallback = fallback.to(device=self.device, dtype=torch.float32)
        while fallback.ndim < vectors.ndim:
            fallback = fallback.unsqueeze(-2)
        fallback = fallback.expand_as(vectors)
        norms = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        valid = torch.isfinite(norms) & (norms > 1.0e-9) & torch.isfinite(vectors).all(dim=-1, keepdim=True)
        return torch.where(valid, vectors / torch.clamp(norms, min=1.0e-9), fallback).to(dtype=torch.float32)

    def _batched_marker_samples(
        self,
        depth: torch.Tensor,
        surface_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        surface_normals_w: torch.Tensor | None,
        marker_points_w: torch.Tensor | None,
        marker_normals_w: torch.Tensor | None,
        marker_valid_override: torch.Tensor | None,
        marker_depth_override: torch.Tensor | None = None,
        marker_depth_valid_override: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_count, height, width = depth.shape
        marker_uv = self._scaled_marker_uv_t(int(width), int(height))
        marker_count = int(marker_uv.shape[0])
        xs = torch.clamp(torch.round(marker_uv[:, 0]).to(torch.long), 0, int(width) - 1)
        ys = torch.clamp(torch.round(marker_uv[:, 1]).to(torch.long), 0, int(height) - 1)
        sampled_points = surface_points_w[:, ys, xs]
        marker_depth = depth[:, ys, xs]
        sampled_valid = surface_valid[:, ys, xs] & torch.isfinite(sampled_points).all(dim=-1)
        row_axes, col_axes = self._batched_marker_tangent_axes(surface_points_w, surface_valid, xs, ys)
        sampled_normals = self._batched_marker_normals(surface_normals_w, row_axes, col_axes, xs, ys)

        marker_depth_override_count = sum(
            value is not None for value in (marker_depth_override, marker_depth_valid_override)
        )
        if marker_depth_override_count not in (0, 2):
            raise ValueError("marker depth and validity overrides must be provided together")

        if marker_points_w is None and marker_normals_w is None and marker_valid_override is None:
            if marker_depth_override_count:
                raise ValueError("marker depth override requires calibrated marker point overrides")
            return (
                marker_uv,
                sampled_points,
                marker_depth,
                sampled_valid,
                row_axes,
                col_axes,
                sampled_normals,
            )
        if marker_points_w is None or marker_normals_w is None or marker_valid_override is None:
            raise ValueError("marker point, normal, and valid overrides must be provided together")

        expected_vec = (int(batch_count), marker_count, 3)
        expected_mask = (int(batch_count), marker_count)
        if tuple(marker_points_w.shape) != expected_vec:
            raise ValueError(f"Expected marker_points_w shape {expected_vec}, got {tuple(marker_points_w.shape)}")
        if tuple(marker_normals_w.shape) != expected_vec:
            raise ValueError(f"Expected marker_normals_w shape {expected_vec}, got {tuple(marker_normals_w.shape)}")
        if tuple(marker_valid_override.shape) != expected_mask:
            raise ValueError(f"Expected marker_valid shape {expected_mask}, got {tuple(marker_valid_override.shape)}")

        marker_points = marker_points_w.to(device=self.device, dtype=torch.float32)
        marker_normals = marker_normals_w.to(device=self.device, dtype=torch.float32)
        override_valid = (
            marker_valid_override.to(device=self.device, dtype=torch.bool)
            & torch.isfinite(marker_points).all(dim=-1)
            & torch.isfinite(marker_normals).all(dim=-1)
            & (torch.linalg.norm(marker_normals, dim=-1) > 1.0e-9)
        )
        marker_normals = self._normalize_rows_batched(marker_normals, sampled_normals)
        marker_normals = torch.where(
            (torch.sum(marker_normals * sampled_normals, dim=-1) < 0.0)[..., None],
            -marker_normals,
            marker_normals,
        )
        if marker_depth_override_count:
            if tuple(marker_depth_override.shape) != expected_mask:
                raise ValueError(
                    f"Expected marker_depth shape {expected_mask}, got {tuple(marker_depth_override.shape)}"
                )
            if tuple(marker_depth_valid_override.shape) != expected_mask:
                raise ValueError(
                    "Expected marker_depth_valid shape "
                    f"{expected_mask}, got {tuple(marker_depth_valid_override.shape)}"
                )
            independent_finite = torch.isfinite(marker_depth_override)
            marker_depth = torch.nan_to_num(
                marker_depth_override.to(device=self.device, dtype=torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min(0.0)
            marker_valid = (
                override_valid
                & marker_depth_valid_override.to(device=self.device, dtype=torch.bool)
                & independent_finite.to(device=self.device)
            )
        else:
            marker_depth = torch.where(sampled_valid, marker_depth, torch.zeros_like(marker_depth))
            marker_valid = override_valid
        return (
            marker_uv,
            marker_points,
            marker_depth,
            marker_valid,
            row_axes,
            col_axes,
            marker_normals,
        )

    def _batched_marker_tangent_axes(
        self,
        surface_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        xs: torch.Tensor,
        ys: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_count, height, width, _ = surface_points_w.shape
        x0 = torch.clamp(xs - 1, 0, int(width) - 1)
        x1 = torch.clamp(xs + 1, 0, int(width) - 1)
        y0 = torch.clamp(ys - 1, 0, int(height) - 1)
        y1 = torch.clamp(ys + 1, 0, int(height) - 1)

        col = surface_points_w[:, ys, x1] - surface_points_w[:, ys, x0]
        row = surface_points_w[:, y1, xs] - surface_points_w[:, y0, xs]
        col_valid = surface_valid[:, ys, x1] & surface_valid[:, ys, x0] & torch.isfinite(col).all(dim=-1)
        row_valid = surface_valid[:, y1, xs] & surface_valid[:, y0, xs] & torch.isfinite(row).all(dim=-1)
        fallback_col = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device).expand(
            batch_count, xs.shape[0], 3
        )
        fallback_row = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device).expand(
            batch_count, xs.shape[0], 3
        )
        col = torch.where(col_valid[..., None], col, fallback_col)
        col = self._normalize_rows_batched(col, fallback_col)
        row = torch.where(row_valid[..., None], row, fallback_row)
        row = row - col * torch.sum(row * col, dim=-1, keepdim=True)
        row = self._normalize_rows_batched(row, fallback_row)
        return row.to(dtype=torch.float32), col.to(dtype=torch.float32)

    def _batched_uniform_marker_projection_axes(
        self,
        surface_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        marker_row_axes_w: torch.Tensor,
        marker_col_axes_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one orthonormal UV projection frame per sensor slot."""

        batch_count, marker_count, _ = marker_row_axes_w.shape
        row_vec, col_vec = self._batched_grid_metric_vectors(
            surface_points_w,
            surface_valid,
            marker_row_axes_w,
            marker_col_axes_w,
        )

        fallback_col = self._normalize_rows_batched(
            torch.mean(marker_col_axes_w, dim=1),
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
        )
        col_axis = self._normalize_rows_batched(col_vec, fallback_col)

        basis_candidates = torch.eye(3, dtype=torch.float32, device=self.device).expand(batch_count, 3, 3)
        alignment = torch.abs(torch.sum(basis_candidates * col_axis[:, None, :], dim=-1))
        seed_ids = torch.argmin(alignment, dim=1)
        seed = basis_candidates[torch.arange(batch_count, device=self.device), seed_ids]
        seed = seed - col_axis * torch.sum(seed * col_axis, dim=-1, keepdim=True)
        seed = self._normalize_rows_batched(
            seed,
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device),
        )

        fallback_row = torch.mean(marker_row_axes_w, dim=1)
        fallback_row = fallback_row - col_axis * torch.sum(fallback_row * col_axis, dim=-1, keepdim=True)
        fallback_row = self._normalize_rows_batched(fallback_row, seed)
        row_axis = row_vec - col_axis * torch.sum(row_vec * col_axis, dim=-1, keepdim=True)
        row_axis = self._normalize_rows_batched(row_axis, fallback_row)

        row_axes = row_axis[:, None, :].expand(batch_count, marker_count, 3)
        col_axes = col_axis[:, None, :].expand(batch_count, marker_count, 3)
        return row_axes.to(dtype=torch.float32), col_axes.to(dtype=torch.float32)

    def _batched_marker_normals(
        self,
        surface_normals_w: torch.Tensor | None,
        row_axes_w: torch.Tensor,
        col_axes_w: torch.Tensor,
        xs: torch.Tensor,
        ys: torch.Tensor,
    ) -> torch.Tensor:
        fallback = torch.linalg.cross(col_axes_w, row_axes_w, dim=-1)
        fallback = self._normalize_rows_batched(
            fallback,
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
        )
        if surface_normals_w is None:
            normals = fallback
        else:
            samples = surface_normals_w[:, ys, xs].to(dtype=torch.float32)
            sample_valid = torch.isfinite(samples).all(dim=-1) & (torch.linalg.norm(samples, dim=-1) > 1.0e-9)
            samples = self._normalize_rows_batched(samples, fallback)
            normals = torch.where(sample_valid[..., None], samples, fallback)

        reference = self._normalize_rows_batched(
            torch.mean(fallback, dim=1),
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
        )
        orient = torch.sum(normals * reference[:, None, :], dim=-1)
        return torch.where((orient < 0.0)[..., None], -normals, normals).to(dtype=torch.float32)

    def _batched_grid_metric_vectors(
        self,
        surface_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        row_axes_w: torch.Tensor,
        col_axes_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fallback_row = self._normalize_rows_batched(
            torch.mean(row_axes_w, dim=1),
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device),
        )
        fallback_col = self._normalize_rows_batched(
            torch.mean(col_axes_w, dim=1),
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
        )
        points = surface_points_w.to(dtype=torch.float32)
        valid = surface_valid.bool() & torch.isfinite(points).all(dim=-1)
        col_deltas = points[:, :, 1:, :] - points[:, :, :-1, :]
        col_valid = valid[:, :, 1:] & valid[:, :, :-1] & torch.isfinite(col_deltas).all(dim=-1)
        row_deltas = points[:, 1:, :, :] - points[:, :-1, :, :]
        row_valid = valid[:, 1:, :] & valid[:, :-1, :] & torch.isfinite(row_deltas).all(dim=-1)

        col_count = torch.count_nonzero(col_valid, dim=(1, 2)).to(dtype=torch.float32)
        row_count = torch.count_nonzero(row_valid, dim=(1, 2)).to(dtype=torch.float32)
        col_sum = torch.where(col_valid[..., None], col_deltas, torch.zeros_like(col_deltas)).sum(dim=(1, 2))
        row_sum = torch.where(row_valid[..., None], row_deltas, torch.zeros_like(row_deltas)).sum(dim=(1, 2))
        col_vec = torch.where((col_count > 0.0)[:, None], col_sum / torch.clamp(col_count[:, None], min=1.0), fallback_col)
        row_vec = torch.where((row_count > 0.0)[:, None], row_sum / torch.clamp(row_count[:, None], min=1.0), fallback_row)
        return row_vec.to(dtype=torch.float32), col_vec.to(dtype=torch.float32)

    def _compute_dilation_batched(
        self,
        marker_uv: torch.Tensor,
        marker_depth: torch.Tensor,
        marker_contact: torch.Tensor,
        ray_row_vec: torch.Tensor,
        ray_col_vec: torch.Tensor,
    ) -> torch.Tensor:
        batch_count, marker_count = marker_depth.shape
        out = torch.zeros((batch_count, marker_count, 3), dtype=torch.float32, device=self.device)
        chunk_size = max(1, int(self.cfg.batch_slot_chunk_size))
        uv = marker_uv.to(dtype=torch.float32)
        target_u = uv[:, 0].reshape(1, marker_count, 1)
        target_v = uv[:, 1].reshape(1, marker_count, 1)
        source_u = uv[:, 0].reshape(1, 1, marker_count)
        source_v = uv[:, 1].reshape(1, 1, marker_count)
        du = target_u - source_u
        dv = target_v - source_v

        source_depth = torch.where(marker_contact, marker_depth, torch.zeros_like(marker_depth))
        for start in range(0, batch_count, chunk_size):
            end = min(start + chunk_size, batch_count)
            dvec = (
                du[..., None] * ray_col_vec[start:end, None, None, :]
                + dv[..., None] * ray_row_vec[start:end, None, None, :]
            )
            dist2 = torch.sum(dvec * dvec, dim=-1)
            weights = torch.exp(-float(self.cfg.lambda_dilate) * dist2).to(dtype=torch.float32)
            out[start:end] = float(self.cfg.dilate_scale) * torch.sum(
                source_depth[start:end, None, :, None] * dvec * weights[..., None],
                dim=2,
            )
        return out.to(dtype=torch.float32)

    def _constant_stride_lattice_from_normalized(
        self,
        coord_low: torch.Tensor,
        coord_high: torch.Tensor,
        *,
        reference_size: int,
        sample_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct the adaptive TacMap integer lattice from normalized ROI bounds."""

        if reference_size <= 0 or sample_count <= 0 or sample_count > reference_size:
            raise ValueError("Adaptive dilation lattice sizes must satisfy reference >= samples > 0")
        low_float = torch.clamp(coord_low, 0.0, 1.0) * float(reference_size - 1)
        high_float = torch.clamp(coord_high, 0.0, 1.0) * float(reference_size - 1)
        low = torch.clamp(torch.floor(low_float).to(dtype=torch.long), min=0, max=reference_size - 1)
        high = torch.clamp(torch.ceil(high_float).to(dtype=torch.long), min=0, max=reference_size - 1)
        if sample_count == 1:
            center = torch.round(0.5 * (low + high).to(dtype=torch.float32)).to(dtype=torch.long)
            return center[:, None], torch.ones_like(center)

        max_stride = max(1, (reference_size - 1) // (sample_count - 1))
        required_span = torch.clamp(high - low, min=0)
        stride = torch.div(
            required_span + (sample_count - 2),
            sample_count - 1,
            rounding_mode="floor",
        )
        stride = torch.clamp(stride, min=1, max=max_stride)
        lattice_span = stride * (sample_count - 1)
        center = 0.5 * (low + high).to(dtype=torch.float32)
        start = torch.round(center - 0.5 * lattice_span.to(dtype=torch.float32)).to(dtype=torch.long)
        start = torch.maximum(start, torch.zeros_like(start))
        start = torch.minimum(start, (reference_size - 1) - lattice_span)
        offsets = torch.arange(sample_count, device=coord_low.device, dtype=torch.long).reshape(1, -1)
        return start[:, None] + offsets * stride[:, None], stride

    def _compute_adaptive_roi_dilation_batched(
        self,
        marker_uv: torch.Tensor,
        source_depth: torch.Tensor,
        source_roi_norm: torch.Tensor,
        source_active: torch.Tensor,
        ray_row_vec: torch.Tensor,
        ray_col_vec: torch.Tensor,
        *,
        grid_width: int,
        grid_height: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Spread a dense adaptive TacMap depth patch onto the fixed real-marker targets."""

        if source_depth.ndim != 3:
            raise ValueError(f"Expected adaptive dilation depth [B,H,W], got {tuple(source_depth.shape)}")
        batch_count, source_rows, source_cols = source_depth.shape
        marker_count = int(marker_uv.shape[0])
        expected_roi = (int(batch_count), 4)
        if tuple(source_roi_norm.shape) != expected_roi:
            raise ValueError(
                f"Expected adaptive dilation ROI shape {expected_roi}, got {tuple(source_roi_norm.shape)}"
            )
        if tuple(source_active.shape) != (int(batch_count),):
            raise ValueError(
                f"Expected adaptive dilation active shape {(int(batch_count),)}, got {tuple(source_active.shape)}"
            )
        if int(ray_row_vec.shape[0]) != batch_count or int(ray_col_vec.shape[0]) != batch_count:
            raise ValueError("Adaptive dilation source batch must match the ray metric batch")

        reference_width = max(1, int(self.cfg.width))
        reference_height = max(1, int(self.cfg.height))
        roi = torch.nan_to_num(
            source_roi_norm.to(device=self.device, dtype=torch.float32),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        source_depth = torch.nan_to_num(
            source_depth.to(device=self.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        source_active = source_active.to(device=self.device, dtype=torch.bool).reshape(batch_count)

        source_cols_ref, col_stride_ref = self._constant_stride_lattice_from_normalized(
            roi[:, 0],
            roi[:, 1],
            reference_size=reference_width,
            sample_count=int(source_cols),
        )
        source_rows_ref, row_stride_ref = self._constant_stride_lattice_from_normalized(
            roi[:, 2],
            roi[:, 3],
            reference_size=reference_height,
            sample_count=int(source_rows),
        )
        # The existing dilation code measures UV offsets in coarse TacMap pixel
        # steps.  Map the adaptive reference-pixel lattice into that same frame.
        col_scale = float(grid_width) / float(reference_width)
        row_scale = float(grid_height) / float(reference_height)
        source_u = source_cols_ref.to(dtype=torch.float32) * col_scale
        source_v = source_rows_ref.to(dtype=torch.float32) * row_scale
        source_u = source_u[:, None, :].expand(batch_count, source_rows, source_cols).reshape(batch_count, -1)
        source_v = source_v[:, :, None].expand(batch_count, source_rows, source_cols).reshape(batch_count, -1)
        source_depth_flat = source_depth.reshape(batch_count, -1)

        contact_threshold_m = float(self.cfg.contact_threshold_mm) * 1.0e-3
        source_contact = source_active[:, None] & (source_depth_flat > contact_threshold_m)
        use_source = source_active & torch.any(source_contact, dim=1)
        active_batch_ids = torch.nonzero(use_source, as_tuple=False).flatten()

        target_uv = marker_uv.to(device=self.device, dtype=torch.float32)
        target_u = target_uv[:, 0].reshape(1, marker_count, 1)
        target_v = target_uv[:, 1].reshape(1, marker_count, 1)
        out = torch.zeros((batch_count, marker_count, 3), dtype=torch.float32, device=self.device)
        if active_batch_ids.numel() == 0:
            return out, use_source

        batch_chunk = max(1, min(int(self.cfg.batch_slot_chunk_size), 16))
        source_chunk = max(1, min(int(self.cfg.object_chunk_size), 256))
        diagonal = torch.eye(marker_count, dtype=torch.bool, device=self.device).reshape(1, marker_count, marker_count)
        marker_du = target_uv[:, 0].reshape(1, marker_count, 1) - target_uv[:, 0].reshape(1, 1, marker_count)
        marker_dv = target_uv[:, 1].reshape(1, marker_count, 1) - target_uv[:, 1].reshape(1, 1, marker_count)

        for b_start in range(0, int(active_batch_ids.numel()), batch_chunk):
            batch_ids = active_batch_ids[b_start : b_start + batch_chunk]
            row_metric = ray_row_vec[batch_ids]
            col_metric = ray_col_vec[batch_ids]
            col_norm2 = torch.sum(col_metric * col_metric, dim=-1)[:, None, None]
            row_norm2 = torch.sum(row_metric * row_metric, dim=-1)[:, None, None]
            row_col_dot = torch.sum(row_metric * col_metric, dim=-1)[:, None, None]

            marker_dist2 = (
                marker_du * marker_du * col_norm2
                + marker_dv * marker_dv * row_norm2
                + 2.0 * marker_du * marker_dv * row_col_dot
            )
            marker_dist2 = torch.where(
                diagonal | (marker_dist2 <= 1.0e-12),
                torch.full_like(marker_dist2, torch.inf),
                marker_dist2,
            )
            marker_reference_area = torch.median(torch.amin(marker_dist2, dim=2), dim=1).values
            grid_cell_area = torch.linalg.norm(
                torch.linalg.cross(col_metric, row_metric, dim=-1),
                dim=-1,
            )
            marker_reference_area = torch.where(
                torch.isfinite(marker_reference_area) & (marker_reference_area > 1.0e-12),
                marker_reference_area,
                torch.clamp(grid_cell_area, min=1.0e-12),
            )

            # Treat the sum as a surface quadrature: one dense source represents
            # its lattice cell, relative to one cell in the original marker grid.
            # This prevents 1000 sources from increasing the response merely
            # because there are about ten times as many samples as markers.
            col_step = col_stride_ref[batch_ids].to(dtype=torch.float32) * col_scale
            row_step = row_stride_ref[batch_ids].to(dtype=torch.float32) * row_scale
            cell_col = col_metric * col_step[:, None]
            cell_row = row_metric * row_step[:, None]
            cell_area = torch.linalg.norm(torch.linalg.cross(cell_col, cell_row, dim=-1), dim=-1)
            area_weight = torch.nan_to_num(
                cell_area / marker_reference_area,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            accum = torch.zeros((batch_ids.numel(), marker_count, 3), dtype=torch.float32, device=self.device)
            point_count = int(source_depth_flat.shape[1])
            for p_start in range(0, point_count, source_chunk):
                p_end = min(p_start + source_chunk, point_count)
                du = target_u - source_u[batch_ids, None, p_start:p_end]
                dv = target_v - source_v[batch_ids, None, p_start:p_end]
                dist2 = (
                    du * du * col_norm2
                    + dv * dv * row_norm2
                    + 2.0 * du * dv * row_col_dot
                )
                weights = torch.exp(-float(self.cfg.lambda_dilate) * dist2).to(dtype=torch.float32)
                weights = torch.where(
                    source_contact[batch_ids, None, p_start:p_end],
                    weights,
                    torch.zeros_like(weights),
                )
                amplitude = source_depth_flat[batch_ids, p_start:p_end] * area_weight[:, None]
                weighted_depth = amplitude[:, None, :] * weights
                sum_u = torch.sum(weighted_depth * du, dim=2)
                sum_v = torch.sum(weighted_depth * dv, dim=2)
                accum = accum + (
                    sum_u[..., None] * col_metric[:, None, :]
                    + sum_v[..., None] * row_metric[:, None, :]
                )
            out[batch_ids] = float(self.cfg.dilate_scale) * accum

        return out.to(dtype=torch.float32), use_source

    def _select_dilation_source_batched(
        self,
        marker_mdilate: torch.Tensor,
        marker_uv: torch.Tensor,
        source_depth: torch.Tensor | None,
        source_roi_norm: torch.Tensor | None,
        source_active: torch.Tensor | None,
        ray_row_vec: torch.Tensor,
        ray_col_vec: torch.Tensor,
        *,
        grid_width: int,
        grid_height: int,
    ) -> torch.Tensor:
        source_count = sum(value is not None for value in (source_depth, source_roi_norm, source_active))
        if source_count == 0:
            return marker_mdilate
        if source_count != 3:
            raise ValueError("Adaptive dilation depth, ROI, and active mask must be provided together")
        assert source_depth is not None
        assert source_roi_norm is not None
        assert source_active is not None
        adaptive_mdilate, use_source = self._compute_adaptive_roi_dilation_batched(
            marker_uv,
            source_depth,
            source_roi_norm,
            source_active,
            ray_row_vec,
            ray_col_vec,
            grid_width=grid_width,
            grid_height=grid_height,
        )
        return torch.where(use_source[:, None, None], adaptive_mdilate, marker_mdilate).to(dtype=torch.float32)

    def _compute_original_dilation_batched(
        self,
        marker_points: torch.Tensor,
        marker_depth: torch.Tensor,
        marker_contact: torch.Tensor,
    ) -> torch.Tensor:
        """Original HydroShear dilation with Euclidean marker-to-marker distance."""

        batch_count, marker_count, _ = marker_points.shape
        out = torch.zeros((batch_count, marker_count, 3), dtype=torch.float32, device=self.device)
        source_depth = torch.where(marker_contact, marker_depth, torch.zeros_like(marker_depth))
        batch_chunk = max(1, int(self.cfg.batch_slot_chunk_size))
        for start in range(0, batch_count, batch_chunk):
            end = min(start + batch_chunk, batch_count)
            points = marker_points[start:end]
            dvec = points[:, :, None, :] - points[:, None, :, :]
            dist2 = torch.sum(dvec * dvec, dim=-1)
            weights = torch.exp(-float(self.cfg.lambda_dilate) * dist2).to(dtype=torch.float32)
            out[start:end] = float(self.cfg.dilate_scale) * torch.sum(
                source_depth[start:end, None, :, None] * dvec * weights[..., None],
                dim=2,
            )
        return out.to(dtype=torch.float32)

    def _select_object_sample_roi_batched(
        self,
        roi_candidate: torch.Tensor,
        roi_priority: torch.Tensor,
        sample_contact: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_count, point_count = roi_candidate.shape
        roi_count = max(1, min(int(self.cfg.object_sample_roi_count), int(point_count)))
        shape_ids = (int(batch_count), int(roi_count))

        roi_candidate = roi_candidate.to(device=self.device, dtype=torch.bool)
        sample_contact = sample_contact.to(device=self.device, dtype=torch.bool) & roi_candidate
        finite_priority = roi_candidate & torch.isfinite(roi_priority)
        candidate_priority = torch.where(
            finite_priority,
            torch.clamp(roi_priority.to(device=self.device, dtype=torch.float32), min=0.0),
            torch.zeros_like(roi_priority, device=self.device, dtype=torch.float32),
        )
        priority_max = torch.amax(candidate_priority, dim=1, keepdim=True)
        replace_margin = max(0.0, float(self.cfg.object_sample_roi_replace_margin_m))
        class_gap = priority_max + max(replace_margin, 1.0e-9)

        prev_ids = self._batch_prev_sample_ids
        if (
            prev_ids is None
            or tuple(prev_ids.shape) != shape_ids
            or prev_ids.device != self.device
        ):
            prev_ids = torch.full(shape_ids, -1, dtype=torch.long, device=self.device)

        prev_valid = prev_ids >= 0
        prev_clamped = torch.clamp(prev_ids, 0, max(0, int(point_count) - 1))
        prev_still_roi = prev_valid & roi_candidate.gather(1, prev_clamped)
        prev_contact = prev_still_roi & sample_contact.gather(1, prev_clamped)
        prev_priority = candidate_priority.gather(1, prev_clamped)

        exclude = torch.zeros((int(batch_count), int(point_count)), dtype=torch.int8, device=self.device)
        exclude.scatter_reduce_(
            1,
            prev_clamped,
            prev_valid.to(dtype=torch.int8),
            reduce="amax",
            include_self=True,
        )
        new_candidate = roi_candidate & ~exclude.bool()
        neg_inf = torch.full_like(roi_priority, -torch.inf)
        new_rank = candidate_priority + sample_contact.to(dtype=torch.float32) * class_gap
        new_rank = torch.where(new_candidate, new_rank, neg_inf)
        new_values, new_ids = torch.topk(new_rank, k=roi_count, dim=1)
        new_valid = torch.isfinite(new_values)
        new_contact = sample_contact.gather(1, new_ids)
        new_priority = candidate_priority.gather(1, new_ids)

        # Empty slots are weakest, followed by pre-contact and then contact slots.
        prev_strength = prev_priority + prev_contact.to(dtype=torch.float32) * class_gap
        prev_strength = torch.where(prev_still_roi, prev_strength, torch.full_like(prev_strength, -torch.inf))
        weak_slot_order = torch.argsort(prev_strength, dim=1, descending=False)
        slot_still_roi = prev_still_roi.gather(1, weak_slot_order)
        slot_contact = prev_contact.gather(1, weak_slot_order)
        slot_priority = prev_priority.gather(1, weak_slot_order)

        same_class = new_contact == slot_contact
        replace = new_valid & (
            ~slot_still_roi
            | (new_contact & ~slot_contact)
            | (same_class & (new_priority > slot_priority + replace_margin))
        )

        selected_ids = torch.where(prev_still_roi, prev_ids, -torch.ones_like(prev_ids))
        slot_ids = selected_ids.gather(1, weak_slot_order)
        slot_ids = torch.where(replace, new_ids, slot_ids)
        selected_ids = selected_ids.scatter(1, weak_slot_order, slot_ids)
        selected_valid = selected_ids >= 0
        history_valid = selected_valid & prev_still_roi & (selected_ids == prev_ids)
        return selected_ids.to(dtype=torch.long), selected_valid, history_valid

    def _gather_roi_batched(
        self,
        values: torch.Tensor,
        selected_ids: torch.Tensor,
        selected_valid: torch.Tensor,
        *,
        fill: float = 0.0,
    ) -> torch.Tensor:
        ids = torch.clamp(selected_ids, 0, max(0, int(values.shape[1]) - 1))
        if values.ndim == 2:
            out = values.gather(1, ids)
            return torch.where(selected_valid, out, torch.full_like(out, float(fill)))
        ids_expanded = ids[..., None].expand(-1, -1, int(values.shape[-1]))
        out = values.gather(1, ids_expanded)
        return torch.where(selected_valid[..., None], out, torch.full_like(out, float(fill)))

    def _ensure_batch_prev_buffers(self, batch_count: int, point_count: int) -> None:
        shape_sdf = (int(batch_count), int(point_count))
        shape_points = (int(batch_count), int(point_count), 3)
        if (
            self._batch_prev_sdf is None
            or self._batch_prev_indenter_points is None
            or self._batch_prev_normals is None
            or self._batch_hydrosoft_forces is None
            or self._batch_state_valid is None
            or self._batch_prev_sample_ids is None
            or tuple(self._batch_prev_sdf.shape) != shape_sdf
            or tuple(self._batch_prev_indenter_points.shape) != shape_points
            or tuple(self._batch_state_valid.shape) != shape_sdf
            or self._batch_prev_sdf.device != self.device
        ):
            self._batch_prev_sdf = torch.zeros(shape_sdf, dtype=torch.float32, device=self.device)
            self._batch_prev_indenter_points = torch.zeros(shape_points, dtype=torch.float32, device=self.device)
            self._batch_prev_normals = torch.zeros(shape_points, dtype=torch.float32, device=self.device)
            self._batch_hydrosoft_forces = torch.zeros(shape_points, dtype=torch.float32, device=self.device)
            self._batch_state_valid = torch.zeros(shape_sdf, dtype=torch.bool, device=self.device)
            self._batch_prev_sample_ids = torch.full(shape_sdf, -1, dtype=torch.long, device=self.device)

    def _step_hydrosoft_forces_batched(
        self,
        sdf: torch.Tensor,
        indenter_points: torch.Tensor,
        normals: torch.Tensor,
        slot_active: torch.Tensor,
        *,
        point_history_valid: torch.Tensor | None = None,
        current_sample_ids: torch.Tensor | None = None,
        force_slot_active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_count, point_count = sdf.shape
        self._ensure_batch_prev_buffers(batch_count, point_count)
        assert self._batch_prev_sdf is not None
        assert self._batch_prev_indenter_points is not None
        assert self._batch_prev_normals is not None
        assert self._batch_hydrosoft_forces is not None
        assert self._batch_state_valid is not None
        assert self._batch_prev_sample_ids is not None

        sdf = self._finite_tensor(sdf).reshape(batch_count, point_count)
        indenter_points = self._finite_tensor(indenter_points).reshape(batch_count, point_count, 3)
        normals = self._normalize_rows_batched(
            normals,
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
        )
        current_valid = torch.ones((batch_count, point_count), dtype=torch.bool, device=self.device)
        if current_sample_ids is not None:
            current_valid = current_sample_ids.to(device=self.device, dtype=torch.long).reshape(batch_count, point_count) >= 0
        state_valid = self._batch_state_valid & slot_active.bool()[:, None] & current_valid
        if point_history_valid is not None:
            state_valid = state_valid & point_history_valid.to(device=self.device, dtype=torch.bool).reshape(batch_count, point_count)
        prev_sdf = torch.where(state_valid, self._batch_prev_sdf, sdf)
        prev_points = torch.where(state_valid[..., None], self._batch_prev_indenter_points, indenter_points)
        prev_normals = torch.where(state_valid[..., None], self._batch_prev_normals, normals)
        hydrosoft_forces = torch.where(
            state_valid[..., None],
            self._batch_hydrosoft_forces,
            torch.zeros_like(indenter_points),
        )

        prev_penetration = torch.relu(-prev_sdf)
        penetration = torch.relu(-sdf)
        denom = prev_sdf - sdf
        alpha_d = -((prev_penetration - penetration) / denom)
        alpha_fallback = -0.5 * (torch.sign(sdf) - 1.0)
        alpha_d = torch.where(torch.isfinite(alpha_d), alpha_d, alpha_fallback)
        displacement = alpha_d[..., None] * (prev_points - indenter_points)

        prev_normals = self._normalize_rows_batched(
            prev_normals,
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
        )
        prev_force_n = torch.sum(hydrosoft_forces * prev_normals, dim=-1)
        prev_force_t = hydrosoft_forces - prev_force_n[..., None] * prev_normals
        prev_force_t = prev_force_t - torch.sum(prev_force_t * normals, dim=-1, keepdim=True) * normals
        displacement_n = torch.sum(displacement * normals, dim=-1)
        displacement_t = displacement - displacement_n[..., None] * normals

        sample_area = float(self.cfg.hydrosoft_area) * float(self._object_sample_area_scale)
        fn = prev_force_n + float(self.cfg.hydrosoft_e) * sample_area * displacement_n
        ft = prev_force_t + float(self.cfg.hydrosoft_k) * sample_area * displacement_t
        fn_bar = torch.relu(fn)
        norm_ft = torch.linalg.norm(ft, dim=-1)
        ft_limit = torch.minimum(float(self.cfg.hydrosoft_mu) * fn_bar, norm_ft)
        ft_bar = ft_limit[..., None] * ft / (norm_ft[..., None] + 1.0e-8)
        fbar = ft_bar + fn_bar[..., None] * normals
        force_active = slot_active if force_slot_active is None else force_slot_active.to(device=self.device, dtype=torch.bool)
        fbar = torch.where(((sdf < 0.0) & force_active[:, None])[..., None], fbar, torch.zeros_like(fbar))
        fbar = fbar.to(dtype=torch.float32)

        self._batch_hydrosoft_forces = fbar.detach().clone()
        self._batch_prev_indenter_points = indenter_points.detach().clone()
        self._batch_prev_sdf = sdf.detach().clone()
        self._batch_prev_normals = normals.detach().clone()
        self._batch_state_valid = (slot_active[:, None].to(dtype=torch.bool) & current_valid).detach().clone()
        if current_sample_ids is not None:
            self._batch_prev_sample_ids = current_sample_ids.detach().clone().to(device=self.device, dtype=torch.long)
        return fbar

    def _compute_shear_from_grid_batched(
        self,
        marker_uv: torch.Tensor,
        marker_points: torch.Tensor,
        marker_normals: torch.Tensor,
        marker_valid: torch.Tensor,
        depth_m: torch.Tensor,
        surface_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        object_points_w: torch.Tensor,
        object_valid: torch.Tensor,
        ray_row_vec: torch.Tensor,
        ray_col_vec: torch.Tensor,
        surface_frame_pose_wxyz: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_count, height, width = depth_m.shape
        marker_count = marker_uv.shape[0]
        point_count = int(height) * int(width)
        flat_depth = depth_m.reshape(batch_count, point_count)
        surface_flat = surface_points_w.reshape(batch_count, point_count, 3)
        object_flat = object_points_w.reshape(batch_count, point_count, 3)
        flat_surface_valid = surface_valid.reshape(batch_count, point_count).bool()
        flat_object_valid = object_valid.reshape(batch_count, point_count).bool()
        finite_surface = torch.isfinite(surface_flat).all(dim=-1)
        finite_object = torch.isfinite(object_flat).all(dim=-1)
        contact_threshold_m = float(self.cfg.contact_threshold_mm) * 1.0e-3
        active = (
            flat_surface_valid
            & flat_object_valid
            & finite_surface
            & finite_object
            & torch.isfinite(flat_depth)
            & (flat_depth > contact_threshold_m)
        )
        slot_active = torch.any(active, dim=1) & torch.any(marker_valid, dim=1)
        fallback_points = torch.where(finite_surface[..., None], surface_flat, torch.zeros_like(surface_flat))
        obj = torch.where((flat_object_valid & finite_object)[..., None], object_flat, fallback_points)
        sdf = torch.full_like(flat_depth, max(contact_threshold_m, 1.0e-9))
        sdf = torch.where(active, -flat_depth, sdf)

        fallback_normal = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        normal_vectors = surface_flat - obj
        normal_vectors = self._normalize_rows_batched(normal_vectors, fallback_normal)
        active_count = torch.count_nonzero(active, dim=1).to(dtype=torch.float32)
        normal_sum = torch.where(active[..., None], normal_vectors, torch.zeros_like(normal_vectors)).sum(dim=1)
        ray_dir = self._normalize_rows_batched(normal_sum, fallback_normal)
        ray_dir = torch.where((active_count > 0.0)[:, None], ray_dir, fallback_normal.reshape(1, 3))
        normals = ray_dir[:, None, :].expand(batch_count, point_count, 3).contiguous()

        frame = self._batched_surface_motion_frame(surface_flat, flat_surface_valid & finite_surface, ray_row_vec, ray_col_vec, ray_dir)
        origin, basis = frame
        obj_for_history = torch.matmul(obj - origin[:, None, :], basis)
        normals_for_history = torch.matmul(normals, basis)
        fbar_local = self._step_hydrosoft_forces_batched(sdf, obj_for_history, normals_for_history, slot_active)
        fbar = torch.matmul(fbar_local, basis.transpose(1, 2))
        normal_force = torch.sum(fbar * normals, dim=-1)
        h = torch.clamp(normal_force, min=0.0).to(dtype=torch.float32)
        contact = (sdf < 0.0) & torch.isfinite(sdf) & torch.isfinite(fbar).all(dim=-1)
        h = torch.where(contact, h, torch.zeros_like(h))
        fbar_tangent = fbar - normal_force[..., None] * normals

        basis = torch.stack((ray_col_vec, ray_row_vec), dim=-1).to(dtype=torch.float32)
        basis_ok = torch.isfinite(basis).all(dim=(1, 2)) & (
            torch.linalg.norm(torch.linalg.cross(ray_col_vec, ray_row_vec, dim=-1), dim=-1) > 1.0e-12
        )
        pinv = torch.linalg.pinv(basis)
        uv_delta = torch.einsum("bnc,bdc->bnd", fbar_tangent, pinv)
        uv_delta = torch.where(basis_ok[:, None, None], uv_delta, torch.zeros_like(uv_delta))

        flat_ids = torch.arange(point_count, dtype=torch.float32, device=self.device)
        sample_uv = torch.stack((torch.remainder(flat_ids, float(width)), torch.floor(flat_ids / float(width))), dim=-1)
        affected_uv = sample_uv[None, :, :] + uv_delta
        use_curved_distance = (
            surface_frame_pose_wxyz is not None
            and self._shear_reference_starts_l is not None
        )
        out = torch.zeros((batch_count, marker_count, 3), dtype=torch.float32, device=self.device)
        chunk_size = max(1, int(self.cfg.batch_slot_chunk_size))
        target_uv = marker_uv.to(dtype=torch.float32)
        target_u = target_uv[:, 0].reshape(1, marker_count, 1)
        target_v = target_uv[:, 1].reshape(1, marker_count, 1)

        for start in range(0, batch_count, chunk_size):
            end = min(start + chunk_size, batch_count)
            if use_curved_distance:
                curved_source_points, curved_source_valid = self._curved_surface_points_from_uv_batched(
                    affected_uv[start:end],
                    surface_frame_pose_wxyz[start:end],
                    batch_offset=start,
                )
                assert curved_source_points is not None
                assert curved_source_valid is not None
                dvec = (
                    marker_points[start:end, :, None, :]
                    - curved_source_points[:, None, :, :]
                )
            else:
                source_u = affected_uv[start:end, :, 0].reshape(end - start, 1, point_count)
                source_v = affected_uv[start:end, :, 1].reshape(end - start, 1, point_count)
                du = target_u - source_u
                dv = target_v - source_v
                dvec = (
                    du[..., None] * ray_col_vec[start:end, None, None, :]
                    + dv[..., None] * ray_row_vec[start:end, None, None, :]
                )
            dist2 = torch.sum(dvec * dvec, dim=-1)
            weights = torch.exp(-float(self.cfg.lambda_shear) * dist2).to(dtype=torch.float32)
            if use_curved_distance:
                assert curved_source_valid is not None
                weights = torch.where(
                    curved_source_valid[:, None, :],
                    weights,
                    torch.zeros_like(weights),
                )
            normals_chunk = marker_normals[start:end, :, None, :]
            tangent_chunk = fbar_tangent[start:end, None, :, :]
            normal_component = torch.sum(tangent_chunk * normals_chunk, dim=-1, keepdim=True) * normals_chunk
            transported = tangent_chunk - normal_component
            contribution = torch.sum(
                h[start:end, None, :, None] * -transported * weights[..., None],
                dim=2,
            )
            out[start:end] = float(self.cfg.shear_scale) * contribution

        return torch.where(marker_valid[..., None], out, torch.zeros_like(out)).to(dtype=torch.float32)

    def _object_pose_batch_t(self, pose: np.ndarray | torch.Tensor, sensor_count: int) -> torch.Tensor:
        pose_t = self._to_tensor(pose)
        if pose_t.ndim == 1:
            pose_t = pose_t.reshape(1, -1).expand(int(sensor_count), -1)
        else:
            pose_t = pose_t.reshape(-1, pose_t.shape[-1])
        out = torch.zeros((int(sensor_count), 7), dtype=torch.float32, device=self.device)
        out[:, 3] = 1.0
        count = min(int(sensor_count), int(pose_t.shape[0]))
        if count <= 0 or pose_t.shape[-1] < 7:
            return out
        candidate = pose_t[:count, :7].to(dtype=torch.float32)
        finite = torch.isfinite(candidate).all(dim=-1)
        quat = candidate[:, 3:7]
        quat_norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
        valid = finite & (quat_norm.reshape(-1) > 1.0e-8)
        candidate = torch.where(valid[:, None], candidate, out[:count])
        candidate[:, 3:7] = torch.where(
            valid[:, None],
            quat / torch.clamp(quat_norm, min=1.0e-8),
            out[:count, 3:7],
        )
        out[:count] = candidate
        return out

    def _pose_apply_batched_t(self, pose: torch.Tensor, points_l: torch.Tensor) -> torch.Tensor:
        pose = pose.to(device=self.device, dtype=torch.float32).reshape(-1, 7)
        points_l = points_l.to(device=self.device, dtype=torch.float32).reshape(-1, 3)
        vectors = points_l.unsqueeze(0).expand(pose.shape[0], -1, -1)
        quat = pose[:, 3:7]
        qvec = quat[:, None, 1:4].expand_as(vectors)
        uv = torch.linalg.cross(qvec, vectors, dim=-1)
        uuv = torch.linalg.cross(qvec, uv, dim=-1)
        return (vectors + 2.0 * (quat[:, None, 0:1] * uv + uuv) + pose[:, None, :3]).to(dtype=torch.float32)

    def _pose_apply_per_batch_t(self, pose: torch.Tensor, points_l: torch.Tensor) -> torch.Tensor:
        """Apply one WXYZ pose to the corresponding variable point set in each batch slot."""

        pose = pose.to(device=self.device, dtype=torch.float32).reshape(-1, 7)
        points_l = points_l.to(device=self.device, dtype=torch.float32)
        if points_l.ndim != 3 or int(points_l.shape[0]) != int(pose.shape[0]) or points_l.shape[-1] != 3:
            raise ValueError("Per-batch pose application expects pose [B,7] and points [B,N,3]")
        quat = pose[:, 3:7]
        qvec = quat[:, None, 1:4].expand_as(points_l)
        uv = torch.linalg.cross(qvec, points_l, dim=-1)
        uuv = torch.linalg.cross(qvec, uv, dim=-1)
        return (
            points_l
            + 2.0 * (quat[:, None, 0:1] * uv + uuv)
            + pose[:, None, :3]
        ).to(dtype=torch.float32)

    def _batched_ray_direction(
        self,
        surface_points_w: torch.Tensor,
        object_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        object_valid: torch.Tensor,
        depth_m: torch.Tensor,
        *,
        ray_directions_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_count = int(depth_m.shape[0])
        contact_threshold_m = float(self.cfg.contact_threshold_mm) * 1.0e-3
        ray_vec = surface_points_w - object_points_w
        active = (
            surface_valid.bool()
            & object_valid.bool()
            & torch.isfinite(ray_vec).all(dim=-1)
            & torch.isfinite(depth_m)
            & (depth_m > contact_threshold_m)
        )
        fallback = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device).expand(batch_count, 3)
        normalized = self._normalize_rows_batched(ray_vec, fallback[:, None, :])
        count = torch.count_nonzero(active, dim=(1, 2)).to(dtype=torch.float32)
        ray_sum = torch.where(active[..., None], normalized, torch.zeros_like(normalized)).sum(dim=(1, 2))
        ray_dir = self._normalize_rows_batched(ray_sum, fallback)
        contact_direction = torch.where((count > 0.0)[:, None], ray_dir, fallback)
        if ray_directions_w is None:
            return contact_direction.to(dtype=torch.float32)

        configured = ray_directions_w.to(device=self.device, dtype=torch.float32).reshape(batch_count, -1, 3)
        configured_norm = torch.linalg.norm(configured, dim=-1, keepdim=True)
        configured_valid = torch.isfinite(configured).all(dim=-1, keepdim=True) & (configured_norm > 1.0e-9)
        configured_unit = torch.where(
            configured_valid,
            configured / torch.clamp(configured_norm, min=1.0e-9),
            torch.zeros_like(configured),
        )
        configured_count = torch.count_nonzero(configured_valid.squeeze(-1), dim=1).to(dtype=torch.float32)
        configured_mean = configured_unit.sum(dim=1) / torch.clamp(configured_count[:, None], min=1.0)
        configured_mean = self._normalize_rows_batched(configured_mean, contact_direction)
        return torch.where(
            (configured_count > 0.0)[:, None],
            configured_mean,
            contact_direction,
        ).to(dtype=torch.float32)

    def _batched_ray_grid_metric_vectors(
        self,
        surface_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        surface_raw_m: torch.Tensor,
        row_axes_w: torch.Tensor,
        col_axes_w: torch.Tensor,
        ray_dir: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fallback_row, fallback_col = self._batched_grid_metric_vectors(surface_points_w, surface_valid, row_axes_w, col_axes_w)
        starts = surface_points_w - surface_raw_m[..., None] * ray_dir[:, None, None, :]
        valid = (
            surface_valid.bool()
            & torch.isfinite(surface_raw_m)
            & (surface_raw_m > 0.0)
            & torch.isfinite(starts).all(dim=-1)
        )
        col_deltas = starts[:, :, 1:, :] - starts[:, :, :-1, :]
        col_valid = valid[:, :, 1:] & valid[:, :, :-1] & torch.isfinite(col_deltas).all(dim=-1)
        row_deltas = starts[:, 1:, :, :] - starts[:, :-1, :, :]
        row_valid = valid[:, 1:, :] & valid[:, :-1, :] & torch.isfinite(row_deltas).all(dim=-1)
        col_count = torch.count_nonzero(col_valid, dim=(1, 2)).to(dtype=torch.float32)
        row_count = torch.count_nonzero(row_valid, dim=(1, 2)).to(dtype=torch.float32)
        col_sum = torch.where(col_valid[..., None], col_deltas, torch.zeros_like(col_deltas)).sum(dim=(1, 2))
        row_sum = torch.where(row_valid[..., None], row_deltas, torch.zeros_like(row_deltas)).sum(dim=(1, 2))
        col_vec = torch.where((col_count > 0.0)[:, None], col_sum / torch.clamp(col_count[:, None], min=1.0), fallback_col)
        row_vec = torch.where((row_count > 0.0)[:, None], row_sum / torch.clamp(row_count[:, None], min=1.0), fallback_row)
        return row_vec.to(dtype=torch.float32), col_vec.to(dtype=torch.float32)

    def _ray_grid_coordinates_batched(
        self,
        points_w: torch.Tensor,
        ray_starts_w: torch.Tensor,
        ray_start_valid: torch.Tensor,
        ray_dir: torch.Tensor,
        ray_row_vec: torch.Tensor,
        ray_col_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map world points into the padded exact row-wise camera-range ray grid."""

        starts = ray_starts_w.to(device=self.device, dtype=torch.float32)
        valid = ray_start_valid.to(device=self.device, dtype=torch.bool)
        points = points_w.to(device=self.device, dtype=torch.float32)
        if starts.ndim != 4 or starts.shape[-1] != 3 or valid.shape != starts.shape[:3]:
            raise ValueError("Ray-grid coordinate mapping expects starts [B,H,W,3] and valid [B,H,W]")
        if points.ndim != 3 or points.shape[0] != starts.shape[0] or points.shape[-1] != 3:
            raise ValueError("Ray-grid coordinate mapping expects points [B,N,3]")

        batch_count, height, width, _ = starts.shape
        padding_cells = float(self.cfg.object_sample_roi_boundary_padding_cells)
        if not np.isfinite(padding_cells):
            padding_cells = 0.0
        padding_cells = max(0.0, padding_cells)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=self.device),
            torch.arange(width, dtype=torch.float32, device=self.device),
            indexing="ij",
        )

        # Preserve the old affine result as a per-slot fallback for malformed
        # or partially unavailable grids. Valid calibrated grids take the
        # exact row-wise path below.
        affine_origin_samples = (
            starts
            - grid_x[None, :, :, None] * ray_col_vec[:, None, None, :]
            - grid_y[None, :, :, None] * ray_row_vec[:, None, None, :]
        )
        valid_count = torch.count_nonzero(valid, dim=(1, 2)).to(dtype=torch.float32)
        affine_origin = torch.where(
            valid[..., None],
            affine_origin_samples,
            torch.zeros_like(affine_origin_samples),
        ).sum(dim=(1, 2))
        affine_origin = affine_origin / torch.clamp(valid_count[:, None], min=1.0)
        affine_basis = torch.stack((ray_col_vec, ray_row_vec, ray_dir), dim=2).to(dtype=torch.float32)
        affine_inverse = torch.linalg.pinv(affine_basis)
        affine_coords = torch.matmul(
            points - affine_origin[:, None, :],
            affine_inverse.transpose(1, 2),
        )
        affine_x = affine_coords[..., 0]
        affine_y = affine_coords[..., 1]
        affine_depth = affine_coords[..., 2]
        affine_inside = (
            torch.isfinite(affine_x)
            & torch.isfinite(affine_y)
            & (affine_x >= -padding_cells)
            & (affine_x <= float(width - 1) + padding_cells)
            & (affine_y >= -padding_cells)
            & (affine_y <= float(height - 1) + padding_cells)
        )

        fallback_ray = torch.tensor(
            [1.0, 0.0, 0.0],
            dtype=torch.float32,
            device=self.device,
        ).expand(batch_count, 3)
        ray_axis = self._normalize_rows_batched(ray_dir, fallback_ray)
        fallback_col = self._normalize_rows_batched(
            ray_col_vec,
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device),
        )
        col_planar = ray_col_vec - ray_axis * torch.sum(ray_col_vec * ray_axis, dim=-1, keepdim=True)
        col_axis = self._normalize_rows_batched(col_planar, fallback_col)
        fallback_row = self._normalize_rows_batched(
            ray_row_vec,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device),
        )
        row_planar = ray_row_vec - ray_axis * torch.sum(ray_row_vec * ray_axis, dim=-1, keepdim=True)
        row_planar = row_planar - col_axis * torch.sum(row_planar * col_axis, dim=-1, keepdim=True)
        row_axis = self._normalize_rows_batched(row_planar, fallback_row)

        plane_origin = torch.where(valid[..., None], starts, torch.zeros_like(starts)).sum(dim=(1, 2))
        plane_origin = plane_origin / torch.clamp(valid_count[:, None], min=1.0)
        start_delta = starts - plane_origin[:, None, None, :]
        point_delta = points - plane_origin[:, None, :]
        grid_u = torch.sum(start_delta * col_axis[:, None, None, :], dim=-1)
        grid_v = torch.sum(start_delta * row_axis[:, None, None, :], dim=-1)
        point_u = torch.sum(point_delta * col_axis[:, None, :], dim=-1)
        point_v = torch.sum(point_delta * row_axis[:, None, :], dim=-1)
        exact_depth = torch.sum(point_delta * ray_axis[:, None, :], dim=-1)

        row_count = torch.count_nonzero(valid, dim=2)
        row_v = torch.where(valid, grid_v, torch.zeros_like(grid_v)).sum(dim=2)
        row_v = row_v / torch.clamp(row_count.to(dtype=torch.float32), min=1.0)
        positive_inf = torch.full_like(grid_u, torch.inf)
        negative_inf = torch.full_like(grid_u, -torch.inf)
        row_u_min = torch.amin(torch.where(valid, grid_u, positive_inf), dim=2)
        row_u_max = torch.amax(torch.where(valid, grid_u, negative_inf), dim=2)

        if height > 1:
            finite_point_v = torch.nan_to_num(point_v, nan=0.0, posinf=0.0, neginf=0.0)
            upper_row = torch.searchsorted(row_v.contiguous(), finite_point_v.contiguous(), right=True)
            upper_row = torch.clamp(upper_row, min=1, max=height - 1)
            lower_row = upper_row - 1
            lower_v = torch.gather(row_v, 1, lower_row)
            upper_v = torch.gather(row_v, 1, upper_row)
            row_alpha_unclamped = (point_v - lower_v) / torch.clamp(upper_v - lower_v, min=1.0e-9)
            row_alpha = torch.clamp(row_alpha_unclamped, 0.0, 1.0)
            exact_y = lower_row.to(dtype=torch.float32) + row_alpha_unclamped
            lower_u_min = torch.gather(row_u_min, 1, lower_row)
            upper_u_min = torch.gather(row_u_min, 1, upper_row)
            lower_u_max = torch.gather(row_u_max, 1, lower_row)
            upper_u_max = torch.gather(row_u_max, 1, upper_row)
            point_u_min = lower_u_min + row_alpha * (upper_u_min - lower_u_min)
            point_u_max = lower_u_max + row_alpha * (upper_u_max - lower_u_max)
            row_monotonic = torch.all((row_v[:, 1:] - row_v[:, :-1]) > 1.0e-9, dim=1)
            v_inside = (
                (exact_y >= -padding_cells)
                & (exact_y <= float(height - 1) + padding_cells)
            )
        else:
            exact_y = torch.zeros_like(point_v)
            point_u_min = row_u_min[:, :1].expand_as(point_u)
            point_u_max = row_u_max[:, :1].expand_as(point_u)
            row_monotonic = torch.ones(batch_count, dtype=torch.bool, device=self.device)
            v_inside = torch.isfinite(point_v)

        point_u_extent = point_u_max - point_u_min
        if width > 1:
            exact_x = (point_u - point_u_min) / torch.clamp(point_u_extent, min=1.0e-9) * float(width - 1)
            u_inside = (
                (exact_x >= -padding_cells)
                & (exact_x <= float(width - 1) + padding_cells)
            )
            row_span_valid = torch.all((row_u_max - row_u_min) > 1.0e-9, dim=1)
        else:
            exact_x = torch.zeros_like(point_u)
            u_inside = torch.isfinite(point_u)
            row_span_valid = torch.ones(batch_count, dtype=torch.bool, device=self.device)

        exact_grid_valid = (
            (valid_count > 0.0)
            & torch.all(row_count > 0, dim=1)
            & row_monotonic
            & row_span_valid
            & torch.isfinite(row_v).all(dim=1)
            & torch.isfinite(row_u_min).all(dim=1)
            & torch.isfinite(row_u_max).all(dim=1)
        )
        exact_inside = (
            v_inside
            & u_inside
            & torch.isfinite(exact_x)
            & torch.isfinite(exact_y)
            & torch.isfinite(exact_depth)
        )
        use_exact = exact_grid_valid[:, None]
        return (
            torch.where(use_exact, exact_x, affine_x).to(dtype=torch.float32),
            torch.where(use_exact, exact_y, affine_y).to(dtype=torch.float32),
            torch.where(use_exact, exact_depth, affine_depth).to(dtype=torch.float32),
            torch.where(use_exact, exact_inside, affine_inside),
        )

    def _estimate_object_sample_sdf_batched(
        self,
        indenter_points_w: torch.Tensor,
        depth_m: torch.Tensor,
        surface_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        object_points_w: torch.Tensor,
        object_valid: torch.Tensor,
        surface_raw_m: torch.Tensor,
        ray_dir: torch.Tensor,
        ray_row_vec: torch.Tensor,
        ray_col_vec: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_count, height, width = depth_m.shape
        point_count = int(indenter_points_w.shape[1])
        contact_threshold_m = float(self.cfg.contact_threshold_mm) * 1.0e-3
        raw_valid = (
            surface_valid.bool()
            & torch.isfinite(surface_raw_m)
            & (surface_raw_m > 0.0)
            & torch.isfinite(surface_points_w).all(dim=-1)
        )
        starts = surface_points_w - surface_raw_m[..., None] * ray_dir[:, None, None, :]
        x, y, point_depth, uv_in_bounds = self._ray_grid_coordinates_batched(
            indenter_points_w,
            starts,
            raw_valid,
            ray_dir,
            ray_row_vec,
            ray_col_vec,
        )
        sample_x = torch.clamp(x, min=0.0, max=float(width - 1))
        sample_y = torch.clamp(y, min=0.0, max=float(height - 1))
        surface_depth, lookup_valid = self._bilinear_sample_batched(
            surface_raw_m,
            raw_valid,
            sample_x,
            sample_y,
        )
        inside = uv_in_bounds & lookup_valid & torch.isfinite(point_depth) & (point_depth > 0.0)
        surface_depth_max = torch.where(raw_valid, surface_raw_m, torch.zeros_like(surface_raw_m)).amax(dim=(1, 2))
        fallback_depth = surface_depth_max[:, None] * float(self.cfg.object_sample_roi_invalid_depth_ratio)
        local_depth_limit = surface_depth + float(self.cfg.object_sample_roi_surface_margin_m)
        depth_limit = torch.where(lookup_valid, local_depth_limit, fallback_depth)
        roi_candidate = (
            uv_in_bounds
            & torch.isfinite(point_depth)
            & (point_depth > 0.0)
            & torch.isfinite(depth_limit)
            & (point_depth <= depth_limit)
            & (depth_limit > 0.0)
        )
        roi_priority = torch.where(roi_candidate, depth_limit - point_depth, torch.full_like(point_depth, -torch.inf))

        flat_depth = depth_m.reshape(batch_count, -1)
        flat_surface = surface_points_w.reshape(batch_count, -1, 3)
        flat_object = object_points_w.reshape(batch_count, -1, 3)
        flat_surface_valid = surface_valid.reshape(batch_count, -1).bool()
        flat_object_valid = object_valid.reshape(batch_count, -1).bool()
        dense_contact = (
            flat_surface_valid
            & flat_object_valid
            & torch.isfinite(flat_depth)
            & (flat_depth > contact_threshold_m)
            & torch.isfinite(flat_surface).all(dim=-1)
            & torch.isfinite(flat_object).all(dim=-1)
        )
        dense_contact_active = torch.any(dense_contact, dim=1)
        sample_contact = (
            roi_candidate
            & inside
            & (point_depth < surface_depth)
            & dense_contact_active[:, None]
        )
        full_force_slot_active = torch.any(sample_contact, dim=1)
        slot_active = torch.any(roi_candidate, dim=1)
        roi_candidate = roi_candidate & slot_active[:, None]
        roi_priority = torch.where(roi_candidate, roi_priority, torch.full_like(roi_priority, -torch.inf))
        selected_ids, roi_valid, history_valid = self._select_object_sample_roi_batched(
            roi_candidate,
            roi_priority,
            sample_contact,
        )

        x_roi = self._gather_roi_batched(x, selected_ids, roi_valid, fill=float("nan"))
        y_roi = self._gather_roi_batched(y, selected_ids, roi_valid, fill=float("nan"))
        point_depth_roi = self._gather_roi_batched(point_depth, selected_ids, roi_valid, fill=0.0)
        surface_depth_roi = self._gather_roi_batched(surface_depth, selected_ids, roi_valid, fill=0.0)
        inside_roi = self._gather_roi_batched(inside.to(dtype=torch.float32), selected_ids, roi_valid, fill=0.0) > 0.5

        signed_roi = point_depth_roi - surface_depth_roi
        sdf = torch.where(inside_roi, signed_roi, torch.ones_like(signed_roi))
        sample_uv = torch.stack((x_roi, y_roi), dim=-1).to(dtype=torch.float32)
        sample_uv = torch.where(inside_roi[..., None], sample_uv, torch.full_like(sample_uv, float("nan")))
        roi_count = int(selected_ids.shape[1])
        normals = ray_dir[:, None, :].expand(batch_count, roi_count, 3).contiguous()
        slot_active = slot_active & torch.any(roi_valid, dim=1)
        force_slot_active = full_force_slot_active & torch.any((sdf < 0.0) & roi_valid, dim=1)

        return (
            selected_ids,
            roi_valid,
            history_valid,
            sdf.to(dtype=torch.float32),
            normals.to(dtype=torch.float32),
            sample_uv,
            slot_active,
            force_slot_active,
        )

    def _estimate_all_object_sample_sdf_batched(
        self,
        indenter_points_w: torch.Tensor,
        depth_m: torch.Tensor,
        surface_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        object_points_w: torch.Tensor,
        object_valid: torch.Tensor,
        surface_raw_m: torch.Tensor,
        ray_dir: torch.Tensor,
        ray_row_vec: torch.Tensor,
        ray_col_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate SDF for every object sample without applying the OURS ROI selector."""

        batch_count, height, width = depth_m.shape
        raw_valid = (
            surface_valid.bool()
            & torch.isfinite(surface_raw_m)
            & (surface_raw_m > 0.0)
            & torch.isfinite(surface_points_w).all(dim=-1)
        )
        starts = surface_points_w - surface_raw_m[..., None] * ray_dir[:, None, None, :]
        x, y, point_depth, uv_in_bounds = self._ray_grid_coordinates_batched(
            indenter_points_w,
            starts,
            raw_valid,
            ray_dir,
            ray_row_vec,
            ray_col_vec,
        )
        sample_x = torch.clamp(x, min=0.0, max=float(width - 1))
        sample_y = torch.clamp(y, min=0.0, max=float(height - 1))
        surface_depth, lookup_valid = self._bilinear_sample_batched(
            surface_raw_m,
            raw_valid,
            sample_x,
            sample_y,
        )
        inside = uv_in_bounds & lookup_valid & torch.isfinite(point_depth) & (point_depth > 0.0)
        signed = point_depth - surface_depth
        sdf = torch.where(inside, signed, torch.ones_like(signed)).to(dtype=torch.float32)

        threshold_m = float(self.cfg.contact_threshold_mm) * 1.0e-3
        dense_contact = (
            surface_valid.bool()
            & object_valid.bool()
            & torch.isfinite(depth_m)
            & (depth_m > threshold_m)
            & torch.isfinite(surface_points_w).all(dim=-1)
            & torch.isfinite(object_points_w).all(dim=-1)
        )
        slot_active = torch.any(dense_contact, dim=(1, 2)) & torch.any(sdf < 0.0, dim=1)
        normals = ray_dir[:, None, :].expand(batch_count, indenter_points_w.shape[1], 3).contiguous()
        return sdf, normals.to(dtype=torch.float32), slot_active

    def _static_sensor_bilinear_stencil(
        self,
        sensor_ids: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        sensor_count: int,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Build bilinear gather indices without expanding a static grid over all environments."""

        x = x.to(device=self.device, dtype=torch.float32)
        y = y.to(device=self.device, dtype=torch.float32)
        sensor_ids = sensor_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        if x.shape != y.shape or x.ndim != 2 or int(x.shape[0]) != int(sensor_ids.shape[0]):
            raise ValueError("Static sensor bilinear sampling expects x/y [B,N] and sensor_ids [B]")
        sensor_inside = (sensor_ids >= 0) & (sensor_ids < int(sensor_count))
        safe_sensor_ids = torch.clamp(sensor_ids, 0, max(0, int(sensor_count) - 1))
        inside = (
            sensor_inside[:, None]
            & torch.isfinite(x)
            & torch.isfinite(y)
            & (x >= 0.0)
            & (y >= 0.0)
            & (x <= float(width - 1))
            & (y <= float(height - 1))
        )
        x0 = torch.floor(torch.clamp(x, 0.0, float(width - 1))).to(torch.long)
        y0 = torch.floor(torch.clamp(y, 0.0, float(height - 1))).to(torch.long)
        x1 = torch.minimum(x0 + 1, torch.full_like(x0, int(width) - 1))
        y1 = torch.minimum(y0 + 1, torch.full_like(y0, int(height) - 1))
        wx = torch.clamp(x - x0.to(dtype=torch.float32), 0.0, 1.0)
        wy = torch.clamp(y - y0.to(dtype=torch.float32), 0.0, 1.0)
        sensor_offset = safe_sensor_ids[:, None] * int(height) * int(width)

        def _index(yy: torch.Tensor, xx: torch.Tensor) -> torch.Tensor:
            return sensor_offset + yy * int(width) + xx

        indices = (
            _index(y0, x0),
            _index(y0, x1),
            _index(y1, x0),
            _index(y1, x1),
        )
        weights = (
            (1.0 - wx) * (1.0 - wy),
            wx * (1.0 - wy),
            (1.0 - wx) * wy,
            wx * wy,
        )
        return inside, indices, weights

    def _bilinear_sample_static_sensor_field_batched(
        self,
        values: torch.Tensor,
        valid: torch.Tensor,
        sensor_ids: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one static vector grid per finger without making a copy per environment."""

        values = values.to(device=self.device, dtype=torch.float32)
        valid = valid.to(device=self.device, dtype=torch.bool)
        if values.ndim != 4 or valid.shape != values.shape[:-1]:
            raise ValueError("Static vector fields must have shapes [S,H,W,C] and [S,H,W]")
        sensor_count, height, width, channels = values.shape
        inside, indices, weights = self._static_sensor_bilinear_stencil(
            sensor_ids,
            x,
            y,
            sensor_count=int(sensor_count),
            height=int(height),
            width=int(width),
        )
        flat_values = values.reshape(-1, channels)
        flat_valid = valid.reshape(-1)
        sampled_sum = torch.zeros((*x.shape, channels), dtype=torch.float32, device=self.device)
        weight_sum = torch.zeros_like(x, dtype=torch.float32, device=self.device)
        for index, weight in zip(indices, weights, strict=True):
            sample = flat_values[index]
            sample_valid = flat_valid[index] & torch.isfinite(sample).all(dim=-1)
            masked_weight = weight * sample_valid.to(dtype=torch.float32)
            sampled_sum = sampled_sum + masked_weight[..., None] * sample
            weight_sum = weight_sum + masked_weight
        sampled = sampled_sum / torch.clamp(weight_sum[..., None], min=1.0e-12)
        return sampled.to(dtype=torch.float32), inside & (weight_sum > 1.0e-6)

    def _bilinear_sample_reference_surface_batched(
        self,
        sensor_ids: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate the cached ray-hit surface directly in each finger link frame."""

        assert self._shear_reference_starts_l is not None
        assert self._shear_reference_directions_l is not None
        assert self._shear_reference_depth_m is not None
        assert self._shear_reference_valid is not None
        starts = self._shear_reference_starts_l
        directions = self._shear_reference_directions_l
        depth = self._shear_reference_depth_m
        valid = self._shear_reference_valid
        sensor_count, height, width, _ = starts.shape
        inside, indices, weights = self._static_sensor_bilinear_stencil(
            sensor_ids,
            x,
            y,
            sensor_count=int(sensor_count),
            height=int(height),
            width=int(width),
        )
        flat_starts = starts.reshape(-1, 3)
        flat_directions = directions.reshape(-1, 3)
        flat_depth = depth.reshape(-1)
        flat_valid = valid.reshape(-1)
        point_sum = torch.zeros((*x.shape, 3), dtype=torch.float32, device=self.device)
        weight_sum = torch.zeros_like(x, dtype=torch.float32, device=self.device)
        for index, weight in zip(indices, weights, strict=True):
            corner_depth = flat_depth[index]
            corner = flat_starts[index] + flat_directions[index] * corner_depth[..., None]
            corner_valid = (
                flat_valid[index]
                & torch.isfinite(corner).all(dim=-1)
                & torch.isfinite(corner_depth)
                & (corner_depth > 0.0)
            )
            masked_weight = weight * corner_valid.to(dtype=torch.float32)
            point_sum = point_sum + masked_weight[..., None] * corner
            weight_sum = weight_sum + masked_weight
        points_l = point_sum / torch.clamp(weight_sum[..., None], min=1.0e-12)
        return points_l.to(dtype=torch.float32), inside & (weight_sum > 1.0e-6)

    def _curved_surface_points_from_uv_batched(
        self,
        affected_uv: torch.Tensor,
        surface_frame_pose_wxyz: torch.Tensor | None,
        *,
        batch_offset: int = 0,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Map coarse affected UV to the real 320x240 curved surface in world coordinates."""

        reference_fields = (
            self._shear_reference_starts_l,
            self._shear_reference_directions_l,
            self._shear_reference_depth_m,
            self._shear_reference_valid,
            self._shear_coarse_xy_camera_m,
            self._shear_reference_bounds_camera_m,
            self._shear_reference_row_axis_camera,
        )
        if surface_frame_pose_wxyz is None or any(value is None for value in reference_fields):
            return None, None
        assert self._shear_reference_starts_l is not None
        assert self._shear_coarse_xy_camera_m is not None
        assert self._shear_reference_bounds_camera_m is not None
        assert self._shear_reference_row_axis_camera is not None
        affected_uv = affected_uv.to(device=self.device, dtype=torch.float32)
        pose = surface_frame_pose_wxyz.to(device=self.device, dtype=torch.float32).reshape(-1, 7)
        if affected_uv.ndim != 3 or affected_uv.shape[-1] != 2 or int(affected_uv.shape[0]) != int(pose.shape[0]):
            raise ValueError("Curved Shear lookup expects affected_uv [B,N,2] and surface poses [B,7]")

        batch_count = int(affected_uv.shape[0])
        sensor_count = int(self._shear_reference_starts_l.shape[0])
        sensor_ids = torch.remainder(
            torch.arange(batch_count, device=self.device, dtype=torch.long) + int(batch_offset),
            sensor_count,
        )
        coarse_xy_valid = torch.isfinite(self._shear_coarse_xy_camera_m).all(dim=-1)
        camera_xy, coarse_lookup_valid = self._bilinear_sample_static_sensor_field_batched(
            self._shear_coarse_xy_camera_m,
            coarse_xy_valid,
            sensor_ids,
            affected_uv[..., 0],
            affected_uv[..., 1],
        )
        bounds = self._shear_reference_bounds_camera_m[sensor_ids]
        x_span = bounds[:, 1] - bounds[:, 0]
        y_span = bounds[:, 3] - bounds[:, 2]
        bounds_valid = (
            torch.isfinite(bounds).all(dim=-1)
            & (x_span > 1.0e-12)
            & (y_span > 1.0e-12)
        )
        normalized_x = (camera_xy[..., 0] - bounds[:, None, 0]) / torch.clamp(
            x_span[:, None],
            min=1.0e-12,
        )
        normalized_y = (camera_xy[..., 1] - bounds[:, None, 2]) / torch.clamp(
            y_span[:, None],
            min=1.0e-12,
        )
        reference_height = int(self._shear_reference_starts_l.shape[1])
        reference_width = int(self._shear_reference_starts_l.shape[2])
        row_is_camera_y = self._shear_reference_row_axis_camera[sensor_ids] == 1
        reference_x = torch.where(
            row_is_camera_y[:, None],
            normalized_x * float(reference_width - 1),
            normalized_y * float(reference_width - 1),
        )
        reference_y = torch.where(
            row_is_camera_y[:, None],
            normalized_y * float(reference_height - 1),
            normalized_x * float(reference_height - 1),
        )
        points_l, reference_lookup_valid = self._bilinear_sample_reference_surface_batched(
            sensor_ids,
            reference_x,
            reference_y,
        )
        points_w = self._pose_apply_per_batch_t(pose, points_l)
        valid = (
            coarse_lookup_valid
            & bounds_valid[:, None]
            & reference_lookup_valid
            & torch.isfinite(points_w).all(dim=-1)
        )
        return points_w.to(dtype=torch.float32), valid

    def _bilinear_sample_batched(
        self,
        values: torch.Tensor,
        valid: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = values.to(dtype=torch.float32)
        valid = valid.bool()
        x = x.to(dtype=torch.float32)
        y = y.to(dtype=torch.float32)
        batch_count, height, width = values.shape
        inside = (
            torch.isfinite(x)
            & torch.isfinite(y)
            & (x >= 0.0)
            & (y >= 0.0)
            & (x <= float(width - 1))
            & (y <= float(height - 1))
        )
        x0 = torch.floor(torch.clamp(x, 0.0, float(width - 1))).to(torch.long)
        y0 = torch.floor(torch.clamp(y, 0.0, float(height - 1))).to(torch.long)
        x1 = torch.minimum(x0 + 1, torch.full_like(x0, int(width) - 1))
        y1 = torch.minimum(y0 + 1, torch.full_like(y0, int(height) - 1))
        wx = torch.clamp(x - x0.to(dtype=torch.float32), 0.0, 1.0)
        wy = torch.clamp(y - y0.to(dtype=torch.float32), 0.0, 1.0)

        flat_values = values.reshape(batch_count, -1)
        flat_valid = valid.reshape(batch_count, -1)

        def _gather(grid: torch.Tensor, yy: torch.Tensor, xx: torch.Tensor) -> torch.Tensor:
            return torch.gather(grid, 1, yy * int(width) + xx)

        v00 = _gather(flat_values, y0, x0)
        v10 = _gather(flat_values, y0, x1)
        v01 = _gather(flat_values, y1, x0)
        v11 = _gather(flat_values, y1, x1)
        m00 = _gather(flat_valid, y0, x0) & torch.isfinite(v00) & (v00 > 0.0)
        m10 = _gather(flat_valid, y0, x1) & torch.isfinite(v10) & (v10 > 0.0)
        m01 = _gather(flat_valid, y1, x0) & torch.isfinite(v01) & (v01 > 0.0)
        m11 = _gather(flat_valid, y1, x1) & torch.isfinite(v11) & (v11 > 0.0)
        w00 = (1.0 - wx) * (1.0 - wy)
        w10 = wx * (1.0 - wy)
        w01 = (1.0 - wx) * wy
        w11 = wx * wy
        m00f = m00.to(dtype=torch.float32)
        m10f = m10.to(dtype=torch.float32)
        m01f = m01.to(dtype=torch.float32)
        m11f = m11.to(dtype=torch.float32)
        weight_sum = w00 * m00f + w10 * m10f + w01 * m01f + w11 * m11f
        sampled = (w00 * m00f * v00 + w10 * m10f * v10 + w01 * m01f * v01 + w11 * m11f * v11) / torch.clamp(
            weight_sum, min=1.0e-12
        )
        return sampled.to(dtype=torch.float32), inside & (weight_sum > 1.0e-6)

    def _compute_shear_from_samples_batched(
        self,
        marker_uv: torch.Tensor,
        marker_normals: torch.Tensor,
        marker_valid: torch.Tensor,
        obj: torch.Tensor,
        sdf: torch.Tensor,
        fbar: torch.Tensor,
        normals: torch.Tensor,
        sample_uv: torch.Tensor,
        ray_row_vec: torch.Tensor,
        ray_col_vec: torch.Tensor,
        *,
        marker_points: torch.Tensor | None = None,
        surface_frame_pose_wxyz: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_count, marker_count, _ = marker_normals.shape
        point_count = int(obj.shape[1])
        contact = (
            (sdf < 0.0)
            & torch.isfinite(sdf)
            & torch.isfinite(obj).all(dim=-1)
            & torch.isfinite(fbar).all(dim=-1)
            & torch.isfinite(normals).all(dim=-1)
            & torch.isfinite(sample_uv).all(dim=-1)
        )
        normal_force = torch.sum(fbar * normals, dim=-1)
        h = torch.where(contact, torch.clamp(normal_force, min=0.0), torch.zeros_like(normal_force))
        fbar_tangent = fbar - normal_force[..., None] * normals
        fbar_tangent = torch.where(contact[..., None], fbar_tangent, torch.zeros_like(fbar_tangent))
        basis = torch.stack((ray_col_vec, ray_row_vec), dim=-1).to(dtype=torch.float32)
        basis_ok = torch.isfinite(basis).all(dim=(1, 2)) & (
            torch.linalg.norm(torch.linalg.cross(ray_col_vec, ray_row_vec, dim=-1), dim=-1) > 1.0e-12
        )
        pinv = torch.linalg.pinv(basis)
        uv_delta = torch.einsum("bnc,bdc->bnd", fbar_tangent, pinv)
        uv_delta = torch.where(basis_ok[:, None, None], uv_delta, torch.zeros_like(uv_delta))
        sample_uv = torch.where(contact[..., None], sample_uv, torch.zeros_like(sample_uv))
        affected_uv = sample_uv + uv_delta
        use_curved_distance = (
            marker_points is not None
            and surface_frame_pose_wxyz is not None
            and self._shear_reference_starts_l is not None
        )

        out = torch.zeros((batch_count, marker_count, 3), dtype=torch.float32, device=self.device)
        batch_chunk = max(1, min(int(self.cfg.batch_slot_chunk_size), 16 if point_count > 1024 else int(self.cfg.batch_slot_chunk_size)))
        point_chunk = max(1, min(int(self.cfg.object_chunk_size), 1024))
        target_uv = marker_uv.to(dtype=torch.float32)
        target_u = target_uv[:, 0].reshape(1, marker_count, 1)
        target_v = target_uv[:, 1].reshape(1, marker_count, 1)
        lookup_chunk = (
            max(batch_chunk, int(self.cfg.curved_surface_lookup_chunk_size))
            if use_curved_distance
            else batch_chunk
        )
        for lookup_start in range(0, batch_count, lookup_chunk):
            lookup_end = min(lookup_start + lookup_chunk, batch_count)
            curved_source_points = None
            curved_source_valid = None
            if use_curved_distance:
                curved_source_points, curved_source_valid = self._curved_surface_points_from_uv_batched(
                    affected_uv[lookup_start:lookup_end],
                    surface_frame_pose_wxyz[lookup_start:lookup_end],
                    batch_offset=lookup_start,
                )
                assert curved_source_points is not None
                assert curved_source_valid is not None
            for b_start in range(lookup_start, lookup_end, batch_chunk):
                b_end = min(b_start + batch_chunk, lookup_end)
                accum = torch.zeros((b_end - b_start, marker_count, 3), dtype=torch.float32, device=self.device)
                normals_chunk = marker_normals[b_start:b_end, :, None, :]
                curve_start = b_start - lookup_start
                curve_end = b_end - lookup_start
                for p_start in range(0, point_count, point_chunk):
                    p_end = min(p_start + point_chunk, point_count)
                    source_uv = affected_uv[b_start:b_end, p_start:p_end]
                    if use_curved_distance:
                        assert marker_points is not None
                        assert curved_source_points is not None
                        dvec = (
                            marker_points[b_start:b_end, :, None, :]
                            - curved_source_points[curve_start:curve_end, None, p_start:p_end, :]
                        )
                    else:
                        source_u = source_uv[..., 0].reshape(b_end - b_start, 1, p_end - p_start)
                        source_v = source_uv[..., 1].reshape(b_end - b_start, 1, p_end - p_start)
                        du = target_u - source_u
                        dv = target_v - source_v
                        dvec = (
                            du[..., None] * ray_col_vec[b_start:b_end, None, None, :]
                            + dv[..., None] * ray_row_vec[b_start:b_end, None, None, :]
                        )
                    dist2 = torch.sum(dvec * dvec, dim=-1)
                    weights = torch.exp(-float(self.cfg.lambda_shear) * dist2).to(dtype=torch.float32)
                    contact_chunk = contact[b_start:b_end, None, p_start:p_end]
                    weights = torch.where(contact_chunk, weights, torch.zeros_like(weights))
                    if use_curved_distance:
                        assert curved_source_valid is not None
                        weights = torch.where(
                            curved_source_valid[curve_start:curve_end, None, p_start:p_end],
                            weights,
                            torch.zeros_like(weights),
                        )
                    tangent = fbar_tangent[b_start:b_end, None, p_start:p_end, :]
                    normal_component = torch.sum(tangent * normals_chunk, dim=-1, keepdim=True) * normals_chunk
                    transported = tangent - normal_component
                    h_chunk = h[b_start:b_end, None, p_start:p_end, None]
                    accum = accum + torch.sum(
                        h_chunk * -transported * weights[..., None],
                        dim=2,
                    )
                out[b_start:b_end] = float(self.cfg.shear_scale) * accum

        return torch.where(marker_valid[..., None], out, torch.zeros_like(out)).to(dtype=torch.float32)

    def _compute_original_shear_batched(
        self,
        marker_points: torch.Tensor,
        marker_valid: torch.Tensor,
        obj: torch.Tensor,
        sdf: torch.Tensor,
        fbar: torch.Tensor,
        normals: torch.Tensor,
    ) -> torch.Tensor:
        """Original HydroShear shear with exact Euclidean Gaussian propagation."""

        batch_count, marker_count, _ = marker_points.shape
        point_count = int(obj.shape[1])
        out = torch.zeros((batch_count, marker_count, 3), dtype=torch.float32, device=self.device)
        contact = (
            (sdf < 0.0)
            & torch.isfinite(sdf)
            & torch.isfinite(obj).all(dim=-1)
            & torch.isfinite(fbar).all(dim=-1)
            & torch.isfinite(normals).all(dim=-1)
        )
        normalized_normals = self._normalize_rows_batched(
            normals,
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
        )
        normal_force = torch.sum(fbar * normalized_normals, dim=-1)
        h = torch.where(contact, torch.clamp(normal_force, min=0.0), torch.zeros_like(normal_force))
        affected = obj + fbar

        batch_chunk = max(1, int(self.cfg.batch_slot_chunk_size))
        pair_budget = 4_000_000
        for b_start in range(0, batch_count, batch_chunk):
            b_end = min(b_start + batch_chunk, batch_count)
            local_batch = b_end - b_start
            point_chunk = max(1, pair_budget // max(1, local_batch * marker_count))
            point_chunk = min(point_count, max(1, int(self.cfg.object_chunk_size)), point_chunk)
            accum = torch.zeros((local_batch, marker_count, 3), dtype=torch.float32, device=self.device)
            for p_start in range(0, point_count, point_chunk):
                p_end = min(p_start + point_chunk, point_count)
                affected_chunk = affected[b_start:b_end, p_start:p_end]
                dvec = marker_points[b_start:b_end, :, None, :] - affected_chunk[:, None, :, :]
                dist2 = torch.sum(dvec * dvec, dim=-1)
                weights = torch.exp(-float(self.cfg.lambda_shear) * dist2).to(dtype=torch.float32)
                contact_chunk = contact[b_start:b_end, None, p_start:p_end]
                weights = torch.where(contact_chunk, weights, torch.zeros_like(weights))
                contribution = (
                    h[b_start:b_end, None, p_start:p_end, None]
                    * -fbar[b_start:b_end, None, p_start:p_end, :]
                    * weights[..., None]
                )
                accum = accum + torch.sum(contribution, dim=2)
            out[b_start:b_end] = float(self.cfg.shear_scale) * accum

        return torch.where(marker_valid[..., None], out, torch.zeros_like(out)).to(dtype=torch.float32)

    def _batched_surface_motion_frame(
        self,
        surface_flat: torch.Tensor,
        surface_valid: torch.Tensor,
        row_vec: torch.Tensor,
        col_vec: torch.Tensor,
        normal_ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_count = int(surface_flat.shape[0])
        valid = surface_valid.bool() & torch.isfinite(surface_flat).all(dim=-1)
        counts = torch.count_nonzero(valid, dim=1).to(dtype=torch.float32)
        origin_sum = torch.where(valid[..., None], surface_flat, torch.zeros_like(surface_flat)).sum(dim=1)
        origin = origin_sum / torch.clamp(counts[:, None], min=1.0)
        origin = torch.where((counts > 0.0)[:, None], origin, torch.zeros_like(origin))

        fallback_col = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device).expand(batch_count, 3)
        fallback_row = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device).expand(batch_count, 3)
        col_axis = self._normalize_rows_batched(col_vec, fallback_col)
        row_axis = row_vec - col_axis * torch.sum(row_vec * col_axis, dim=-1, keepdim=True)
        row_axis = self._normalize_rows_batched(row_axis, fallback_row)
        normal_axis = torch.linalg.cross(col_axis, row_axis, dim=-1)
        normal_axis = self._normalize_rows_batched(
            normal_axis,
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
        )
        normal_ref = self._normalize_rows_batched(normal_ref, normal_axis)
        flip = torch.sum(normal_axis * normal_ref, dim=-1) < 0.0
        normal_axis = torch.where(flip[:, None], -normal_axis, normal_axis)
        row_axis = row_axis - normal_axis * torch.sum(row_axis * normal_axis, dim=-1, keepdim=True)
        row_axis = row_axis - col_axis * torch.sum(row_axis * col_axis, dim=-1, keepdim=True)
        row_axis = self._normalize_rows_batched(row_axis, torch.linalg.cross(normal_axis, col_axis, dim=-1))
        basis = torch.stack((col_axis, row_axis, normal_axis), dim=2).to(dtype=torch.float32)
        return origin.to(dtype=torch.float32), basis

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def step(
        self,
        penetration_depth_m: np.ndarray,
        surface_points_w: np.ndarray,
        surface_valid: np.ndarray,
        object_points_w: np.ndarray,
        object_valid: np.ndarray,
        object_pose_wxyz: np.ndarray | None = None,
        surface_raw_m: np.ndarray | None = None,
        *,
        surface_normals_w: np.ndarray | None = None,
        ray_directions_w: np.ndarray | torch.Tensor | None = None,
    ) -> CurvedHydroShearOutput:
        depth = self._to_tensor(penetration_depth_m)
        if depth.ndim == 2:
            depth = depth[None, ...]

        surface_points = self._to_tensor(surface_points_w)
        if surface_points.ndim == 3:
            surface_points = surface_points[None, ...]
        surface_valid_arr = self._to_tensor(surface_valid, dtype=torch.bool)
        if surface_valid_arr.ndim == 2:
            surface_valid_arr = surface_valid_arr[None, ...]

        object_points = self._to_tensor(object_points_w)
        if object_points.ndim == 3:
            object_points = object_points[None, ...]
        object_valid_arr = self._to_tensor(object_valid, dtype=torch.bool)
        if object_valid_arr.ndim == 2:
            object_valid_arr = object_valid_arr[None, ...]
        surface_raw = None
        if surface_raw_m is not None:
            surface_raw = self._to_tensor(surface_raw_m)
            if surface_raw.ndim == 2:
                surface_raw = surface_raw[None, ...]
        surface_normals = None
        if surface_normals_w is not None:
            surface_normals = self._to_tensor(surface_normals_w)
            if surface_normals.ndim == 3:
                surface_normals = surface_normals[None, ...]
        ray_directions = None
        if ray_directions_w is not None:
            ray_directions = self._to_tensor(ray_directions_w).reshape(-1, 3)

        counts = [
            depth.shape[0],
            surface_points.shape[0],
            surface_valid_arr.shape[0],
            object_points.shape[0],
            object_valid_arr.shape[0],
        ]
        if surface_raw is not None:
            counts.append(surface_raw.shape[0])
        if surface_normals is not None:
            counts.append(surface_normals.shape[0])
        if ray_directions is not None:
            counts.append(ray_directions.shape[0])
        sensor_count = min(counts)
        self._ensure_prev_buffers(sensor_count)

        flows = []
        images = []
        original_images = []
        marker_displacement_images = []
        sdf_images = []
        projection_images = []
        mdilate_images = []
        object_sample_points_w = []
        all_object_sample_points_w = []
        marker_points_w = []
        marker_normals_w = []
        marker_row_axes_w = []
        marker_col_axes_w = []
        displacements = []
        max_depths = []
        active_markers = []
        active_objects = []
        max_motion_px = []

        for i in range(sensor_count):
            object_pose_i = self._pose_for_sensor_t(object_pose_wxyz, i)
            result = self._step_one(
                i,
                depth[i],
                surface_points[i],
                surface_valid_arr[i],
                object_points[i],
                object_valid_arr[i],
                object_pose_i,
                None if surface_raw is None else surface_raw[i],
                None if surface_normals is None else surface_normals[i],
                None if ray_directions is None else ray_directions[i],
            )
            (
                flow,
                image,
                original_image,
                marker_disp_img,
                sdf_img,
                projection_img,
                mdilate_img,
                sample_points_w,
                all_sample_points_w,
                marker_points_valid_w,
                marker_normals_valid_w,
                marker_row_axes_valid_w,
                marker_col_axes_valid_w,
                disp_m,
                active_marker_count,
                active_object_count,
            ) = result
            flows.append(flow)
            images.append(image)
            original_images.append(original_image)
            marker_displacement_images.append(marker_disp_img)
            sdf_images.append(sdf_img)
            projection_images.append(projection_img)
            mdilate_images.append(mdilate_img)
            object_sample_points_w.append(sample_points_w)
            all_object_sample_points_w.append(all_sample_points_w)
            marker_points_w.append(marker_points_valid_w)
            marker_normals_w.append(marker_normals_valid_w)
            marker_row_axes_w.append(marker_row_axes_valid_w)
            marker_col_axes_w.append(marker_col_axes_valid_w)
            displacements.append(disp_m)
            depth_i = self._finite_tensor(depth[i])
            max_depths.append(torch.amax(depth_i) * 1000.0)
            active_markers.append(torch.as_tensor(active_marker_count, dtype=torch.float32, device=self.device))
            active_objects.append(torch.as_tensor(active_object_count, dtype=torch.float32, device=self.device))
            max_motion_px.append(
                torch.amax(torch.linalg.norm(flow[1] - flow[0], dim=-1)) if flow.numel() else torch.tensor(0.0, device=self.device)
            )

        if not images:
            images = [np.zeros((1, 1, 3), dtype=np.uint8)]
            original_images = [np.zeros((1, 1, 3), dtype=np.uint8)]
            marker_displacement_images = [np.zeros((1, 1, 3), dtype=np.uint8)]
            sdf_images = [np.zeros((1, 1, 3), dtype=np.uint8)]
            projection_images = [np.zeros((1, 1, 3), dtype=np.uint8)]
            mdilate_images = [np.zeros((1, 1, 3), dtype=np.uint8)]
            object_sample_points_w = [np.zeros((0, 3), dtype=np.float32)]
            all_object_sample_points_w = [np.zeros((0, 3), dtype=np.float32)]
            marker_points_w = [np.zeros((0, 3), dtype=np.float32)]
            marker_normals_w = [np.zeros((0, 3), dtype=np.float32)]
            marker_row_axes_w = [np.zeros((0, 3), dtype=np.float32)]
            marker_col_axes_w = [np.zeros((0, 3), dtype=np.float32)]
            flows = [self._empty_tensor(2, 0, 2)]
            displacements = [self._empty_tensor(0, 3)]
            max_depths = [torch.tensor(0.0, dtype=torch.float32, device=self.device)]
            active_markers = [torch.tensor(0.0, dtype=torch.float32, device=self.device)]
            active_objects = [torch.tensor(0.0, dtype=torch.float32, device=self.device)]
            max_motion_px = [torch.tensor(0.0, dtype=torch.float32, device=self.device)]

        return CurvedHydroShearOutput(
            marker_flow=torch.stack(flows, dim=0),
            marker_images=np.stack(images, axis=0),
            original_marker_images=np.stack(original_images, axis=0),
            debug_marker_displacement_images=np.stack(marker_displacement_images, axis=0),
            debug_sdf_images=np.stack(sdf_images, axis=0),
            debug_projection_images=np.stack(projection_images, axis=0),
            debug_mdilate_images=np.stack(mdilate_images, axis=0),
            debug_object_sample_points_w=object_sample_points_w,
            debug_all_object_sample_points_w=all_object_sample_points_w,
            debug_marker_points_w=marker_points_w,
            debug_marker_normals_w=marker_normals_w,
            debug_marker_row_axes_w=marker_row_axes_w,
            debug_marker_col_axes_w=marker_col_axes_w,
            displacement_m=torch.stack(displacements, dim=0),
            max_depth_mm=torch.stack(max_depths).to(dtype=torch.float32),
            active_markers=torch.stack(active_markers).to(dtype=torch.float32),
            active_object_samples=torch.stack(active_objects).to(dtype=torch.float32),
            max_marker_motion_px=torch.stack(max_motion_px).to(dtype=torch.float32),
        )

    def _ensure_prev_buffers(self, sensor_count: int) -> None:
        while len(self._hydrosoft_forces) < sensor_count:
            self._hydrosoft_forces.append(None)
            self._prev_sdf.append(None)
            self._prev_indenter_points.append(None)
            self._prev_normals.append(None)

    def _step_one_displacement_only(
        self,
        sensor_index: int,
        depth_m: torch.Tensor,
        surface_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        object_points_w: torch.Tensor,
        object_valid: torch.Tensor,
        object_pose_wxyz: np.ndarray | torch.Tensor | None,
        surface_raw_m: torch.Tensor | None,
        surface_normals_w: torch.Tensor | None,
        ray_direction_w: torch.Tensor | None,
    ) -> torch.Tensor:
        depth_m = torch.clamp(self._finite_tensor(depth_m), min=0.0)
        surface_points_w = self._finite_tensor(surface_points_w)
        object_points_w = self._finite_tensor(object_points_w)
        surface_normals = None
        if surface_normals_w is not None:
            surface_normals = self._finite_tensor(surface_normals_w)

        h, w = depth_m.shape
        marker_uv_sample = self._scaled_marker_uv_t(int(w), int(h))
        xs = torch.clamp(torch.round(marker_uv_sample[:, 0]).to(torch.long), 0, int(w) - 1)
        ys = torch.clamp(torch.round(marker_uv_sample[:, 1]).to(torch.long), 0, int(h) - 1)

        marker_points = surface_points_w[ys, xs]
        marker_depth = depth_m[ys, xs]
        marker_valid = surface_valid[ys, xs].bool() & torch.isfinite(marker_points).all(dim=-1)
        contact_threshold_m = float(self.cfg.contact_threshold_mm) * 1.0e-3
        marker_contact = marker_valid & (marker_depth > contact_threshold_m)

        surface_row_axes, surface_col_axes = self._marker_tangent_axes_t(surface_points_w, surface_valid, xs, ys)
        marker_normals = self._marker_normals_t(surface_normals, xs, ys, surface_row_axes, surface_col_axes)
        ray_row_vec, ray_col_vec = self._raycaster_metric_vectors_t(
            surface_points_w,
            surface_valid,
            surface_raw_m,
            surface_normals,
            ray_direction_w=ray_direction_w,
            fallback_row_axes=surface_row_axes,
            fallback_col_axes=surface_col_axes,
        )
        mdilate = self._compute_dilation_t(
            marker_points, marker_uv_sample, marker_depth, marker_contact, ray_row_vec, ray_col_vec
        )
        mshear, _mshear_visual, _original_mshear_visual, _active_object_count, _object_sample_uv, _sample_points_w = (
            self._compute_shear_t(
                sensor_index,
                marker_points,
                marker_uv_sample,
                marker_normals,
                marker_valid,
                marker_contact,
                depth_m,
                surface_points_w,
                surface_valid,
                object_points_w,
                object_valid,
                object_pose_wxyz,
                surface_raw_m,
                surface_normals,
                ray_row_vec,
                ray_col_vec,
                ray_direction_w,
            )
        )

        displacement = (mdilate + mshear).to(dtype=torch.float32)
        return torch.where(marker_valid[:, None], displacement, torch.zeros_like(displacement))

    def _step_one(
        self,
        sensor_index: int,
        depth_m: torch.Tensor,
        surface_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        object_points_w: torch.Tensor,
        object_valid: torch.Tensor,
        object_pose_wxyz: np.ndarray | None,
        surface_raw_m: torch.Tensor | None,
        surface_normals_w: np.ndarray | None,
        ray_direction_w: torch.Tensor | None,
    ) -> tuple:
        depth_m = torch.clamp(self._finite_tensor(depth_m), min=0.0)
        surface_points_w = self._finite_tensor(surface_points_w)
        object_points_w = self._finite_tensor(object_points_w)
        surface_normals = None
        if surface_normals_w is not None:
            surface_normals = self._finite_tensor(surface_normals_w)

        h, w = depth_m.shape
        marker_uv_sample = self._scaled_marker_uv_t(int(w), int(h))
        marker_uv_render = self._marker_uv_t.clone()
        render_scale = torch.tensor(
            [
                float(self.cfg.width) / max(1.0, float(w)),
                float(self.cfg.height) / max(1.0, float(h)),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        xs = torch.clamp(torch.round(marker_uv_sample[:, 0]).to(torch.long), 0, int(w) - 1)
        ys = torch.clamp(torch.round(marker_uv_sample[:, 1]).to(torch.long), 0, int(h) - 1)

        marker_points = surface_points_w[ys, xs]
        marker_depth = depth_m[ys, xs]
        marker_valid = surface_valid[ys, xs].bool() & torch.isfinite(marker_points).all(dim=-1)
        contact_threshold_m = float(self.cfg.contact_threshold_mm) * 1.0e-3
        marker_contact = marker_valid & (marker_depth > contact_threshold_m)

        surface_row_axes, surface_col_axes = self._marker_tangent_axes_t(surface_points_w, surface_valid, xs, ys)
        marker_normals = self._marker_normals_t(surface_normals, xs, ys, surface_row_axes, surface_col_axes)
        ray_row_vec, ray_col_vec = self._raycaster_metric_vectors_t(
            surface_points_w,
            surface_valid,
            surface_raw_m,
            surface_normals,
            ray_direction_w=ray_direction_w,
            fallback_row_axes=surface_row_axes,
            fallback_col_axes=surface_col_axes,
        )
        ray_row_axis, ray_col_axis = self._orthonormalize_axes_t(
            ray_row_vec,
            ray_col_vec,
            self._safe_normalize_t(
                torch.mean(surface_row_axes, dim=0),
                torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device),
            ),
            self._safe_normalize_t(
                torch.mean(surface_col_axes, dim=0),
                torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
            ),
        )
        row_axes = ray_row_axis.reshape(1, 3).repeat(marker_points.shape[0], 1)
        col_axes = ray_col_axis.reshape(1, 3).repeat(marker_points.shape[0], 1)
        current_pose = self._valid_pose_t(object_pose_wxyz)
        all_sample_points_w = self._all_object_sample_points_w_t(current_pose)

        mdilate = self._compute_dilation_t(
            marker_points, marker_uv_sample, marker_depth, marker_contact, ray_row_vec, ray_col_vec
        )
        original_mdilate = self._compute_original_dilation_t(marker_points, marker_depth, marker_contact)
        mshear, mshear_visual, original_mshear_visual, active_object_count, object_sample_uv, sample_points_w = (
            self._compute_shear_t(
                sensor_index,
                marker_points,
                marker_uv_sample,
                marker_normals,
                marker_valid,
                marker_contact,
                depth_m,
                surface_points_w,
                surface_valid,
                object_points_w,
                object_valid,
                object_pose_wxyz,
                surface_raw_m,
                surface_normals,
                ray_row_vec,
                ray_col_vec,
                ray_direction_w,
            )
        )

        original_visual_displacement = original_mdilate + original_mshear_visual
        original_visual_displacement = torch.where(marker_valid[:, None], original_visual_displacement, 0.0)
        original_flow = self._flow_from_displacement_t(
            marker_uv_render,
            original_visual_displacement,
            row_axes,
            col_axes,
        )
        original_image = self._render_vector_field_image(
            self._to_numpy(original_flow).astype(np.float32),
            color=_MARKER_GREEN,
            draw_points=False,
        )

        displacement = mdilate + mshear
        displacement = torch.where(marker_valid[:, None], displacement, 0.0).to(dtype=torch.float32)
        visual_displacement = mdilate + mshear_visual
        visual_displacement = torch.where(marker_valid[:, None], visual_displacement, 0.0).to(dtype=torch.float32)
        flow = self._flow_from_displacement_t(
            marker_uv_render,
            visual_displacement,
            row_axes,
            col_axes,
        )
        flow_np = self._to_numpy(flow).astype(np.float32)
        image = self._render_vector_field_image(flow_np, color=_MARKER_GREEN, draw_points=False)
        shear_flow = self._flow_from_displacement_t(
            marker_uv_render,
            mshear_visual,
            row_axes,
            col_axes,
        )
        mdilate_flow = self._flow_from_displacement_t(
            marker_uv_render,
            mdilate,
            row_axes,
            col_axes,
        )
        marker_displacement_img = image.copy()
        sdf_img = self._render_marker_height_image(self._to_numpy(marker_depth).astype(np.float32))
        projection_img = self._render_vector_field_image(
            self._to_numpy(shear_flow).astype(np.float32),
            color=_MARKER_GREEN,
            draw_points=False,
            object_sample_uv=self._to_numpy(object_sample_uv * render_scale.reshape(1, 2)).astype(np.float32),
        )
        mdilate_img = self._render_vector_field_image(
            self._to_numpy(mdilate_flow).astype(np.float32),
            color=_MARKER_GREEN,
            draw_points=False,
        )
        marker_valid_np = self._to_numpy(marker_valid).astype(bool)
        row_axes_np = self._to_numpy(row_axes).astype(np.float32)
        col_axes_np = self._to_numpy(col_axes).astype(np.float32)
        return (
            flow,
            image,
            original_image,
            marker_displacement_img,
            sdf_img,
            projection_img,
            mdilate_img,
            self._to_numpy(sample_points_w).astype(np.float32),
            self._to_numpy(all_sample_points_w).astype(np.float32),
            self._to_numpy(marker_points[marker_valid]).astype(np.float32),
            self._to_numpy(marker_normals[marker_valid]).astype(np.float32),
            row_axes_np[marker_valid_np].astype(np.float32),
            col_axes_np[marker_valid_np].astype(np.float32),
            displacement,
            int(torch.count_nonzero(marker_contact).detach().cpu()),
            int(active_object_count),
        )

    def _all_object_sample_points_w_t(self, current_pose: torch.Tensor | None) -> torch.Tensor:
        if self._object_sample_points_l is None or current_pose is None:
            return self._empty_tensor(0, 3)
        return self._pose_apply_t(current_pose, self._object_sample_points_l)

    def _compute_dilation_t(
        self,
        marker_points: torch.Tensor,
        marker_uv: torch.Tensor,
        marker_depth: torch.Tensor,
        marker_contact: torch.Tensor,
        ray_row_vec: torch.Tensor,
        ray_col_vec: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(torch.any(marker_contact).detach().cpu()):
            return torch.zeros_like(marker_points)

        source_depth = torch.where(marker_contact, marker_depth, torch.zeros_like(marker_depth))
        dvec = self._uv_metric_vectors_t(marker_uv, marker_uv, ray_row_vec, ray_col_vec)
        dist2 = torch.sum(dvec * dvec, dim=-1)
        weights = torch.exp(-float(self.cfg.lambda_dilate) * dist2)
        disp = float(self.cfg.dilate_scale) * torch.sum(source_depth[None, :, None] * dvec * weights[..., None], dim=1)
        return disp.to(dtype=torch.float32)

    def _compute_original_dilation_t(
        self,
        marker_points: torch.Tensor,
        marker_depth: torch.Tensor,
        marker_contact: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(torch.any(marker_contact).detach().cpu()):
            return torch.zeros_like(marker_points)

        source_depth = torch.where(marker_contact, marker_depth, torch.zeros_like(marker_depth))
        dvec = marker_points[:, None, :] - marker_points[None, :, :]
        dist2 = torch.sum(dvec * dvec, dim=-1)
        weights = torch.exp(-float(self.cfg.lambda_dilate) * dist2)
        disp = float(self.cfg.dilate_scale) * torch.sum(source_depth[None, :, None] * dvec * weights[..., None], dim=1)
        return disp.to(dtype=torch.float32)

    def _compute_shear_t(
        self,
        sensor_index: int,
        marker_points: torch.Tensor,
        marker_uv: torch.Tensor,
        marker_normals: torch.Tensor,
        marker_valid: torch.Tensor,
        marker_contact: torch.Tensor,
        depth_m: torch.Tensor,
        surface_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        object_points_w: torch.Tensor,
        object_valid: torch.Tensor,
        object_pose_wxyz: np.ndarray | torch.Tensor | None,
        surface_raw_m: torch.Tensor | None,
        surface_normals_w: torch.Tensor | None,
        ray_row_vec: torch.Tensor,
        ray_col_vec: torch.Tensor,
        ray_direction_w: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        empty_uv = self._empty_tensor(0, 2)
        empty_points = self._empty_tensor(0, 3)
        empty_shear = torch.zeros_like(marker_points)
        flat_object = object_points_w.reshape(-1, 3)
        flat_valid = object_valid.reshape(-1).bool()
        flat_depth = depth_m.reshape(-1)
        contact_threshold_m = float(self.cfg.contact_threshold_mm) * 1.0e-3
        active = flat_valid & (flat_depth > contact_threshold_m)
        active_count = int(torch.count_nonzero(active).detach().cpu())
        dense_contact = bool(torch.any(flat_depth > contact_threshold_m).detach().cpu())

        current_pose = self._valid_pose_t(object_pose_wxyz)

        if self._object_sample_points_l is not None and current_pose is not None:
            if not dense_contact:
                self._reset_hydrosoft_state(sensor_index)
                return empty_shear, empty_shear, empty_shear, 0, empty_uv, empty_points

            obj = self._pose_apply_t(current_pose, self._object_sample_points_l)
            sdf_result = self._estimate_indenter_sdf_from_link_surface_t(
                obj,
                depth_m,
                surface_points_w,
                surface_valid,
                object_points_w,
                object_valid,
                surface_raw_m,
                surface_normals_w,
                ray_direction_w=ray_direction_w,
            )
            if sdf_result is None:
                self._reset_hydrosoft_state(sensor_index)
                return empty_shear, empty_shear, empty_shear, 0, empty_uv, empty_points
            sdf, normals, _row_axes, _col_axes, sample_uv = sdf_result
            if not bool(torch.any(sdf < 0.0).detach().cpu()):
                self._reset_hydrosoft_state(sensor_index)
                return empty_shear, empty_shear, empty_shear, 0, empty_uv, empty_points

            local_frame = self._surface_motion_frame_t(surface_points_w, surface_valid, normals)
            if local_frame is not None:
                origin, basis = local_frame
                obj_local = self._points_to_frame_t(obj, origin, basis)
                normals_local = self._vectors_to_frame_t(normals, basis)
                fbar_local = self._step_hydrosoft_forces_t(sensor_index, sdf, obj_local, normals_local)
                fbar = self._vectors_from_frame_t(fbar_local, basis)
            else:
                fbar = self._step_hydrosoft_forces_t(sensor_index, sdf, obj, normals)
            shear, shear_visual, active_count = self._official_hydroshear_shear_t(
                marker_points,
                marker_uv,
                marker_normals,
                marker_valid,
                obj,
                sdf,
                fbar,
                normals,
                sample_uv,
                ray_row_vec,
                ray_col_vec,
            )
            _original_shear, original_shear_visual, _original_active_count = self._original_hydroshear_shear_t(
                marker_points,
                marker_valid,
                obj,
                sdf,
                fbar,
                normals,
            )
            contact = (sdf < 0.0) & torch.isfinite(sample_uv).all(dim=-1)
            return shear, shear_visual, original_shear_visual, active_count, sample_uv[contact], obj[contact]

        if not bool(torch.any(active).detach().cpu()) or not bool(torch.any(marker_valid).detach().cpu()):
            self._reset_hydrosoft_state(sensor_index)
            return empty_shear, empty_shear, empty_shear, active_count, empty_uv, empty_points

        surface_flat = surface_points_w.reshape(-1, 3)
        finite_object = torch.isfinite(flat_object).all(dim=-1)
        finite_surface = torch.isfinite(surface_flat).all(dim=-1)
        fallback_points = torch.where(finite_surface[:, None], surface_flat, torch.zeros_like(surface_flat))
        obj = torch.where((flat_valid & finite_object)[:, None], flat_object, fallback_points)

        sdf = torch.full((flat_depth.shape[0],), max(contact_threshold_m, 1.0e-9), dtype=torch.float32, device=self.device)
        sdf = torch.where(active, -flat_depth, sdf)

        normal_vectors = surface_flat - obj
        fallback_normal = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        if (
            ray_direction_w is not None
            and torch.isfinite(ray_direction_w).all()
            and torch.linalg.norm(ray_direction_w) > 1.0e-9
        ):
            ray_dir = self._safe_normalize_t(ray_direction_w, fallback_normal)
        else:
            ray_dir = self._safe_normalize_t(
                torch.mean(self._normalize_rows_t(normal_vectors[active], fallback=fallback_normal), dim=0),
                fallback_normal,
            )
        normals = ray_dir.reshape(1, 3).repeat(obj.shape[0], 1)

        local_frame = self._surface_motion_frame_t(surface_points_w, surface_valid, normals)
        if local_frame is not None:
            origin, basis = local_frame
            obj_local = self._points_to_frame_t(obj, origin, basis)
            normals_local = self._vectors_to_frame_t(normals, basis)
            fbar_local = self._step_hydrosoft_forces_t(sensor_index, sdf, obj_local, normals_local)
            fbar = self._vectors_from_frame_t(fbar_local, basis)
        else:
            fbar = self._step_hydrosoft_forces_t(sensor_index, sdf, obj, normals)
        grid_w = int(depth_m.shape[1])
        flat_ids = torch.arange(flat_depth.shape[0], dtype=torch.float32, device=self.device)
        sample_uv = torch.stack((torch.remainder(flat_ids, float(grid_w)), torch.floor(flat_ids / float(grid_w))), dim=-1)
        shear, shear_visual, active_count = self._official_hydroshear_shear_t(
            marker_points,
            marker_uv,
            marker_normals,
            marker_valid,
            obj,
            sdf,
            fbar,
            normals,
            sample_uv,
            ray_row_vec,
            ray_col_vec,
        )
        _original_shear, original_shear_visual, _original_active_count = self._original_hydroshear_shear_t(
            marker_points,
            marker_valid,
            obj,
            sdf,
            fbar,
            normals,
        )
        return shear, shear_visual, original_shear_visual, active_count, sample_uv[active], obj[active]

    def _estimate_indenter_sdf_from_link_surface_t(
        self,
        indenter_points_w: torch.Tensor,
        depth_m: torch.Tensor,
        surface_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        object_points_w: torch.Tensor,
        object_valid: torch.Tensor,
        surface_raw_m: torch.Tensor | None,
        surface_normals_w: torch.Tensor | None = None,
        *,
        ray_direction_w: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        flat_depth = depth_m.reshape(-1).to(dtype=torch.float32)
        flat_surface = surface_points_w.reshape(-1, 3).to(dtype=torch.float32)
        flat_object = object_points_w.reshape(-1, 3).to(dtype=torch.float32)
        flat_surface_valid = surface_valid.reshape(-1).bool()
        flat_object_valid = object_valid.reshape(-1).bool()
        contact_threshold_m = float(self.cfg.contact_threshold_mm) * 1.0e-3
        active = (
            flat_surface_valid
            & flat_object_valid
            & torch.isfinite(flat_depth)
            & (flat_depth > contact_threshold_m)
            & torch.isfinite(flat_surface).all(dim=-1)
            & torch.isfinite(flat_object).all(dim=-1)
        )
        if not bool(torch.any(active).detach().cpu()):
            return None

        if surface_raw_m is None:
            return None
        surface_raw = self._finite_tensor(surface_raw_m)
        if surface_raw.shape != surface_valid.shape:
            surface_raw = surface_raw.reshape(surface_valid.shape)
        active_surface = flat_surface[active]
        active_object = flat_object[active]
        fallback_normal = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        if (
            ray_direction_w is not None
            and torch.isfinite(ray_direction_w).all()
            and torch.linalg.norm(ray_direction_w) > 1.0e-9
        ):
            ray_dir = self._safe_normalize_t(ray_direction_w, fallback_normal)
        else:
            ray_dir = self._safe_normalize_t(
                torch.mean(self._normalize_rows_t(active_surface - active_object, fallback=fallback_normal), dim=0),
                fallback_normal,
            )
        frame = self._fit_ray_grid_frame_t(surface_points_w, surface_valid, surface_raw, ray_dir)
        if frame is None:
            return None
        origin, col_vec, row_vec, ray_dir = frame
        basis = torch.stack((col_vec, row_vec, ray_dir), dim=1).to(dtype=torch.float32)
        try:
            inv_basis = torch.linalg.inv(basis)
        except RuntimeError:
            return None
        if not torch.isfinite(inv_basis).all():
            return None

        points = indenter_points_w.reshape(-1, 3).to(dtype=torch.float32)
        sdf = torch.full((points.shape[0],), 1.0, dtype=torch.float32, device=self.device)
        sample_uv = torch.full((points.shape[0], 2), float("nan"), dtype=torch.float32, device=self.device)
        normals = ray_dir.reshape(1, 3).repeat(points.shape[0], 1).to(dtype=torch.float32)
        row_axis = self._safe_normalize_t(row_vec, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device))
        col_axis = self._safe_normalize_t(col_vec, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device))
        row_axes = row_axis.reshape(1, 3).repeat(points.shape[0], 1)
        col_axes = col_axis.reshape(1, 3).repeat(points.shape[0], 1)

        chunk_size = max(1, int(self.cfg.object_chunk_size))
        for start in range(0, points.shape[0], chunk_size):
            end = min(start + chunk_size, points.shape[0])
            points_chunk = points[start:end]
            coords = (points_chunk - origin.reshape(1, 3)) @ inv_basis.T
            x = coords[:, 0]
            y = coords[:, 1]
            point_depth = coords[:, 2]
            surface_depth, lookup_valid = self._bilinear_sample_t(surface_raw, surface_valid.bool(), x, y)
            inside = lookup_valid & torch.isfinite(point_depth) & (point_depth > 0.0)
            signed = point_depth - surface_depth
            sdf[start:end] = torch.where(inside, signed, torch.ones_like(signed))
            uv_chunk = torch.stack((x, y), dim=-1).to(dtype=torch.float32)
            sample_uv[start:end] = torch.where(inside[:, None], uv_chunk, sample_uv[start:end])

        return sdf, normals, row_axes, col_axes, sample_uv

    def _fit_ray_grid_frame_t(
        self,
        surface_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        surface_raw_m: torch.Tensor,
        ray_dir_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        valid = (
            surface_valid.bool()
            & torch.isfinite(surface_raw_m)
            & (surface_raw_m > 0.0)
            & torch.isfinite(surface_points_w).all(dim=-1)
        )
        if int(torch.count_nonzero(valid).detach().cpu()) < 6:
            return None
        ray_dir = self._safe_normalize_t(ray_dir_w, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device))
        ray_starts = surface_points_w - surface_raw_m[..., None] * ray_dir.reshape(1, 1, 3)
        ys, xs = torch.nonzero(valid, as_tuple=True)
        design = torch.stack(
            (
                torch.ones_like(xs, dtype=torch.float32, device=self.device),
                xs.to(dtype=torch.float32),
                ys.to(dtype=torch.float32),
            ),
            dim=1,
        )
        values = ray_starts[valid].to(dtype=torch.float32)
        try:
            coeff = torch.linalg.lstsq(design, values).solution
        except RuntimeError:
            coeff = torch.linalg.pinv(design) @ values
        origin = coeff[0].to(dtype=torch.float32)
        col_vec = coeff[1].to(dtype=torch.float32)
        row_vec = coeff[2].to(dtype=torch.float32)
        if (
            not torch.isfinite(coeff).all()
            or torch.linalg.norm(col_vec) < 1.0e-9
            or torch.linalg.norm(row_vec) < 1.0e-9
        ):
            return None
        return origin, col_vec, row_vec, ray_dir

    def _raycaster_metric_vectors_t(
        self,
        surface_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        surface_raw_m: torch.Tensor | None,
        surface_normals_w: torch.Tensor | None,
        *,
        ray_direction_w: torch.Tensor | None = None,
        fallback_row_axes: torch.Tensor | None = None,
        fallback_col_axes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fallback_row, fallback_col = self._global_grid_metric_vectors_t(
            surface_points_w,
            surface_valid,
            fallback_row_axes=fallback_row_axes,
            fallback_col_axes=fallback_col_axes,
        )

        ray_dir = None
        if ray_direction_w is not None:
            candidate = ray_direction_w.to(device=self.device, dtype=torch.float32).reshape(-1)
            if (
                candidate.shape[0] == 3
                and torch.isfinite(candidate).all()
                and torch.linalg.norm(candidate) > 1.0e-9
            ):
                ray_dir = self._safe_normalize_t(
                    candidate,
                    torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
                )
        if ray_dir is None and surface_normals_w is not None:
            normals = surface_normals_w.to(dtype=torch.float32)
            valid = (
                surface_valid.bool()
                & torch.isfinite(normals).all(dim=-1)
                & (torch.linalg.norm(normals, dim=-1) > 1.0e-9)
            )
            if bool(torch.any(valid).detach().cpu()):
                ray_dir = self._stable_normal_reference_t(normals[valid])
        if ray_dir is None:
            ray_dir = self._safe_normalize_t(
                torch.linalg.cross(fallback_col, fallback_row),
                torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
            )

        if surface_raw_m is not None:
            surface_raw = self._finite_tensor(surface_raw_m)
            if surface_raw.shape != surface_valid.shape:
                surface_raw = surface_raw.reshape(surface_valid.shape)
            frame = self._fit_ray_grid_frame_t(surface_points_w, surface_valid, surface_raw, ray_dir)
            if frame is not None:
                _origin, col_vec, row_vec, _ray_dir = frame
                return row_vec.to(dtype=torch.float32), col_vec.to(dtype=torch.float32)

        return fallback_row.to(dtype=torch.float32), fallback_col.to(dtype=torch.float32)

    def _global_grid_metric_vectors_t(
        self,
        surface_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        *,
        fallback_row_axes: torch.Tensor | None = None,
        fallback_col_axes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fallback_row = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device)
        fallback_col = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        if fallback_row_axes is not None and fallback_row_axes.numel():
            fallback_row = self._safe_normalize_t(torch.mean(fallback_row_axes, dim=0), fallback_row)
        if fallback_col_axes is not None and fallback_col_axes.numel():
            fallback_col = self._safe_normalize_t(torch.mean(fallback_col_axes, dim=0), fallback_col)

        points = surface_points_w.to(dtype=torch.float32)
        valid = surface_valid.bool() & torch.isfinite(points).all(dim=-1)
        col_deltas = points[:, 1:, :] - points[:, :-1, :]
        col_valid = valid[:, 1:] & valid[:, :-1] & torch.isfinite(col_deltas).all(dim=-1)
        row_deltas = points[1:, :, :] - points[:-1, :, :]
        row_valid = valid[1:, :] & valid[:-1, :] & torch.isfinite(row_deltas).all(dim=-1)

        col_vec = fallback_col
        if bool(torch.any(col_valid).detach().cpu()):
            col_vec = torch.mean(col_deltas[col_valid], dim=0).to(dtype=torch.float32)

        row_vec = fallback_row
        if bool(torch.any(row_valid).detach().cpu()):
            row_vec = torch.mean(row_deltas[row_valid], dim=0).to(dtype=torch.float32)

        return row_vec.to(dtype=torch.float32), col_vec.to(dtype=torch.float32)

    def _orthonormalize_axes_t(
        self,
        row_vec: torch.Tensor,
        col_vec: torch.Tensor,
        fallback_row: torch.Tensor,
        fallback_col: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        col_axis = self._safe_normalize_t(col_vec, fallback_col)
        row_vec = row_vec.to(dtype=torch.float32)
        row_vec = row_vec - col_axis * torch.sum(row_vec * col_axis)
        row_axis = self._safe_normalize_t(row_vec, fallback_row)
        return row_axis.to(dtype=torch.float32), col_axis.to(dtype=torch.float32)

    @staticmethod
    def _uv_metric_vectors_t(
        target_uv: torch.Tensor,
        source_uv: torch.Tensor,
        row_vec: torch.Tensor,
        col_vec: torch.Tensor,
    ) -> torch.Tensor:
        target_uv = target_uv.to(dtype=torch.float32).reshape(-1, 2)
        source_uv = source_uv.to(dtype=torch.float32).reshape(-1, 2)
        du = target_uv[:, None, 0] - source_uv[None, :, 0]
        dv = target_uv[:, None, 1] - source_uv[None, :, 1]
        return du[..., None] * col_vec.reshape(1, 1, 3) + dv[..., None] * row_vec.reshape(1, 1, 3)

    def _vector_to_uv_delta_t(self, vectors: torch.Tensor, row_vec: torch.Tensor, col_vec: torch.Tensor) -> torch.Tensor:
        vectors = vectors.to(dtype=torch.float32).reshape(-1, 3)
        basis = torch.stack((col_vec, row_vec), dim=1).to(dtype=torch.float32)
        if (
            not torch.isfinite(basis).all()
            or torch.linalg.norm(torch.linalg.cross(col_vec, row_vec)) < 1.0e-12
        ):
            return self._empty_tensor(vectors.shape[0], 2)
        try:
            return (vectors @ torch.linalg.pinv(basis).T).to(dtype=torch.float32)
        except RuntimeError:
            return self._empty_tensor(vectors.shape[0], 2)

    def _prepare_marker_normals_t(
        self,
        marker_normals: torch.Tensor,
        marker_points: torch.Tensor,
        contact_normals: torch.Tensor,
    ) -> torch.Tensor:
        fallback = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        if contact_normals.numel():
            fallback = self._safe_normalize_t(torch.mean(contact_normals, dim=0), fallback)
        normals = marker_normals.to(dtype=torch.float32).reshape(-1, 3)
        if normals.shape[0] != marker_points.shape[0]:
            normals = fallback.reshape(1, 3).repeat(marker_points.shape[0], 1)
        return self._normalize_rows_t(normals, fallback=fallback).to(dtype=torch.float32)

    def _project_to_marker_tangent_t(self, vectors: torch.Tensor, marker_normals: torch.Tensor) -> torch.Tensor:
        vectors = vectors.to(dtype=torch.float32).reshape(-1, 3)
        marker_normals = self._normalize_rows_t(
            marker_normals,
            fallback=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
        )
        expanded = vectors.reshape(1, -1, 3)
        normals = marker_normals.reshape(-1, 1, 3)
        normal_component = torch.sum(expanded * normals, dim=-1, keepdim=True) * normals
        return (expanded - normal_component).to(dtype=torch.float32)

    def _step_hydrosoft_forces_t(
        self,
        sensor_index: int,
        sdf: torch.Tensor,
        indenter_points: torch.Tensor,
        normals: torch.Tensor,
    ) -> torch.Tensor:
        sdf = self._finite_tensor(sdf).reshape(-1)
        indenter_points = self._finite_tensor(indenter_points).reshape(-1, 3)
        normals = self._normalize_rows_t(
            normals,
            fallback=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
        )

        prev_sdf = self._prev_sdf[sensor_index]
        prev_points = self._prev_indenter_points[sensor_index]
        prev_normals = self._prev_normals[sensor_index]
        hydrosoft_forces = self._hydrosoft_forces[sensor_index]
        if (
            prev_sdf is None
            or prev_points is None
            or prev_normals is None
            or hydrosoft_forces is None
            or prev_sdf.shape != sdf.shape
            or prev_points.shape != indenter_points.shape
            or prev_normals.shape != normals.shape
            or hydrosoft_forces.shape != indenter_points.shape
            or prev_sdf.device != self.device
        ):
            prev_sdf = sdf.clone()
            prev_points = indenter_points.clone()
            prev_normals = normals.clone()
            hydrosoft_forces = torch.zeros_like(indenter_points)

        prev_penetration = torch.relu(-prev_sdf)
        penetration = torch.relu(-sdf)
        denom = prev_sdf - sdf
        alpha_d = -((prev_penetration - penetration) / denom)
        alpha_fallback = -0.5 * (torch.sign(sdf) - 1.0)
        alpha_d = torch.where(torch.isfinite(alpha_d), alpha_d, alpha_fallback)
        displacement = alpha_d[:, None] * (prev_points - indenter_points)

        prev_normals = self._normalize_rows_t(
            prev_normals,
            fallback=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
        )
        prev_force_n = torch.sum(hydrosoft_forces * prev_normals, dim=-1)
        prev_force_t = hydrosoft_forces - prev_force_n[:, None] * prev_normals
        prev_force_t = prev_force_t - torch.sum(prev_force_t * normals, dim=-1)[:, None] * normals
        displacement_n = torch.sum(displacement * normals, dim=-1)
        displacement_t = displacement - displacement_n[:, None] * normals

        sample_area = float(self.cfg.hydrosoft_area) * float(self._object_sample_area_scale)
        fn = prev_force_n + float(self.cfg.hydrosoft_e) * sample_area * displacement_n
        ft = prev_force_t + float(self.cfg.hydrosoft_k) * sample_area * displacement_t
        fn_bar = torch.relu(fn)
        norm_ft = torch.linalg.norm(ft, dim=-1)
        ft_limit = torch.minimum(float(self.cfg.hydrosoft_mu) * fn_bar, norm_ft)
        ft_bar = ft_limit[:, None] * ft / (norm_ft[:, None] + 1.0e-8)
        fbar = ft_bar + fn_bar[:, None] * normals
        fbar = torch.where((sdf < 0.0)[:, None], fbar, torch.zeros_like(fbar)).to(dtype=torch.float32)

        self._hydrosoft_forces[sensor_index] = fbar.detach().clone()
        self._prev_indenter_points[sensor_index] = indenter_points.detach().clone()
        self._prev_sdf[sensor_index] = sdf.detach().clone()
        self._prev_normals[sensor_index] = normals.detach().clone()
        return fbar

    def _surface_motion_frame_t(
        self,
        surface_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        normals_w: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        points = surface_points_w.reshape(-1, 3).to(dtype=torch.float32)
        valid = surface_valid.reshape(-1).bool() & torch.isfinite(points).all(dim=-1)
        if int(torch.count_nonzero(valid).detach().cpu()) < 6:
            return None

        origin = torch.mean(points[valid], dim=0).to(dtype=torch.float32)
        grid = surface_points_w.to(dtype=torch.float32)
        mask = surface_valid.bool() & torch.isfinite(grid).all(dim=-1)
        col_deltas = grid[:, 1:, :] - grid[:, :-1, :]
        col_valid = mask[:, 1:] & mask[:, :-1] & torch.isfinite(col_deltas).all(dim=-1)
        row_deltas = grid[1:, :, :] - grid[:-1, :, :]
        row_valid = mask[1:, :] & mask[:-1, :] & torch.isfinite(row_deltas).all(dim=-1)
        if not bool(torch.any(col_valid).detach().cpu()) or not bool(torch.any(row_valid).detach().cpu()):
            return None

        col_axis = self._safe_normalize_t(
            torch.mean(col_deltas[col_valid], dim=0),
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device),
        )
        row_axis = self._safe_normalize_t(
            torch.mean(row_deltas[row_valid], dim=0),
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device),
        )
        normal_axis = self._safe_normalize_t(
            torch.linalg.cross(col_axis, row_axis),
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
        )
        if normals_w is not None and normals_w.numel():
            normal_ref = self._stable_normal_reference_t(normals_w.reshape(-1, 3).to(dtype=torch.float32))
            if float(torch.sum(normal_axis * normal_ref).detach().cpu()) < 0.0:
                normal_axis = -normal_axis

        col_axis = col_axis - normal_axis * torch.sum(col_axis * normal_axis)
        col_axis = self._safe_normalize_t(
            col_axis,
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device),
        )
        row_axis = row_axis - normal_axis * torch.sum(row_axis * normal_axis) - col_axis * torch.sum(row_axis * col_axis)
        row_axis = self._safe_normalize_t(row_axis, torch.linalg.cross(normal_axis, col_axis))
        basis = torch.stack((col_axis, row_axis, normal_axis), dim=1).to(dtype=torch.float32)
        return origin, basis

    @staticmethod
    def _points_to_frame_t(points_w: torch.Tensor, origin_w: torch.Tensor, basis_w: torch.Tensor) -> torch.Tensor:
        points = points_w.to(dtype=torch.float32).reshape(-1, 3)
        return ((points - origin_w.reshape(1, 3)) @ basis_w).to(dtype=torch.float32)

    @staticmethod
    def _vectors_to_frame_t(vectors_w: torch.Tensor, basis_w: torch.Tensor) -> torch.Tensor:
        vectors = vectors_w.to(dtype=torch.float32).reshape(-1, 3)
        return (vectors @ basis_w).to(dtype=torch.float32)

    @staticmethod
    def _vectors_from_frame_t(vectors_l: torch.Tensor, basis_w: torch.Tensor) -> torch.Tensor:
        vectors = vectors_l.to(dtype=torch.float32).reshape(-1, 3)
        return (vectors @ basis_w.T).to(dtype=torch.float32)

    def _official_hydroshear_shear_t(
        self,
        marker_points: torch.Tensor,
        marker_uv: torch.Tensor,
        marker_normals: torch.Tensor,
        marker_valid: torch.Tensor,
        obj: torch.Tensor,
        sdf: torch.Tensor,
        fbar: torch.Tensor,
        normals: torch.Tensor,
        sample_uv: torch.Tensor,
        ray_row_vec: torch.Tensor,
        ray_col_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        sample_uv = sample_uv.to(dtype=torch.float32).reshape(-1, 2)
        if sample_uv.shape[0] != obj.shape[0]:
            empty = torch.zeros_like(marker_points)
            return empty, empty, 0
        contact = (
            (sdf < 0.0)
            & torch.isfinite(sdf)
            & torch.isfinite(obj).all(dim=-1)
            & torch.isfinite(fbar).all(dim=-1)
            & torch.isfinite(sample_uv).all(dim=-1)
        )
        active_count = int(torch.count_nonzero(contact).detach().cpu())
        if active_count == 0 or not bool(torch.any(marker_valid).detach().cpu()):
            empty = torch.zeros_like(marker_points)
            return empty, empty, active_count

        fbar_contact = fbar[contact]
        sample_uv_contact = sample_uv[contact]
        normal_contact = self._normalize_rows_t(
            normals[contact],
            fallback=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
        )
        marker_normals = self._prepare_marker_normals_t(marker_normals, marker_points, normal_contact)
        normal_force = torch.sum(fbar_contact * normal_contact, dim=-1)
        h = torch.clamp(normal_force, min=0.0).to(dtype=torch.float32)
        fbar_tangent = fbar_contact - normal_force[:, None] * normal_contact
        fbar_uv_delta = self._vector_to_uv_delta_t(fbar_tangent, ray_row_vec, ray_col_vec)
        affected_marker_uv = sample_uv_contact + fbar_uv_delta
        out = torch.zeros_like(marker_points)
        out_visual = torch.zeros_like(marker_points)
        chunk_size = max(1, int(self.cfg.object_chunk_size))
        for start in range(0, affected_marker_uv.shape[0], chunk_size):
            end = min(start + chunk_size, affected_marker_uv.shape[0])
            affected_uv_chunk = affected_marker_uv[start:end]
            transported_chunk = self._project_to_marker_tangent_t(fbar_tangent[start:end], marker_normals)
            h_chunk = h[start:end]
            dvec = self._uv_metric_vectors_t(marker_uv, affected_uv_chunk, ray_row_vec, ray_col_vec)
            dist2 = torch.sum(dvec * dvec, dim=-1)
            weights = torch.exp(-float(self.cfg.lambda_shear) * dist2).to(dtype=torch.float32)
            contribution = torch.sum(h_chunk[None, :, None] * -transported_chunk * weights[..., None], dim=1)
            out = out + contribution
            out_visual = out_visual + contribution

        out = float(self.cfg.shear_scale) * out
        out_visual = float(self.cfg.shear_scale) * out_visual
        return (
            torch.where(marker_valid[:, None], out, torch.zeros_like(out)).to(dtype=torch.float32),
            torch.where(marker_valid[:, None], out_visual, torch.zeros_like(out_visual)).to(dtype=torch.float32),
            active_count,
        )

    def _original_hydroshear_shear_t(
        self,
        marker_points: torch.Tensor,
        marker_valid: torch.Tensor,
        obj: torch.Tensor,
        sdf: torch.Tensor,
        fbar: torch.Tensor,
        normals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        contact = (sdf < 0.0) & torch.isfinite(sdf) & torch.isfinite(obj).all(dim=-1) & torch.isfinite(fbar).all(dim=-1)
        active_count = int(torch.count_nonzero(contact).detach().cpu())
        if active_count == 0 or not bool(torch.any(marker_valid).detach().cpu()):
            empty = torch.zeros_like(marker_points)
            return empty, empty, active_count

        obj_contact = obj[contact]
        fbar_contact = fbar[contact]
        normal_contact = self._normalize_rows_t(
            normals[contact],
            fallback=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
        )
        normal_force = torch.sum(fbar_contact * normal_contact, dim=-1)
        affected_marker_positions = obj_contact + fbar_contact
        h = torch.clamp(normal_force, min=0.0).to(dtype=torch.float32)
        fbar_tangent = fbar_contact - normal_force[:, None] * normal_contact
        out = torch.zeros_like(marker_points)
        out_visual = torch.zeros_like(marker_points)
        chunk_size = max(1, int(self.cfg.object_chunk_size))
        for start in range(0, affected_marker_positions.shape[0], chunk_size):
            end = min(start + chunk_size, affected_marker_positions.shape[0])
            affected_chunk = affected_marker_positions[start:end]
            fbar_chunk = fbar_contact[start:end]
            fbar_visual_chunk = fbar_tangent[start:end]
            h_chunk = h[start:end]
            dvec = marker_points[:, None, :] - affected_chunk[None, :, :]
            dist2 = torch.sum(dvec * dvec, dim=-1)
            weights = torch.exp(-float(self.cfg.lambda_shear) * dist2).to(dtype=torch.float32)
            out = out + torch.sum(h_chunk[None, :, None] * -fbar_chunk[None, :, :] * weights[..., None], dim=1)
            out_visual = out_visual + torch.sum(
                h_chunk[None, :, None] * -fbar_visual_chunk[None, :, :] * weights[..., None],
                dim=1,
            )

        out = float(self.cfg.shear_scale) * out
        out_visual = float(self.cfg.shear_scale) * out_visual
        return (
            torch.where(marker_valid[:, None], out, torch.zeros_like(out)).to(dtype=torch.float32),
            torch.where(marker_valid[:, None], out_visual, torch.zeros_like(out_visual)).to(dtype=torch.float32),
            active_count,
        )

    def _marker_tangent_axes_t(
        self,
        surface_points_w: torch.Tensor,
        surface_valid: torch.Tensor,
        xs: torch.Tensor,
        ys: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = surface_valid.shape
        row_axes = []
        col_axes = []
        fallback_col = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        fallback_row = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device)

        for x_t, y_t in zip(xs, ys):
            x = int(x_t.detach().cpu())
            y = int(y_t.detach().cpu())
            x0 = max(0, x - 1)
            x1 = min(int(w) - 1, x + 1)
            y0 = max(0, y - 1)
            y1 = min(int(h) - 1, y + 1)

            col = surface_points_w[y, x1] - surface_points_w[y, x0]
            row = surface_points_w[y1, x] - surface_points_w[y0, x]
            if (
                not bool((surface_valid[y, x0] & surface_valid[y, x1]).detach().cpu())
                or torch.linalg.norm(col) < 1.0e-9
            ):
                col = fallback_col.clone()
            if (
                not bool((surface_valid[y0, x] & surface_valid[y1, x]).detach().cpu())
                or torch.linalg.norm(row) < 1.0e-9
            ):
                row = fallback_row.clone()

            col = self._safe_normalize_t(col, fallback_col)
            row = row - col * torch.sum(row * col)
            row = self._safe_normalize_t(row, fallback_row)
            row_axes.append(row)
            col_axes.append(col)

        return torch.stack(row_axes, dim=0).to(dtype=torch.float32), torch.stack(col_axes, dim=0).to(dtype=torch.float32)

    def _marker_normals_t(
        self,
        surface_normals_w: torch.Tensor | None,
        xs: torch.Tensor,
        ys: torch.Tensor,
        row_axes_w: torch.Tensor,
        col_axes_w: torch.Tensor,
    ) -> torch.Tensor:
        fallback = torch.linalg.cross(col_axes_w, row_axes_w)
        fallback = self._normalize_rows_t(
            fallback,
            fallback=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
        )
        reference = self._stable_normal_reference_t(fallback)
        if surface_normals_w is None:
            orient = torch.sum(fallback * reference.reshape(1, 3), dim=-1)
            return torch.where((orient < 0.0)[:, None], -fallback, fallback).to(dtype=torch.float32)

        normals_grid = surface_normals_w.to(dtype=torch.float32)
        samples = normals_grid[ys.to(torch.long), xs.to(torch.long)]
        sample_valid = torch.isfinite(samples).all(dim=-1) & (torch.linalg.norm(samples, dim=-1) > 1.0e-9)
        samples = self._normalize_rows_t(
            samples,
            fallback=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
        )
        normals = torch.where(sample_valid[:, None], samples, fallback)
        orient = torch.sum(normals * reference.reshape(1, 3), dim=-1)
        normals = torch.where((orient < 0.0)[:, None], -normals, normals)
        return normals.to(dtype=torch.float32)

    def _stable_normal_reference_t(self, normals: torch.Tensor) -> torch.Tensor:
        normals = self._normalize_rows_t(
            normals,
            fallback=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
        )
        valid = torch.isfinite(normals).all(dim=-1) & (torch.linalg.norm(normals, dim=-1) > 1.0e-9)
        if not bool(torch.any(valid).detach().cpu()):
            return torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        candidates = normals[valid]
        reference = self._safe_normalize_t(torch.mean(candidates, dim=0), candidates[0])
        for _ in range(2):
            orient = torch.sum(candidates * reference.reshape(1, 3), dim=-1)
            oriented = torch.where((orient < 0.0)[:, None], -candidates, candidates)
            reference = self._safe_normalize_t(torch.mean(oriented, dim=0), reference)
        return reference.to(dtype=torch.float32)

    def _scaled_marker_uv_t(self, width: int, height: int) -> torch.Tensor:
        if width == self.cfg.width and height == self.cfg.height:
            return self._marker_uv_t.clone()
        scale = torch.tensor(
            [
                float(width) / max(1.0, float(self.cfg.width)),
                float(height) / max(1.0, float(self.cfg.height)),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return self._marker_uv_t * scale

    def _safe_normalize_t(self, vec: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        vec = vec.to(device=self.device, dtype=torch.float32)
        fallback = fallback.to(device=self.device, dtype=torch.float32)
        norm = torch.linalg.norm(vec)
        if (not bool(torch.isfinite(norm).detach().cpu())) or float(norm.detach().cpu()) < 1.0e-9:
            return fallback.to(dtype=torch.float32)
        return (vec / norm).to(dtype=torch.float32)

    def _normalize_rows_t(self, vectors: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        vectors = vectors.to(device=self.device, dtype=torch.float32)
        fallback = fallback.to(device=self.device, dtype=torch.float32)
        if vectors.numel() == 0:
            return vectors.reshape(-1, 3).to(dtype=torch.float32)
        norms = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        valid = torch.isfinite(norms) & (norms > 1.0e-9)
        normalized = vectors / torch.clamp(norms, min=1.0e-9)
        return torch.where(valid, normalized, fallback.reshape(1, 3)).to(dtype=torch.float32)

    def _bilinear_sample_t(
        self,
        values: torch.Tensor,
        valid: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = values.to(dtype=torch.float32)
        valid = valid.bool()
        x = x.to(dtype=torch.float32)
        y = y.to(dtype=torch.float32)
        h, w = values.shape
        inside = (
            torch.isfinite(x)
            & torch.isfinite(y)
            & (x >= 0.0)
            & (y >= 0.0)
            & (x <= float(w - 1))
            & (y <= float(h - 1))
        )
        x0 = torch.floor(torch.clamp(x, 0.0, float(w - 1))).to(torch.long)
        y0 = torch.floor(torch.clamp(y, 0.0, float(h - 1))).to(torch.long)
        x1 = torch.minimum(x0 + 1, torch.full_like(x0, int(w) - 1))
        y1 = torch.minimum(y0 + 1, torch.full_like(y0, int(h) - 1))
        wx = torch.clamp(x - x0.to(dtype=torch.float32), 0.0, 1.0)
        wy = torch.clamp(y - y0.to(dtype=torch.float32), 0.0, 1.0)

        v00 = values[y0, x0]
        v10 = values[y0, x1]
        v01 = values[y1, x0]
        v11 = values[y1, x1]
        w00 = (1.0 - wx) * (1.0 - wy)
        w10 = wx * (1.0 - wy)
        w01 = (1.0 - wx) * wy
        w11 = wx * wy
        m00 = valid[y0, x0] & torch.isfinite(v00) & (v00 > 0.0)
        m10 = valid[y0, x1] & torch.isfinite(v10) & (v10 > 0.0)
        m01 = valid[y1, x0] & torch.isfinite(v01) & (v01 > 0.0)
        m11 = valid[y1, x1] & torch.isfinite(v11) & (v11 > 0.0)
        m00f = m00.to(dtype=torch.float32)
        m10f = m10.to(dtype=torch.float32)
        m01f = m01.to(dtype=torch.float32)
        m11f = m11.to(dtype=torch.float32)
        weight_sum = w00 * m00f + w10 * m10f + w01 * m01f + w11 * m11f
        sampled = (w00 * m00f * v00 + w10 * m10f * v10 + w01 * m01f * v01 + w11 * m11f * v11) / torch.clamp(
            weight_sum, min=1.0e-12
        )
        ok = inside & (weight_sum > 1.0e-6)
        return sampled.to(dtype=torch.float32), ok

    def _valid_pose_t(self, pose: np.ndarray | torch.Tensor | None) -> torch.Tensor | None:
        if pose is None:
            return None
        pose_t = self._to_tensor(pose).reshape(-1)
        if pose_t.numel() < 7:
            return None
        pose_t = pose_t[:7].clone()
        if not bool(torch.isfinite(pose_t).all().detach().cpu()):
            return None
        quat = pose_t[3:7]
        quat_norm = torch.linalg.norm(quat)
        if float(quat_norm.detach().cpu()) < 1.0e-8:
            return None
        pose_t[3:7] = quat / quat_norm
        return pose_t

    def _pose_for_sensor_t(self, pose: np.ndarray | torch.Tensor | None, sensor_index: int) -> torch.Tensor | None:
        if pose is None:
            return None
        pose_t = self._to_tensor(pose)
        if pose_t.ndim >= 2 and pose_t.shape[-1] >= 7:
            pose_t = pose_t.reshape(-1, pose_t.shape[-1])
            if pose_t.shape[0] == 0:
                return None
            return self._valid_pose_t(pose_t[min(max(int(sensor_index), 0), pose_t.shape[0] - 1), :7])
        return self._valid_pose_t(pose_t)

    def _pose_apply_t(self, pose: torch.Tensor, points_l: torch.Tensor) -> torch.Tensor:
        pose = pose.to(device=self.device, dtype=torch.float32)
        points_l = points_l.to(device=self.device, dtype=torch.float32)
        return (self._quat_apply_t(pose[3:7], points_l) + pose[:3][None, :]).to(dtype=torch.float32)

    @staticmethod
    def _quat_apply_t(quat: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        quat = quat.to(dtype=torch.float32)
        vectors = vectors.to(dtype=torch.float32)
        qvec = quat[1:4]
        qvec_expanded = qvec.reshape(1, 3).expand_as(vectors)
        uv = torch.linalg.cross(qvec_expanded, vectors)
        uuv = torch.linalg.cross(qvec_expanded, uv)
        return vectors + 2.0 * (quat[0] * uv + uuv)


    def _reset_hydrosoft_state(self, sensor_index: int) -> None:
        if 0 <= int(sensor_index) < len(self._hydrosoft_forces):
            self._hydrosoft_forces[sensor_index] = None
            self._prev_sdf[sensor_index] = None
            self._prev_indenter_points[sensor_index] = None
            self._prev_normals[sensor_index] = None
        if self._batch_state_valid is not None and 0 <= int(sensor_index) < int(self._batch_state_valid.shape[0]):
            self._batch_state_valid[sensor_index] = False
            if self._batch_prev_sample_ids is not None:
                self._batch_prev_sample_ids[sensor_index].fill_(-1)
            if self._batch_hydrosoft_forces is not None:
                self._batch_hydrosoft_forces[sensor_index].zero_()
            if self._batch_prev_sdf is not None:
                self._batch_prev_sdf[sensor_index].zero_()
            if self._batch_prev_indenter_points is not None:
                self._batch_prev_indenter_points[sensor_index].zero_()
            if self._batch_prev_normals is not None:
                self._batch_prev_normals[sensor_index].zero_()

    def _flow_from_displacement(
        self,
        marker_uv: np.ndarray,
        displacement_w: np.ndarray,
        row_axes_w: np.ndarray,
        col_axes_w: np.ndarray,
    ) -> np.ndarray:
        du = np.sum(displacement_w * col_axes_w, axis=-1) * float(self.cfg.arrow_scale_px_per_m)
        dv = np.sum(displacement_w * row_axes_w, axis=-1) * float(self.cfg.arrow_scale_px_per_m)
        delta_uv = np.stack((du, dv), axis=-1).astype(np.float32)
        flow = np.stack((marker_uv, marker_uv + delta_uv), axis=0).astype(np.float32)
        return flow

    def _flow_from_displacement_t(
        self,
        marker_uv: torch.Tensor,
        displacement_w: torch.Tensor,
        row_axes_w: torch.Tensor,
        col_axes_w: torch.Tensor,
    ) -> torch.Tensor:
        du = torch.sum(displacement_w * col_axes_w, dim=-1) * float(self.cfg.arrow_scale_px_per_m)
        dv = torch.sum(displacement_w * row_axes_w, dim=-1) * float(self.cfg.arrow_scale_px_per_m)
        delta_uv = torch.stack((du, dv), dim=-1).to(dtype=torch.float32)
        return torch.stack((marker_uv.to(dtype=torch.float32), marker_uv.to(dtype=torch.float32) + delta_uv), dim=0)

    def _render_marker_image(
        self,
        marker_flow: np.ndarray,
        depth_m: np.ndarray,
        marker_uv: np.ndarray,
        marker_depth_m: np.ndarray,
    ) -> np.ndarray:
        if bool(self.cfg.show_depth_background) and float(np.max(depth_m)) > 0.0:
            img = self._depth_background(depth_m)
        else:
            img = np.full((int(self.cfg.height), int(self.cfg.width), 3), 255, dtype=np.uint8)

        marker_depth_norm = marker_depth_m / (float(self.cfg.depth_background_max_mm) * 1.0e-3 + 1.0e-12)
        for uv, norm in zip(marker_uv, marker_depth_norm):
            radius = int(round(float(self.cfg.marker_radius) + 2.0 * float(np.clip(norm, 0.0, 1.0))))
            _draw_circle(img, uv, radius=max(1, radius), color=_MARKER_GREEN)

        for start, end in zip(marker_flow[0], marker_flow[1]):
            delta = end - start
            length = float(np.linalg.norm(delta))
            if length < 1.0e-6:
                continue
            min_len = max(0.0, float(self.cfg.min_arrow_length_px))
            if 0.0 < length < min_len:
                end = start + delta / length * min_len
            _draw_arrow(
                img,
                start,
                end,
                color=_MARKER_GREEN,
                thickness=int(self.cfg.arrow_thickness),
                tip_length=float(self.cfg.arrow_tip_length),
            )
        return img

    def _render_vector_field_image(
        self,
        marker_flow: np.ndarray,
        *,
        color: tuple[int, int, int],
        draw_points: bool,
        object_sample_uv: np.ndarray | None = None,
        padding_px: int = 0,
    ) -> np.ndarray:
        padding = max(0, int(padding_px))
        offset = np.asarray((padding, padding), dtype=np.float32)
        img = np.zeros(
            (int(self.cfg.height) + 2 * padding, int(self.cfg.width) + 2 * padding, 3),
            dtype=np.uint8,
        )
        if object_sample_uv is not None:
            sample_uv = np.asarray(object_sample_uv, dtype=np.float32).reshape(-1, 2)
            finite = np.isfinite(sample_uv).all(axis=-1)
            for uv in sample_uv[finite]:
                _draw_circle(img, uv + offset, radius=1, color=(255, 0, 0))
        for start, end in zip(marker_flow[0], marker_flow[1]):
            start = start + offset
            end = end + offset
            delta = end - start
            length = float(np.linalg.norm(delta))
            if length < 1.0e-6:
                continue
            min_len = max(0.0, float(self.cfg.min_arrow_length_px))
            if 0.0 < length < min_len:
                end = start + delta / length * min_len
            _draw_arrow(
                img,
                start,
                end,
                color=color,
                thickness=int(self.cfg.arrow_thickness),
                tip_length=float(self.cfg.arrow_tip_length),
            )
        if draw_points:
            for start in marker_flow[0]:
                _draw_circle(
                    img,
                    start + offset,
                    radius=max(1, int(self.cfg.marker_radius) - 1),
                    color=color,
                )
        return img

    def _render_marker_height_image(self, marker_depth_m: np.ndarray) -> np.ndarray:
        rows = max(1, int(self.cfg.marker_rows))
        cols = max(1, int(self.cfg.marker_cols))
        depth = np.nan_to_num(np.asarray(marker_depth_m, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if depth.size != rows * cols:
            depth_img = depth.reshape(1, -1) if depth.size else np.zeros((1, 1), dtype=np.float32)
        else:
            depth_img = depth.reshape(rows, cols)
        max_depth = float(np.max(depth_img)) if depth_img.size else 0.0
        if max_depth > 1.0e-12:
            norm = np.clip(depth_img / max_depth, 0.0, 1.0)
        else:
            norm = np.zeros_like(depth_img, dtype=np.float32)
        gray_small = (norm * 255.0).astype(np.uint8)
        row_repeat = max(1, int(np.ceil(float(self.cfg.height) / max(1, gray_small.shape[0]))))
        col_repeat = max(1, int(np.ceil(float(self.cfg.width) / max(1, gray_small.shape[1]))))
        gray = np.repeat(np.repeat(gray_small, row_repeat, axis=0), col_repeat, axis=1)
        gray = gray[: int(self.cfg.height), : int(self.cfg.width)]
        if gray.shape[0] < int(self.cfg.height) or gray.shape[1] < int(self.cfg.width):
            padded = np.zeros((int(self.cfg.height), int(self.cfg.width)), dtype=np.uint8)
            padded[: gray.shape[0], : gray.shape[1]] = gray
            gray = padded
        return np.stack((gray, gray, gray), axis=-1)

    def _depth_background(self, depth_m: np.ndarray) -> np.ndarray:
        denom = float(self.cfg.depth_background_max_mm) * 1.0e-3
        if denom <= 0.0:
            denom = float(np.max(depth_m)) + 1.0e-12
        norm = np.clip(depth_m / (denom + 1.0e-12), 0.0, 1.0)
        gray = (245 - 110 * norm).astype(np.uint8)
        blue = np.clip(gray.astype(np.int16) + 8, 0, 255).astype(np.uint8)
        return np.stack((gray, gray, blue), axis=-1)

    def _scaled_marker_uv(self, width: int, height: int) -> np.ndarray:
        if width == self.cfg.width and height == self.cfg.height:
            return self._marker_uv.copy()
        scale = np.array(
            [
                float(width) / max(1.0, float(self.cfg.width)),
                float(height) / max(1.0, float(self.cfg.height)),
            ],
            dtype=np.float32,
        )
        return self._marker_uv * scale

    @staticmethod
    def _make_marker_uv(cfg: RevoCurvedHydroShearCfg) -> np.ndarray:
        if cfg.marker_uv is not None:
            marker_uv = np.asarray(cfg.marker_uv, dtype=np.float32).reshape(-1, 2)
            expected = max(0, int(cfg.marker_rows)) * max(0, int(cfg.marker_cols))
            if len(marker_uv) != expected:
                raise ValueError(f"Expected {expected} marker UV coordinates, got {len(marker_uv)}")
            if (
                not np.isfinite(marker_uv).all()
                or np.any(marker_uv[:, 0] < 0.0)
                or np.any(marker_uv[:, 0] >= float(cfg.width))
                or np.any(marker_uv[:, 1] < 0.0)
                or np.any(marker_uv[:, 1] >= float(cfg.height))
            ):
                raise ValueError("Custom marker UV coordinates must be finite and inside the render image")
            return marker_uv.copy()
        xs = np.linspace(float(cfg.marker_margin_x), float(cfg.width) - float(cfg.marker_margin_x), int(cfg.marker_cols))
        ys = np.linspace(float(cfg.marker_margin_y), float(cfg.height) - float(cfg.marker_margin_y), int(cfg.marker_rows))
        grid_x, grid_y = np.meshgrid(xs, ys)
        return np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2).astype(np.float32)


def _draw_circle(img: np.ndarray, center: np.ndarray, *, radius: int, color: tuple[int, int, int]) -> None:
    try:
        import cv2

        pt = (int(round(float(center[0]))), int(round(float(center[1]))))
        cv2.circle(img, pt, max(1, int(radius)), color, thickness=-1, lineType=cv2.LINE_AA)
        return
    except Exception:
        pass

    x = int(round(float(center[0])))
    y = int(round(float(center[1])))
    r = max(1, int(radius))
    y0 = max(0, y - r)
    y1 = min(img.shape[0], y + r + 1)
    x0 = max(0, x - r)
    x1 = min(img.shape[1], x + r + 1)
    img[y0:y1, x0:x1] = color


def _draw_line(
    img: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    x0, y0 = int(round(float(start[0]))), int(round(float(start[1])))
    x1, y1 = int(round(float(end[0]))), int(round(float(end[1])))
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, steps + 1).round().astype(int)
    ys = np.linspace(y0, y1, steps + 1).round().astype(int)
    radius = max(0, int(thickness) // 2)
    for x, y in zip(xs, ys):
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            y0p = max(0, y - radius)
            y1p = min(img.shape[0], y + radius + 1)
            x0p = max(0, x - radius)
            x1p = min(img.shape[1], x + radius + 1)
            img[y0p:y1p, x0p:x1p] = color


def _draw_arrow(
    img: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    color: tuple[int, int, int],
    thickness: int,
    tip_length: float,
) -> None:
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 1.0e-3:
        return

    try:
        import cv2

        pt1 = (int(round(float(start[0]))), int(round(float(start[1]))))
        pt2 = (int(round(float(end[0]))), int(round(float(end[1]))))
        cv2.arrowedLine(img, pt1, pt2, color, int(thickness), tipLength=float(tip_length))
        return
    except Exception:
        pass

    _draw_line(img, start, end, color=color, thickness=thickness)
    direction = delta / length
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    head_len = max(4.0, float(tip_length) * length)
    head_width = 0.55 * head_len
    left = end - direction * head_len + normal * head_width
    right = end - direction * head_len - normal * head_width
    _draw_line(img, end, left, color=color, thickness=thickness)
    _draw_line(img, end, right, color=color, thickness=thickness)
