from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:  # Torch is available in the Isaac Lab runtime, but not always in utility environments.
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test envs
    torch = None


Array = Any


@dataclass(frozen=True)
class TaxelLayout:
    """Canonical SI-unit taxel layout for one tactile region."""

    sensor_id: int
    link_name: str
    points_l_m: Array
    normals_l: Array
    image_shape: tuple[int, int]
    taxel_area_m2: Array | float = 1.0
    frame: str = "link"
    source: str = "unknown"

    @classmethod
    def from_taxel_map(
        cls,
        taxel_map: "PressureTaxelMap",
        *,
        sensor_id: int = 0,
        frame: str = "link",
        source: str = "pressure_taxel_map",
    ) -> "TaxelLayout":
        return cls(
            sensor_id=int(sensor_id),
            link_name=taxel_map.link_name,
            points_l_m=taxel_map.points_l,
            normals_l=taxel_map.normals_l,
            image_shape=taxel_map.image_shape,
            taxel_area_m2=taxel_map.calibration.area,
            frame=str(frame),
            source=str(source),
        )


@dataclass(frozen=True)
class PenetrationFrame:
    """Canonical penetration data in SI units.

    Arrays are expected to use shape ``(env, sensor, H, W)`` when they are
    stacked by an Isaac environment.
    """

    penetration_m: Array
    signed_distance_m: Array | None = None
    contact_mask: Array | None = None
    taxel_pose_w: Array | None = None
    normal_w: Array | None = None
    sample_support_fraction: Array | None = None
    sample_active_count: Array | None = None
    sample_mean_penetration_m: Array | None = None
    sample_positive_mean_penetration_m: Array | None = None
    sample_max_penetration_m: Array | None = None
    sample_penetrations_m: Array | None = None
    sample_offsets_l: Array | None = None
    sample_points_l_m: Array | None = None


@dataclass(frozen=True)
class PressureFrame:
    """Canonical calibrated pressure data in SI units."""

    pressure_raw_n: Array
    pressure_norm: Array
    total_force_n: Array
    center_of_pressure_px: Array
    calibration_id: str = "default"

    @classmethod
    def from_maps(
        cls,
        pressure_raw_n: Array,
        pressure_norm: Array,
        *,
        calibration_id: str = "default",
    ) -> "PressureFrame":
        total, center = pressure_map_stats(pressure_raw_n)
        return cls(
            pressure_raw_n=pressure_raw_n,
            pressure_norm=pressure_norm,
            total_force_n=total,
            center_of_pressure_px=center,
            calibration_id=str(calibration_id),
        )


@dataclass(frozen=True)
class PressureCalibration:
    """Per-taxel pressure response parameters.

    The default calibration preserves the previous WarpSDF behavior when used
    with `stiffness` and `max_force`: force = clamp(stiffness * penetration).
    `damping` is optional and is applied only when penetration velocity is
    supplied, giving a Kelvin-Voigt-compatible response.
    """

    gain: float | Array = 1.0
    bias: float | Array = 0.0
    stiffness: float | Array = 5_000.0
    damping: float | Array = 0.0
    max_force: float | Array = 10.0
    gamma: float | Array = 1.0
    threshold: float | Array = 0.0
    area: float | Array = 1.0


@dataclass(frozen=True)
class PressureContactEventBatch:
    """Discrete pressure contacts already expressed in one tactile link frame."""

    source_link: str
    contact_points_l: Array
    contact_normals_l: Array
    normal_forces: Array
    shear_forces_l: Array | None = None

    def is_empty(self) -> bool:
        return int(_shape(self.normal_forces)[0]) == 0


@dataclass(frozen=True)
class PressureMapOutput:
    """Calibrated taxel output in flat and image form."""

    raw_force: Array
    force: Array
    raw_force_map: Array
    force_map: Array


@dataclass(frozen=True)
class PressureTaxelMap:
    """Taxel layout and calibration for one tactile region."""

    link_name: str
    points_l: Array
    normals_l: Array
    image_shape: tuple[int, int]
    calibration: PressureCalibration = PressureCalibration()

    def __post_init__(self) -> None:
        rows, cols = self.image_shape
        if rows <= 0 or cols <= 0:
            raise ValueError(f"image_shape must be positive, got {self.image_shape}")

        point_shape = _shape(self.points_l)
        normal_shape = _shape(self.normals_l)
        if len(point_shape) != 2 or point_shape[-1] != 3:
            raise ValueError(f"points_l must have shape (P, 3), got {point_shape}")
        if normal_shape != point_shape:
            raise ValueError(f"normals_l must match points_l shape, got {normal_shape} vs {point_shape}")
        if point_shape[0] != rows * cols:
            raise ValueError(
                f"image_shape {self.image_shape} expects {rows * cols} taxels, got {point_shape[0]}"
            )

    @classmethod
    def from_grid(
        cls,
        *,
        link_name: str,
        num_rows: int,
        num_cols: int,
        point_distance: float | None = None,
        row_distance: float | None = None,
        col_distance: float | None = None,
        normal_axis: int = 0,
        normal_offset: float = 0.0,
        normal_sign: float = 1.0,
        origin_xyz: tuple[float, float, float] | None = None,
        origin_rpy: tuple[float, float, float] | None = None,
        calibration: PressureCalibration | None = None,
        backend_like: Array | None = None,
    ) -> "PressureTaxelMap":
        """Create a flat rectangular taxel grid in a link-local frame."""

        if normal_axis not in (0, 1, 2):
            raise ValueError("normal_axis must be 0, 1, or 2")
        if row_distance is None:
            row_distance = point_distance
        if col_distance is None:
            col_distance = point_distance
        if row_distance is None or col_distance is None:
            raise ValueError("point_distance or both row_distance/col_distance must be provided")
        if row_distance <= 0.0:
            raise ValueError("row_distance must be positive")
        if col_distance <= 0.0:
            raise ValueError("col_distance must be positive")

        tangential_axes = [0, 1, 2]
        tangential_axes.remove(normal_axis)
        axis_u, axis_v = tangential_axes

        u = (np.arange(num_rows, dtype=np.float32) - (float(num_rows) - 1.0) / 2.0) * float(row_distance)
        v = (np.arange(num_cols, dtype=np.float32) - (float(num_cols) - 1.0) / 2.0) * float(col_distance)
        uu, vv = np.meshgrid(u, v, indexing="ij")
        points = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
        points[:, axis_u] = uu.reshape(-1)
        points[:, axis_v] = vv.reshape(-1)
        points[:, normal_axis] = float(normal_offset)

        normals = np.zeros_like(points)
        normals[:, normal_axis] = 1.0 if normal_sign >= 0.0 else -1.0
        points, normals = _apply_origin_pose_np(points, normals, origin_xyz=origin_xyz, origin_rpy=origin_rpy)

        points = _to_backend(points, backend_like)
        normals = _to_backend(normals, backend_like)
        return cls(
            link_name=link_name,
            points_l=points,
            normals_l=normals,
            image_shape=(int(num_rows), int(num_cols)),
            calibration=calibration or PressureCalibration(),
        )

    @classmethod
    def from_npy(
        cls,
        *,
        link_name: str,
        points_npy: str | Path,
        normals_npy: str | Path,
        resolution_step: int = 1,
        correction_scale: float = 1.0e-3,
        invert_normals: bool = False,
        origin_xyz: tuple[float, float, float] | None = None,
        origin_rpy: tuple[float, float, float] | None = None,
        calibration: PressureCalibration | None = None,
        backend_like: Array | None = None,
    ) -> "PressureTaxelMap":
        """Load an existing tactile surface map from `.npy` point/normal files."""

        step = max(1, int(resolution_step))
        points = np.load(str(points_npy)).astype(np.float32)
        normals = np.load(str(normals_npy)).astype(np.float32)
        if points.shape != normals.shape or points.ndim not in (2, 3) or points.shape[-1] != 3:
            raise ValueError(f"points/normals must match (N,3) or (H,W,3), got {points.shape}/{normals.shape}")
        if points.ndim == 2:
            # Ordered mx_touch taxels have no rectangular image topology.
            points, normals = points[::step, None, :], normals[::step, None, :]
        else:
            points, normals = points[::step, ::step, :], normals[::step, ::step, :]

        rows, cols = int(points.shape[0]), int(points.shape[1])
        points = points.reshape(-1, 3) * float(correction_scale)
        normals = normals.reshape(-1, 3)
        if invert_normals:
            normals = -normals
        normals = _normalize(normals)
        points, normals = _apply_origin_pose_np(points, normals, origin_xyz=origin_xyz, origin_rpy=origin_rpy)

        return cls(
            link_name=link_name,
            points_l=_to_backend(points, backend_like),
            normals_l=_to_backend(normals, backend_like),
            image_shape=(rows, cols),
            calibration=calibration or PressureCalibration(),
        )

    def force_from_penetration(
        self,
        penetration: Array,
        *,
        penetration_velocity: Array | None = None,
        normalize: bool = True,
    ) -> PressureMapOutput:
        """Convert per-taxel penetration depth to a calibrated pressure map."""

        flat, prefix = _flatten_taxel_values(penetration, self.num_taxels)
        velocity_flat = None
        if penetration_velocity is not None:
            velocity_flat, velocity_prefix = _flatten_taxel_values(penetration_velocity, self.num_taxels)
            if velocity_prefix != prefix:
                raise ValueError(
                    f"penetration_velocity prefix {velocity_prefix} does not match penetration prefix {prefix}"
                )
        raw, force = calibrate_penetration(
            flat,
            self.calibration,
            penetration_velocity=velocity_flat,
            normalize=normalize,
        )
        return PressureMapOutput(
            raw_force=raw,
            force=force,
            raw_force_map=_reshape_taxel_values(raw, prefix, self.image_shape),
            force_map=_reshape_taxel_values(force, prefix, self.image_shape),
        )

    def force_from_contacts(
        self,
        events: PressureContactEventBatch,
        *,
        kernel_sigma: float | None = None,
        kernel_radius: float | None = None,
        conserve_total_force: bool = True,
        normalize: bool = True,
    ) -> PressureMapOutput:
        """Spread discrete contact forces onto this taxel grid."""

        raw_flat = spread_contact_events_to_taxels(
            self,
            events,
            kernel_sigma=kernel_sigma,
            kernel_radius=kernel_radius,
            conserve_total_force=conserve_total_force,
        )
        raw, force = calibrate_force(raw_flat, self.calibration, normalize=normalize)
        return PressureMapOutput(
            raw_force=raw,
            force=force,
            raw_force_map=_reshape_taxel_values(raw, (), self.image_shape),
            force_map=_reshape_taxel_values(force, (), self.image_shape),
        )

    @property
    def num_taxels(self) -> int:
        return int(self.image_shape[0] * self.image_shape[1])


class CalibratedPressureMapSensor:
    """Backend-agnostic calibrated pressure-map generator.

    This class is intentionally independent from Isaac Lab. Backends can feed it
    SDF penetrations, PhysX-like contact events, or future URDF pressure contacts.
    """

    def __init__(
        self,
        taxel_maps: Iterable[PressureTaxelMap],
        *,
        kernel_sigma: float | None = None,
        kernel_radius: float | None = None,
        conserve_total_force: bool = True,
        normalize: bool = True,
    ):
        self.taxel_maps = list(taxel_maps)
        if not self.taxel_maps:
            raise ValueError("CalibratedPressureMapSensor requires at least one taxel map")
        self.kernel_sigma = kernel_sigma
        self.kernel_radius = kernel_radius
        self.conserve_total_force = bool(conserve_total_force)
        self.normalize = bool(normalize)

    def from_penetrations(
        self,
        penetrations: Array,
        *,
        penetration_velocity: Array | None = None,
    ) -> PressureMapOutput:
        """Map SDF penetration arrays to stacked pressure maps.

        Accepted shapes are (S, P), (E, S, P), (S, H, W), or (E, S, H, W).
        """

        arr = penetrations
        sensor_count = len(self.taxel_maps)
        sensor_values = _split_sensor_values(arr, self.taxel_maps)
        velocity_values = (
            _split_sensor_values(penetration_velocity, self.taxel_maps)
            if penetration_velocity is not None
            else [None] * sensor_count
        )

        raw_values = []
        values = []
        raw_maps = []
        maps = []
        for taxel_map, sensor_arr, velocity_arr in zip(self.taxel_maps, sensor_values, velocity_values, strict=True):
            output = taxel_map.force_from_penetration(
                sensor_arr,
                penetration_velocity=velocity_arr,
                normalize=self.normalize,
            )
            raw_values.append(output.raw_force)
            values.append(output.force)
            raw_maps.append(output.raw_force_map)
            maps.append(output.force_map)

        axis = 1 if _has_env_axis(arr, sensor_count, self.taxel_maps[0].num_taxels) else 0
        return PressureMapOutput(
            raw_force=_stack(raw_values, axis=axis),
            force=_stack(values, axis=axis),
            raw_force_map=_stack(raw_maps, axis=axis),
            force_map=_stack(maps, axis=axis),
        )

    def from_contact_events(
        self,
        events_by_link: dict[str, PressureContactEventBatch] | Iterable[PressureContactEventBatch],
    ) -> PressureMapOutput:
        """Map contact events to stacked calibrated force maps."""

        if isinstance(events_by_link, dict):
            event_map = events_by_link
        else:
            event_map = {events.source_link: events for events in events_by_link}

        raw_values = []
        values = []
        raw_maps = []
        maps = []
        for taxel_map in self.taxel_maps:
            events = event_map.get(taxel_map.link_name)
            if events is None:
                events = empty_contact_events(taxel_map.link_name, like=taxel_map.points_l)
            output = taxel_map.force_from_contacts(
                events,
                kernel_sigma=self.kernel_sigma,
                kernel_radius=self.kernel_radius,
                conserve_total_force=self.conserve_total_force,
                normalize=self.normalize,
            )
            raw_values.append(output.raw_force)
            values.append(output.force)
            raw_maps.append(output.raw_force_map)
            maps.append(output.force_map)

        return PressureMapOutput(
            raw_force=_stack(raw_values, axis=0),
            force=_stack(values, axis=0),
            raw_force_map=_stack(raw_maps, axis=0),
            force_map=_stack(maps, axis=0),
        )


class UrdfPressureContactAdapter:
    """Convert future dexterous-hand pressure contacts into taxel events.

    The adapter accepts plain dictionaries so it can be wired to URDF metadata,
    robot drivers, PhysX contact records, or logs without changing downstream
    pressure-map code.
    """

    def __init__(self, taxel_maps: Iterable[PressureTaxelMap]):
        self.taxel_maps = {taxel_map.link_name: taxel_map for taxel_map in taxel_maps}

    def events_by_link(
        self,
        contacts: Iterable[dict[str, Any]],
        *,
        link_poses_w: dict[str, tuple[Array, Array]] | None = None,
    ) -> dict[str, PressureContactEventBatch]:
        grouped: dict[str, list[dict[str, Any]]] = {link: [] for link in self.taxel_maps}
        for contact in contacts:
            link = str(_first_present(contact, "source_link", "link_name", "body_name", "body", "source_body") or "")
            if link in grouped:
                grouped[link].append(contact)

        return {
            link: self._events_for_link(link, items, link_poses_w=link_poses_w)
            for link, items in grouped.items()
        }

    def _events_for_link(
        self,
        link: str,
        contacts: list[dict[str, Any]],
        *,
        link_poses_w: dict[str, tuple[Array, Array]] | None,
    ) -> PressureContactEventBatch:
        taxel_map = self.taxel_maps[link]
        if not contacts:
            return empty_contact_events(link, like=taxel_map.points_l)

        points = []
        normals = []
        forces = []
        shear = []
        for contact in contacts:
            point_l = _first_present(contact, "contact_point_l", "point_l", "local_point")
            normal_l = _first_present(contact, "contact_normal_l", "normal_l", "local_normal")
            point_w = _first_present(contact, "contact_point_w", "point_w", "position_w", "position")
            normal_w = _first_present(contact, "contact_normal_w", "normal_w", "normal")
            has_local_normal = normal_l is not None
            has_world_normal = normal_w is not None

            if point_l is None or normal_l is None:
                if link_poses_w is None or link not in link_poses_w:
                    raise ValueError(
                        f"Contact for {link!r} needs local point/normal or a link_poses_w entry"
                    )
                if (point_l is None and point_w is None) or (normal_l is None and normal_w is None):
                    raise ValueError(
                        f"Contact for {link!r} needs contact point/normal in local or world coordinates"
                    )
                pos_w, quat_w = link_poses_w[link]
                if point_l is None:
                    point_l = quat_apply_inverse_np(
                        np.asarray(quat_w, dtype=np.float32),
                        np.asarray(point_w, dtype=np.float32) - np.asarray(pos_w, dtype=np.float32),
                    )
                if normal_l is None:
                    normal_l = quat_apply_inverse_np(
                        np.asarray(quat_w, dtype=np.float32),
                        np.asarray(normal_w, dtype=np.float32),
                    )

            points.append(point_l)
            normals.append(normal_l)
            normal_force = _first_present(contact, "normal_force", "force_n", "normal_force_n")
            force = _first_present(contact, "force")
            if normal_force is None:
                normal_force = _scalar_or_none(force)
            if normal_force is None:
                force_l = _first_present(contact, "force_l", "contact_force_l")
                force_w = _first_present(contact, "force_w", "contact_force_w")
                ambiguous_force = _vector_or_none(force)
                if force_l is None and force_w is None and ambiguous_force is not None:
                    if has_local_normal:
                        force_l = ambiguous_force
                    elif has_world_normal:
                        force_w = ambiguous_force
                if force_l is not None:
                    normal_force = _normal_force_component(force_l, normal_l)
                elif force_w is not None and normal_w is not None:
                    normal_force = _normal_force_component(force_w, normal_w)
                elif force_w is not None and link_poses_w is not None and link in link_poses_w:
                    _pos_w, quat_w = link_poses_w[link]
                    force_l = quat_apply_inverse_np(
                        np.asarray(quat_w, dtype=np.float32),
                        np.asarray(force_w, dtype=np.float32),
                    )
                    normal_force = _normal_force_component(force_l, normal_l)
            forces.append(float(normal_force) if normal_force is not None else 0.0)
            shear_l = _first_present(contact, "shear_force_l", "tangent_force_l")
            if shear_l is not None:
                shear.append(shear_l)

        shear_arr = None
        if len(shear) == len(points):
            shear_arr = _to_backend(np.asarray(shear, dtype=np.float32), taxel_map.points_l)

        return PressureContactEventBatch(
            source_link=link,
            contact_points_l=_to_backend(np.asarray(points, dtype=np.float32), taxel_map.points_l),
            contact_normals_l=_to_backend(_normalize(np.asarray(normals, dtype=np.float32)), taxel_map.points_l),
            normal_forces=_to_backend(np.asarray(forces, dtype=np.float32), taxel_map.points_l),
            shear_forces_l=shear_arr,
        )


def calibrate_penetration(
    penetration: Array,
    calibration: PressureCalibration,
    *,
    penetration_velocity: Array | None = None,
    normalize: bool = True,
) -> tuple[Array, Array]:
    active = penetration > 0.0
    raw = _mul(_as_like(calibration.stiffness, penetration), penetration)
    if penetration_velocity is not None:
        velocity = _as_like(penetration_velocity, penetration)
        raw = _add(raw, _mul(_as_like(calibration.damping, penetration), velocity))
    raw = _mul(_as_like(calibration.gain, penetration), raw)
    raw = _add(raw, _as_like(calibration.bias, penetration))
    raw = _where(active, raw, _zeros_like(raw))
    return _finish_calibration(raw, calibration, normalize=normalize)


def calibrate_force(
    force: Array,
    calibration: PressureCalibration,
    *,
    normalize: bool = True,
) -> tuple[Array, Array]:
    active = force > 0.0
    raw = _mul(_as_like(calibration.gain, force), force)
    raw = _add(raw, _as_like(calibration.bias, force))
    raw = _where(active, raw, _zeros_like(raw))
    return _finish_calibration(raw, calibration, normalize=normalize)


def calibrate_gap_fraction(
    gap_closure: Array,
    shell_thickness: float,
    calibration: PressureCalibration,
    *,
    normalize: bool = True,
) -> tuple[Array, Array]:
    """Map unsigned surface-gap closure to pressure without a spring law."""

    active = gap_closure > 0.0
    denom = max(float(shell_thickness), 1.0e-12)
    fraction = _clip(_div(gap_closure, denom), 0.0, 1.0)
    raw = _mul(fraction, _as_like(calibration.max_force, gap_closure))
    raw = _mul(_as_like(calibration.gain, gap_closure), raw)
    raw = _add(raw, _as_like(calibration.bias, gap_closure))
    raw = _where(active, raw, _zeros_like(raw))
    return _finish_calibration(raw, calibration, normalize=normalize)


def spread_contact_events_to_taxels(
    taxel_map: PressureTaxelMap,
    events: PressureContactEventBatch,
    *,
    kernel_sigma: float | None = None,
    kernel_radius: float | None = None,
    conserve_total_force: bool = True,
) -> Array:
    points = taxel_map.points_l
    normals = _normalize_backend(taxel_map.normals_l)
    if events.is_empty():
        return _zeros((taxel_map.num_taxels,), like=points)

    contact_points = _as_like(events.contact_points_l, points)
    contact_normals = _normalize_backend(_as_like(events.contact_normals_l, points))
    normal_forces = _as_like(events.normal_forces, points)

    sigma = float(kernel_sigma) if kernel_sigma is not None else _default_kernel_sigma(points)
    sigma = max(sigma, 1.0e-8)
    radius = float(kernel_radius) if kernel_radius is not None else 3.0 * sigma

    diff = _sub(_expand_dims(points, 0), _expand_dims(contact_points, 1))  # (C, P, 3)
    normals_c = _expand_dims(normals, 0)
    normal_dist = _sum(_mul(diff, normals_c), axis=-1, keepdims=True)
    tangent = _sub(diff, _mul(normal_dist, normals_c))
    dist2 = _sum(_mul(tangent, tangent), axis=-1)
    align = _clip_min(_sum(_mul(normals_c, _expand_dims(contact_normals, 1)), axis=-1), 0.0)
    weights = _mul(_exp(_mul(dist2, -0.5 / (sigma * sigma))), align)

    if radius > 0.0:
        weights = _where(dist2 <= radius * radius, weights, _zeros_like(weights))

    area = _as_like(taxel_map.calibration.area, points)
    weights = _mul(weights, area)

    if conserve_total_force:
        denom = _sum(weights, axis=-1, keepdims=True)
        weights = _where(denom > 1.0e-12, _div(weights, denom + 1.0e-12), _zeros_like(weights))

    contribution = _mul(weights, _expand_dims(normal_forces, 1))
    return _sum(contribution, axis=0)


def empty_contact_events(source_link: str, *, like: Array | None = None) -> PressureContactEventBatch:
    return PressureContactEventBatch(
        source_link=source_link,
        contact_points_l=_zeros((0, 3), like=like),
        contact_normals_l=_zeros((0, 3), like=like),
        normal_forces=_zeros((0,), like=like),
        shear_forces_l=None,
    )


def pressure_map_stats(force_map: Array) -> tuple[np.ndarray, np.ndarray]:
    """Return total force and center-of-pressure for maps ending in ``(H, W)``."""

    arr = np.nan_to_num(_to_numpy(force_map).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim < 2:
        raise ValueError(f"force_map must end in (H, W), got shape {arr.shape}")
    h, w = arr.shape[-2:]
    total = arr.sum(axis=(-2, -1)).astype(np.float32)
    rows, cols = np.indices((h, w), dtype=np.float64)
    safe_total = np.where(total > 1.0e-12, total, 1.0)
    row_c = (arr * rows).sum(axis=(-2, -1)) / safe_total
    col_c = (arr * cols).sum(axis=(-2, -1)) / safe_total
    center = np.stack(
        [
            np.where(total > 1.0e-12, row_c, -1.0),
            np.where(total > 1.0e-12, col_c, -1.0),
        ],
        axis=-1,
    ).astype(np.float32)
    return total, center


def coerce_taxel_calibration_value(
    value: Array,
    image_shape: tuple[int, int],
    *,
    backend_like: Array | None = None,
    field_name: str = "calibration",
) -> float | Array:
    """Return a scalar or flat per-taxel value compatible with pressure maps.

    Calibration files often store per-taxel values as image-shaped ``(H, W)``
    arrays, while the WarpSDF backend applies calibration to flat ``(P,)``
    taxel vectors. This helper is the shared boundary between those contracts.
    """

    arr = _to_numpy(value)
    if arr.ndim == 0:
        return float(arr)

    rows, cols = int(image_shape[0]), int(image_shape[1])
    expected = rows * cols
    if tuple(arr.shape) == (rows, cols):
        flat = arr.reshape(expected)
    elif tuple(arr.shape) == (expected,):
        flat = arr
    else:
        raise ValueError(
            f"{field_name} must be scalar, shape {(rows, cols)}, or shape {(expected,)}, got {tuple(arr.shape)}"
        )
    return _to_backend(np.asarray(flat, dtype=np.float32), backend_like)


def _apply_origin_pose_np(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    origin_xyz: tuple[float, float, float] | None,
    origin_rpy: tuple[float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(origin_xyz if origin_xyz is not None else (0.0, 0.0, 0.0), dtype=np.float32)
    rpy = np.asarray(origin_rpy if origin_rpy is not None else (0.0, 0.0, 0.0), dtype=np.float32)
    if np.allclose(xyz, 0.0) and np.allclose(rpy, 0.0):
        return points, normals
    rot = _rpy_matrix_np(rpy)
    return points @ rot.T + xyz.reshape(1, 3), _normalize(normals @ rot.T)


def _rpy_matrix_np(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in np.asarray(rpy, dtype=np.float32))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz @ ry @ rx


def quat_apply_inverse_np(quat_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float32)
    v = np.asarray(vector, dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1.0e-12)
    w, x, y, z = q
    conj = np.array([w, -x, -y, -z], dtype=np.float32)
    return quat_apply_np(conj, v)


def quat_apply_np(quat_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float32)
    v = np.asarray(vector, dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1.0e-12)
    w, x, y, z = q
    qvec = np.array([x, y, z], dtype=np.float32)
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (w * uv + uuv)


def _finish_calibration(raw: Array, calibration: PressureCalibration, *, normalize: bool) -> tuple[Array, Array]:
    threshold = _as_like(calibration.threshold, raw)
    raw = _where(raw >= threshold, raw, _zeros_like(raw))
    max_force = _as_like(calibration.max_force, raw)
    raw = _clip(raw, 0.0, max_force)
    if normalize:
        force = _div(raw, max_force + 1.0e-12)
        gamma = _as_like(calibration.gamma, force)
        force = _pow(_clip(force, 0.0, 1.0), gamma)
    else:
        force = raw
    return raw, force


def _split_sensor_values(values: Array, taxel_maps: list[PressureTaxelMap]) -> list[Array]:
    shape = _shape(values)
    sensor_count = len(taxel_maps)
    if len(shape) == 2 and sensor_count == 1 and shape == taxel_maps[0].image_shape:
        return [values]
    if len(shape) == 1 and sensor_count == 1 and shape[0] == taxel_maps[0].num_taxels:
        return [values]
    if len(shape) >= 3 and shape[1] == sensor_count:
        return [_take(values, i, axis=1) for i in range(sensor_count)]
    if len(shape) >= 2 and shape[0] == sensor_count:
        return [_take(values, i, axis=0) for i in range(sensor_count)]
    raise ValueError(f"Could not split values with shape {shape} across {sensor_count} taxel maps")


def _has_env_axis(values: Array, sensor_count: int, num_taxels: int) -> bool:
    shape = _shape(values)
    return len(shape) >= 3 and shape[1] == sensor_count or len(shape) == 3 and shape[-1] == num_taxels


def _flatten_taxel_values(values: Array, num_taxels: int) -> tuple[Array, tuple[int, ...]]:
    shape = _shape(values)
    if len(shape) >= 2 and shape[-2] * shape[-1] == num_taxels:
        prefix = shape[:-2]
        return _reshape(values, (*prefix, num_taxels)), prefix
    if len(shape) >= 1 and shape[-1] == num_taxels:
        return values, shape[:-1]
    raise ValueError(f"Expected last dim or last two dims to contain {num_taxels} taxels, got {shape}")


def _reshape_taxel_values(values: Array, prefix: tuple[int, ...], image_shape: tuple[int, int]) -> Array:
    return _reshape(values, (*prefix, *image_shape))


def _default_kernel_sigma(points: Array) -> float:
    pts = _to_numpy(points)
    if pts.shape[0] <= 1:
        return 1.0e-3
    ref = pts[0]
    distances = np.linalg.norm(pts[1:] - ref, axis=-1)
    positive = distances[distances > 1.0e-9]
    if positive.size == 0:
        return 1.0e-3
    return float(np.min(positive))


def _is_torch(value: Array) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _shape(value: Array) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = np.shape(value)
    return tuple(int(v) for v in shape)


def _first_present(mapping: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _scalar_or_none(value: Array | None) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 0 or arr.size == 1:
        return float(arr.reshape(-1)[0])
    return None


def _vector_or_none(value: Array | None) -> Array | None:
    if value is None:
        return None
    return None if _scalar_or_none(value) is not None else value


def _normal_force_component(force: Array, normal: Array) -> float:
    force_vec = np.asarray(force, dtype=np.float32)
    normal_vec = np.asarray(normal, dtype=np.float32)
    norm = float(np.linalg.norm(normal_vec))
    if norm <= 1.0e-12:
        return 0.0
    return max(float(np.dot(force_vec, normal_vec / norm)), 0.0)


def _as_like(value: Array, like: Array) -> Array:
    if _is_torch(like):
        return torch.as_tensor(value, device=like.device, dtype=like.dtype)
    return np.asarray(value, dtype=np.float32)


def _to_backend(value: Array, like: Array | None) -> Array:
    if like is not None and _is_torch(like):
        return torch.as_tensor(value, device=like.device, dtype=like.dtype)
    return np.asarray(value, dtype=np.float32)


def _to_numpy(value: Array) -> np.ndarray:
    if _is_torch(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize(value: np.ndarray) -> np.ndarray:
    return value / (np.linalg.norm(value, axis=-1, keepdims=True) + 1.0e-12)


def _normalize_backend(value: Array) -> Array:
    if _is_torch(value):
        return value / torch.linalg.norm(value, dim=-1, keepdim=True).clamp_min(1.0e-12)
    return _normalize(np.asarray(value, dtype=np.float32))


def _zeros(shape: tuple[int, ...], *, like: Array | None = None) -> Array:
    if like is not None and _is_torch(like):
        return torch.zeros(shape, device=like.device, dtype=like.dtype)
    return np.zeros(shape, dtype=np.float32)


def _zeros_like(value: Array) -> Array:
    if _is_torch(value):
        return torch.zeros_like(value)
    return np.zeros_like(value, dtype=np.float32)


def _reshape(value: Array, shape: tuple[int, ...]) -> Array:
    return value.reshape(shape)


def _take(value: Array, index: int, *, axis: int) -> Array:
    if _is_torch(value):
        return torch.select(value, dim=axis, index=index)
    return np.take(value, index, axis=axis)


def _stack(values: list[Array], *, axis: int) -> Array:
    if values and _is_torch(values[0]):
        return torch.stack(values, dim=axis)
    return np.stack(values, axis=axis)


def _expand_dims(value: Array, axis: int) -> Array:
    if _is_torch(value):
        return value.unsqueeze(axis)
    return np.expand_dims(value, axis=axis)


def _sum(value: Array, *, axis: int, keepdims: bool = False) -> Array:
    if _is_torch(value):
        return torch.sum(value, dim=axis, keepdim=keepdims)
    return np.sum(value, axis=axis, keepdims=keepdims)


def _exp(value: Array) -> Array:
    if _is_torch(value):
        return torch.exp(value)
    return np.exp(value)


def _pow(value: Array, exponent: Array) -> Array:
    if _is_torch(value):
        return torch.pow(value, exponent)
    return np.power(value, exponent)


def _clip(value: Array, minimum: float | Array, maximum: float | Array) -> Array:
    if _is_torch(value):
        min_value = _as_like(minimum, value)
        max_value = _as_like(maximum, value)
        return torch.minimum(torch.maximum(value, min_value), max_value)
    return np.minimum(np.maximum(value, minimum), maximum)


def _clip_min(value: Array, minimum: float) -> Array:
    if _is_torch(value):
        return torch.clamp_min(value, minimum)
    return np.maximum(value, minimum)


def _where(mask: Array, true_value: Array, false_value: Array) -> Array:
    if _is_torch(true_value) or _is_torch(false_value):
        like = true_value if _is_torch(true_value) else false_value
        return torch.where(mask, _as_like(true_value, like), _as_like(false_value, like))
    return np.where(mask, true_value, false_value)


def _add(left: Array, right: Array) -> Array:
    return left + right


def _sub(left: Array, right: Array) -> Array:
    return left - right


def _mul(left: Array, right: Array) -> Array:
    return left * right


def _div(left: Array, right: Array) -> Array:
    return left / right
