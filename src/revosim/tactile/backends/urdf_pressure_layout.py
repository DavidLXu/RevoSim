from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

from .pressure_taxel_map import PressureCalibration, PressureTaxelMap


PRESSURE_PAD_TAGS = {
    "pressure_pad",
    "pressure_taxel_map",
    "resistive_pad",
    "taxel_map",
    "tactile_pad",
}
PRESSURE_SENSOR_TYPES = {"pressure", "pressure_pad", "resistive", "taxel", "tactile"}
PRESSURE_ORIGIN_SEMANTICS = {
    "pad_surface": "pad_surface",
    "surface": "pad_surface",
    "surface_taxel": "pad_surface",
    "pad_internal": "pad_internal",
    "internal": "pad_internal",
    "internal_taxel": "pad_internal",
    "link_surface": "link_surface",
    "tacmap_link_surface": "link_surface",
}


@dataclass(frozen=True)
class UrdfPressurePadSpec:
    """URDF-declared pressure-pad/taxel layout.

    The parser intentionally accepts custom URDF tags. A future hand URDF can
    declare either a regular grid or point/normal `.npy` files, and downstream
    WarpSDF/pressure-map code can consume the resulting `PressureTaxelMap`.
    """

    link_name: str
    name: str | None = None
    points_npy: Path | None = None
    normals_npy: Path | None = None
    num_rows: int | None = None
    num_cols: int | None = None
    taxel_count: int | None = None
    point_distance: float | None = None
    row_distance: float | None = None
    col_distance: float | None = None
    pad_size: tuple[float, float] | None = None
    pad_size_semantics: str = "cell_extent"
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal_axis: int = 0
    normal_offset: float = 0.0
    normal_sign: float = 1.0
    origin_semantics: str = "pad_surface"
    resolution_step: int = 1
    correction_scale: float = 1.0e-3
    invert_normals: bool = False
    calibration: PressureCalibration = field(default_factory=PressureCalibration)
    attrs: dict[str, str] = field(default_factory=dict)

    @property
    def map_name(self) -> str:
        return self.name or self.link_name

    def to_taxel_map(self, *, backend_like=None) -> PressureTaxelMap:
        if self.points_npy is not None or self.normals_npy is not None:
            if self.points_npy is None or self.normals_npy is None:
                raise ValueError(f"Pressure pad {self.map_name!r} needs both points_npy and normals_npy")
            taxel_map = PressureTaxelMap.from_npy(
                link_name=self.link_name,
                points_npy=self.points_npy,
                normals_npy=self.normals_npy,
                resolution_step=self.resolution_step,
                correction_scale=self.correction_scale,
                invert_normals=self.invert_normals,
                origin_xyz=self.origin_xyz,
                origin_rpy=self.origin_rpy,
                calibration=self.calibration,
                backend_like=backend_like,
            )
            self._validate_taxel_count(taxel_map.image_shape[0] * taxel_map.image_shape[1])
            return taxel_map

        if (
            self.num_rows is None
            or self.num_cols is None
            or (self.point_distance is None and (self.row_distance is None or self.col_distance is None))
        ):
            raise ValueError(
                f"Pressure pad {self.map_name!r} needs either npy files or rows/cols with pitch or pad_size"
            )
        self._validate_taxel_count(int(self.num_rows) * int(self.num_cols))
        return PressureTaxelMap.from_grid(
            link_name=self.link_name,
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            point_distance=self.point_distance,
            row_distance=self.row_distance,
            col_distance=self.col_distance,
            normal_axis=self.normal_axis,
            normal_offset=self.normal_offset,
            normal_sign=self.normal_sign,
            origin_xyz=self.origin_xyz,
            origin_rpy=self.origin_rpy,
            calibration=self.calibration,
            backend_like=backend_like,
        )

    def _validate_taxel_count(self, actual_count: int) -> None:
        if self.taxel_count is not None and int(self.taxel_count) != int(actual_count):
            raise ValueError(
                f"Pressure pad {self.map_name!r} declares taxel_count={self.taxel_count}, "
                f"but layout has {actual_count} taxels"
            )


def load_pressure_pad_specs_from_urdf(
    urdf_path: str | Path,
    *,
    base_dir: str | Path | None = None,
    require_files: bool = True,
) -> list[UrdfPressurePadSpec]:
    """Parse custom pressure-pad declarations from a URDF file.

    Supported examples:

    - `<link name="tip"><pressure_pad rows="30" cols="30" point_distance="0.001" /></link>`
    - `<gazebo reference="tip"><pressure_taxel_map points_npy="..." normals_npy="..." /></gazebo>`
    - `<sensor name="tip_pad" type="pressure" link="tip" rows="30" cols="30" point_distance="0.001" />`
    """

    urdf_path = Path(urdf_path).expanduser().resolve()
    root = ET.parse(urdf_path).getroot()
    resolved_base = Path(base_dir).expanduser().resolve() if base_dir is not None else urdf_path.parent
    parent_map = {child: parent for parent in root.iter() for child in parent}

    specs: list[UrdfPressurePadSpec] = []
    for element in root.iter():
        tag = _local_name(element.tag)
        attrs = _local_attrs(element)
        sensor_type = str(attrs.get("type", "")).lower()
        is_pressure_tag = tag in PRESSURE_PAD_TAGS
        is_pressure_sensor = tag == "sensor" and sensor_type in PRESSURE_SENSOR_TYPES
        if not (is_pressure_tag or is_pressure_sensor):
            continue

        link_name = _resolve_link_name(element, parent_map)
        if not link_name:
            raise ValueError(f"Pressure pad tag {ET.tostring(element, encoding='unicode')} does not identify a link")

        spec = _spec_from_element(element, link_name=link_name, base_dir=resolved_base, require_files=require_files)
        specs.append(spec)

    return specs


def load_pressure_taxel_maps_from_urdf(
    urdf_path: str | Path,
    *,
    base_dir: str | Path | None = None,
    require_files: bool = True,
    backend_like=None,
) -> list[PressureTaxelMap]:
    specs = load_pressure_pad_specs_from_urdf(
        urdf_path,
        base_dir=base_dir,
        require_files=require_files,
    )
    return [spec.to_taxel_map(backend_like=backend_like) for spec in specs]


def select_pressure_pad_spec(
    specs: list[UrdfPressurePadSpec] | tuple[UrdfPressurePadSpec, ...],
    *,
    link_name: str | None = None,
) -> UrdfPressurePadSpec:
    if link_name is not None:
        aliases = [link_name]
        if link_name.endswith("_touch_link"):
            prefix = link_name[: -len("_touch_link")]
            aliases.extend((f"{prefix}_rubber_link", f"{prefix}_tubber_link"))
        matches = [spec for spec in specs if spec.link_name in aliases]
        if len(matches) != 1:
            raise ValueError(f"Expected one pressure pad for link {link_name!r}, found {len(matches)}")
        return matches[0]
    if len(specs) != 1:
        raise ValueError(f"Expected exactly one pressure pad when link_name is omitted, found {len(specs)}")
    return specs[0]


def load_touch_links_from_urdf(urdf_path: str | Path) -> list[str]:
    """Return URDF toucher links, i.e. physical tactile pad links named as ``*_touch_link``."""

    root = ET.parse(Path(urdf_path).expanduser()).getroot()
    return [
        str(link.attrib["name"])
        for link in root.findall("link")
        if "touch_link" in str(link.attrib.get("name", "")).lower()
    ]


def load_pressure_touch_links_from_urdf(urdf_path: str | Path) -> list[str]:
    """Return toucher links whose pressure taxel metadata should be declared with ``<pressure_pad>``."""

    return [link_name for link_name in load_touch_links_from_urdf(urdf_path) if _is_pressure_touch_link(link_name)]


def _spec_from_element(
    element: ET.Element,
    *,
    link_name: str,
    base_dir: Path,
    require_files: bool,
) -> UrdfPressurePadSpec:
    attrs = _merged_attrs(element)
    points_npy = _optional_path(attrs, ("points_npy", "points", "point_map"), base_dir, require_files)
    normals_npy = _optional_path(attrs, ("normals_npy", "normals", "normal_map"), base_dir, require_files)

    calibration = PressureCalibration(
        gain=_float_attr(attrs, ("gain", "pressure_gain"), 1.0),
        bias=_float_attr(attrs, ("bias", "pressure_bias"), 0.0),
        stiffness=_float_attr(attrs, ("stiffness", "k", "pressure_stiffness"), 5_000.0),
        damping=_float_attr(attrs, ("damping", "c", "pressure_damping"), 0.0),
        max_force=_float_attr(attrs, ("max_force", "force_max"), 10.0),
        gamma=_float_attr(attrs, ("gamma", "pressure_gamma"), 1.0),
        threshold=_float_attr(attrs, ("threshold", "pressure_threshold"), 0.0),
        area=_float_attr(attrs, ("area", "taxel_area"), 1.0),
    )
    num_rows, num_cols = _grid_shape_attrs(attrs)
    pad_size_semantics = _pad_size_semantics_attr(attrs)
    pad_size = _pad_size_attrs(attrs)
    point_distance = _float_attr(attrs, ("point_distance", "taxel_pitch", "pitch"), None)
    row_distance, col_distance = _grid_distance_attrs(
        attrs,
        num_rows=num_rows,
        num_cols=num_cols,
        point_distance=point_distance,
        pad_size=pad_size,
        pad_size_semantics=pad_size_semantics,
    )

    return UrdfPressurePadSpec(
        link_name=link_name,
        name=_str_attr(attrs, ("name", "pad_name"), None),
        points_npy=points_npy,
        normals_npy=normals_npy,
        num_rows=num_rows,
        num_cols=num_cols,
        taxel_count=_int_attr(attrs, ("taxel_count", "num_taxels", "taxels", "count"), None),
        point_distance=point_distance,
        row_distance=row_distance,
        col_distance=col_distance,
        pad_size=pad_size,
        pad_size_semantics=pad_size_semantics,
        origin_xyz=_float_triple_attr(
            attrs,
            ("origin_xyz", "xyz", "pad_origin_xyz", "taxel_origin_xyz"),
            (0.0, 0.0, 0.0),
        ),
        origin_rpy=_float_triple_attr(
            attrs,
            ("origin_rpy", "rpy", "pad_origin_rpy", "taxel_origin_rpy"),
            (0.0, 0.0, 0.0),
        ),
        normal_axis=_int_attr(attrs, ("normal_axis",), 0),
        normal_offset=_float_attr(attrs, ("normal_offset",), 0.0),
        normal_sign=_float_attr(attrs, ("normal_sign",), 1.0),
        origin_semantics=_origin_semantics_attr(attrs),
        resolution_step=max(1, _int_attr(attrs, ("resolution_step", "step"), 1)),
        correction_scale=_float_attr(attrs, ("correction_scale", "point_scale"), 1.0e-3),
        invert_normals=_bool_attr(attrs, ("invert_normals",), False),
        calibration=calibration,
        attrs=dict(attrs),
    )


def _resolve_link_name(element: ET.Element, parent_map: dict[ET.Element, ET.Element]) -> str | None:
    attrs = _local_attrs(element)
    for key in ("link", "link_name", "source_link", "reference"):
        value = attrs.get(key)
        if value:
            return str(value)

    current = parent_map.get(element)
    while current is not None:
        tag = _local_name(current.tag)
        attrs = _local_attrs(current)
        if tag == "link" and attrs.get("name"):
            return str(attrs["name"])
        if tag == "gazebo" and attrs.get("reference"):
            return str(attrs["reference"])
        current = parent_map.get(current)
    return None


def _is_pressure_touch_link(link_name: str) -> bool:
    name = str(link_name).lower()
    if "touch_link" not in name:
        return False
    return not ("fingertip" in name or "_tip" in name or "tip_" in name)


def _merged_attrs(element: ET.Element) -> dict[str, str]:
    attrs = _local_attrs(element)
    for child in element:
        tag = _local_name(child.tag)
        if tag in {"calibration", "grid", "map", "origin", "taxels"}:
            attrs.update(_local_attrs(child))
    return attrs


def _local_attrs(element: ET.Element) -> dict[str, str]:
    return {_local_name(str(key)): str(value) for key, value in element.attrib.items()}


def _optional_path(
    attrs: dict[str, str],
    names: tuple[str, ...],
    base_dir: Path,
    require_file: bool,
) -> Path | None:
    raw = _str_attr(attrs, names, None)
    if raw is None:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if require_file and not path.is_file():
        raise FileNotFoundError(path)
    return path


def _str_attr(attrs: dict[str, str], names: tuple[str, ...], default: str | None) -> str | None:
    for name in names:
        if name in attrs and str(attrs[name]) != "":
            return str(attrs[name])
    return default


def _int_attr(attrs: dict[str, str], names: tuple[str, ...], default: int | None) -> int | None:
    raw = _str_attr(attrs, names, None)
    return int(raw) if raw is not None else default


def _float_attr(attrs: dict[str, str], names: tuple[str, ...], default: float | None) -> float | None:
    raw = _str_attr(attrs, names, None)
    return float(raw) if raw is not None else default


def _bool_attr(attrs: dict[str, str], names: tuple[str, ...], default: bool) -> bool:
    raw = _str_attr(attrs, names, None)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _grid_shape_attrs(attrs: dict[str, str]) -> tuple[int | None, int | None]:
    rows = _int_attr(attrs, ("rows", "num_rows", "height", "h"), None)
    cols = _int_attr(attrs, ("cols", "num_cols", "width", "w"), None)
    raw_shape = _str_attr(attrs, ("resolution", "taxel_resolution", "grid_shape", "shape", "image_shape"), None)
    if raw_shape is None:
        return rows, cols

    parts = str(raw_shape).strip().lower().replace("x", " ").replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError(f"pressure pad resolution {raw_shape!r} must be formatted as '<rows>x<cols>'")
    parsed_rows, parsed_cols = int(parts[0]), int(parts[1])
    if rows is not None and rows != parsed_rows:
        raise ValueError(f"pressure pad rows={rows} conflicts with resolution rows={parsed_rows}")
    if cols is not None and cols != parsed_cols:
        raise ValueError(f"pressure pad cols={cols} conflicts with resolution cols={parsed_cols}")
    return parsed_rows, parsed_cols


def _grid_distance_attrs(
    attrs: dict[str, str],
    *,
    num_rows: int | None,
    num_cols: int | None,
    point_distance: float | None,
    pad_size: tuple[float, float] | None,
    pad_size_semantics: str,
) -> tuple[float | None, float | None]:
    row_distance = _float_attr(
        attrs,
        ("row_distance", "row_pitch", "taxel_row_pitch", "pitch_row", "pitch_u"),
        None,
    )
    col_distance = _float_attr(
        attrs,
        ("col_distance", "column_distance", "col_pitch", "column_pitch", "taxel_col_pitch", "pitch_col", "pitch_v"),
        None,
    )
    if row_distance is None:
        row_distance = point_distance
    if col_distance is None:
        col_distance = point_distance
    if pad_size is not None:
        row_extent, col_extent = pad_size
        if row_distance is None and num_rows is not None:
            row_distance = _distance_from_extent(row_extent, int(num_rows), pad_size_semantics, field_name="pad row size")
        if col_distance is None and num_cols is not None:
            col_distance = _distance_from_extent(col_extent, int(num_cols), pad_size_semantics, field_name="pad col size")
    return row_distance, col_distance


def _distance_from_extent(extent: float, count: int, semantics: str, *, field_name: str) -> float:
    if extent <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    if count <= 0:
        return float(extent)
    if semantics == "center_span":
        return float(extent) if count == 1 else float(extent) / float(count - 1)
    return float(extent) / float(count)


def _pad_size_attrs(attrs: dict[str, str]) -> tuple[float, float] | None:
    row_extent = _float_attr(
        attrs,
        ("pad_row_size", "pad_row_extent", "row_extent", "pad_height", "pad_height_m", "taxel_grid_height_m"),
        None,
    )
    col_extent = _float_attr(
        attrs,
        ("pad_col_size", "pad_col_extent", "col_extent", "pad_width", "pad_width_m", "taxel_grid_width_m"),
        None,
    )
    combined = _float_pair_attr(attrs, ("pad_size", "pad_size_m", "taxel_grid_size", "taxel_grid_size_m"))
    if combined is not None:
        if row_extent is not None and abs(float(row_extent) - combined[0]) > 1.0e-12:
            raise ValueError(f"pad row extent {row_extent} conflicts with pad_size row extent {combined[0]}")
        if col_extent is not None and abs(float(col_extent) - combined[1]) > 1.0e-12:
            raise ValueError(f"pad col extent {col_extent} conflicts with pad_size col extent {combined[1]}")
        row_extent = combined[0] if row_extent is None else row_extent
        col_extent = combined[1] if col_extent is None else col_extent
    if row_extent is None and col_extent is None:
        return None
    if row_extent is None or col_extent is None:
        raise ValueError("pressure pad size needs both row and column extents")
    return float(row_extent), float(col_extent)


def _float_pair_attr(attrs: dict[str, str], names: tuple[str, ...]) -> tuple[float, float] | None:
    raw = _str_attr(attrs, names, None)
    if raw is None:
        return None
    parts = str(raw).strip().lower().replace("x", " ").replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError(f"pressure pad size {raw!r} must be formatted as '<row_extent> <col_extent>'")
    return float(parts[0]), float(parts[1])


def _float_triple_attr(
    attrs: dict[str, str],
    names: tuple[str, ...],
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    raw = _str_attr(attrs, names, None)
    if raw is None:
        return default
    parts = str(raw).strip().replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"pressure pad vector {raw!r} must have three values")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _pad_size_semantics_attr(attrs: dict[str, str]) -> str:
    raw = _str_attr(attrs, ("pad_size_semantics", "size_semantics", "extent_semantics"), "cell_extent")
    key = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "cell": "cell_extent",
        "cell_extent": "cell_extent",
        "footprint": "cell_extent",
        "outer": "cell_extent",
        "outer_extent": "cell_extent",
        "surface_extent": "cell_extent",
        "center": "center_span",
        "center_span": "center_span",
        "center_extent": "center_span",
        "center_to_center": "center_span",
        "center_to_center_span": "center_span",
    }
    if key not in aliases:
        raise ValueError(f"unknown pressure pad size semantics {raw!r}; expected one of {sorted(aliases)}")
    return aliases[key]


def _origin_semantics_attr(attrs: dict[str, str]) -> str:
    raw = _str_attr(attrs, ("origin_semantics", "origin_type", "taxel_origin"), "pad_surface")
    key = str(raw).strip().lower().replace("-", "_")
    if key not in PRESSURE_ORIGIN_SEMANTICS:
        raise ValueError(
            f"unknown pressure origin semantics {raw!r}; expected one of {sorted(PRESSURE_ORIGIN_SEMANTICS)}"
        )
    return PRESSURE_ORIGIN_SEMANTICS[key]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1].lower()
