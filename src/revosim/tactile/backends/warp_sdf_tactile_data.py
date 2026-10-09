# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WarpSdfTactileSensorData:
    """Data container for :class:`WarpSdfTactileSensor`."""

    tactile_points_w: torch.Tensor | None = None
    """Flattened tactile points per env: shape (E, S*P, 4) with columns [x, y, z, fn]."""

    tactile_points_w_per_sensor: torch.Tensor | None = None
    """Tactile points per env and sensor: shape (E, S, P, 4)."""

    pressure_force_raw_per_sensor: torch.Tensor | None = None
    """Raw calibrated force before optional normalization: shape (E, S, P)."""

    pressure_force_map_raw: torch.Tensor | None = None
    """Raw calibrated force maps: shape (E, S, H, W)."""

    pressure_force_map: torch.Tensor | None = None
    """Displayed/learning force maps after optional normalization: shape (E, S, H, W)."""

    signed_distance_map: torch.Tensor | None = None
    """Signed SDF values in meters: shape (E, S, H, W)."""

    penetration_map: torch.Tensor | None = None
    """Per-taxel penetration depth in meters: shape (E, S, H, W)."""

    penetration_velocity_map: torch.Tensor | None = None
    """Per-taxel penetration velocity in meters/second: shape (E, S, H, W)."""

    taxel_normals_w_per_sensor: torch.Tensor | None = None
    """World-frame taxel normals: shape (E, S, P, 3)."""
