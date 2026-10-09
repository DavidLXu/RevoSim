"""TacEx GPU-Taxim RGB tactile adapter for RevoLab TacMap raw depth.

This adapter uses the vendored TacEx GPU-Taxim subset under
``integrate/third_party/tacex_gpu_taxim``.  It loads only TacEx's
``gpu_taxim/sim`` package so the full TacEx sensor framework is not required.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


INTEGRATE_ROOT = Path(__file__).resolve().parent
DEFAULT_TACEX_ROOT = INTEGRATE_ROOT / "third_party" / "tacex_gpu_taxim"
DEFAULT_TACEX_SIM_DIR = DEFAULT_TACEX_ROOT / "sim"
DEFAULT_TACEX_CALIB_DIR = DEFAULT_TACEX_ROOT / "calibs" / "640x480"


@dataclass
class RevoTacExRgbCfg:
    width: int = 320
    height: int = 240
    depth_scale: float = 1.0
    with_shadow: bool = False
    device: str = "cuda"
    calib_dir: str = str(DEFAULT_TACEX_CALIB_DIR)
    taxim_sim_dir: str = str(DEFAULT_TACEX_SIM_DIR)


@dataclass
class TacExRgbOutput:
    tactile_rgb: torch.Tensor
    taxim_height_map_mm: torch.Tensor
    depth_mm: torch.Tensor
    max_depth_mm: torch.Tensor
    active_pixels: torch.Tensor


class RevoTacExRgbAdapter:
    """Render TacEx-style tactile RGB from TacMap raw indentation depth.

    Input convention:
        ``tacmap_raw_m`` is positive indentation depth in meters.

    Taxim convention:
        height map is in millimeters, and negative values are in contact.

    Therefore this adapter uses ``taxim_height_map_mm = -tacmap_raw_m * 1000``.
    """

    def __init__(self, cfg: RevoTacExRgbCfg):
        self.cfg = cfg
        self._torch = self._import_torch()
        taxim_pkg = self._load_taxim_sim_package(Path(cfg.taxim_sim_dir).expanduser())
        calib_dir = Path(cfg.calib_dir).expanduser()
        if not calib_dir.is_dir():
            raise FileNotFoundError(f"TacEx RGB calibration folder does not exist: {calib_dir}")
        self._device = self._resolve_device(str(cfg.device))
        self._taxim = taxim_pkg.Taxim(calib_folder=calib_dir, backend="torch", device=self._device)

    def step(self, tacmap_raw_m: np.ndarray | torch.Tensor) -> TacExRgbOutput:
        height_map_mm, depth_mm = self._prepare_taxim_height_map(tacmap_raw_m)
        rgb = self._taxim.render_direct(
            height_map_mm,
            with_shadow=bool(self.cfg.with_shadow),
            press_depth=None,
            orig_hm_fmt=False,
        )
        rgb_u8 = self._rgb_tensor_to_uint8(rgb)
        return TacExRgbOutput(
            tactile_rgb=rgb_u8,
            taxim_height_map_mm=height_map_mm,
            depth_mm=depth_mm,
            max_depth_mm=torch.amax(depth_mm, dim=(1, 2)).to(dtype=torch.float32),
            active_pixels=torch.count_nonzero(depth_mm > 0.0, dim=(1, 2)).to(dtype=torch.float32),
        )

    def _prepare_taxim_height_map(self, tacmap_raw_m: np.ndarray | torch.Tensor):
        if isinstance(tacmap_raw_m, torch.Tensor):
            raw = tacmap_raw_m.detach().to(self._device).float()
        else:
            raw = torch.as_tensor(tacmap_raw_m, dtype=torch.float32, device=self._device)
        if raw.ndim == 2:
            raw = raw[None, ...]
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw = torch.clamp(raw, min=0.0)

        depth = raw * (1000.0 * float(self.cfg.depth_scale))
        if depth.shape[-2:] != (int(self.cfg.height), int(self.cfg.width)):
            depth = self._torch.nn.functional.interpolate(
                depth[:, None, :, :],
                size=(int(self.cfg.height), int(self.cfg.width)),
                mode="bilinear",
                align_corners=False,
            )[:, 0, :, :]
        height_map = -depth
        return height_map, depth

    def _rgb_tensor_to_uint8(self, rgb: Any) -> torch.Tensor:
        rgb = rgb.detach().permute(0, 2, 3, 1).to(self._device)
        finite_rgb = torch.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
        if bool((torch.amax(finite_rgb) <= 1.5).detach().cpu()):
            finite_rgb = finite_rgb * 255.0
        return torch.clamp(finite_rgb, 0.0, 255.0).to(torch.uint8)

    def _resolve_device(self, device: str) -> str:
        device = device or "cuda"
        if device.startswith("cuda") and not self._torch.cuda.is_available():
            return "cpu"
        return device

    @staticmethod
    def _import_torch():
        import torch

        return torch

    @staticmethod
    def _load_taxim_sim_package(sim_dir: Path):
        sim_dir = sim_dir.resolve()
        init_py = sim_dir / "__init__.py"
        if not init_py.is_file():
            raise FileNotFoundError(f"TacEx GPU-Taxim sim package not found: {init_py}")

        package_name = "_revosim_taxim_sim"
        existing = sys.modules.get(package_name)
        if existing is not None:
            return existing

        spec = importlib.util.spec_from_file_location(
            package_name,
            init_py,
            submodule_search_locations=[str(sim_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load TacEx GPU-Taxim sim package from: {init_py}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)
        return module
