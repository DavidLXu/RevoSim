# Third-Party Notices

This repository is released under the MIT License, except where individual
files retain third-party copyright or SPDX notices.

This repository includes files derived from or structured after the Isaac Lab project.

- Isaac Lab
  License: BSD-3-Clause
  Source: https://github.com/isaac-sim/IsaacLab
  Notes: Some files in `source/BrainCo_DexHand/` retain upstream copyright headers and SPDX identifiers.

- FOTS / TacEx marker motion
  License: MIT (TacEx repository notice)
  Source: https://github.com/Rancho-zhao/FOTS and TacEx's FOTS marker simulator
  Notes: `integrate/fots_adapter.py` adapts the marker motion equations for RevoLab TacMap raw depth.

- TacEx GPU-Taxim renderer and GelSight Mini calibration subset
  License: MIT for the GPU-Taxim simulator files; calibration data is copied from TacEx's GelSight Mini assets.
  Source: TacEx GPU-Taxim `sim` package and TacEx GelSight Mini `calibs/640x480`.
  Notes: Vendored under `integrate/third_party/tacex_gpu_taxim/` so TacEx RGB rendering does not require an external `/home/abc/TacEx` checkout.

Where third-party copyright or SPDX headers are present in individual files, those notices remain authoritative for the corresponding file history.
