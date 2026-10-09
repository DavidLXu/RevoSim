"""Experimental forward-only outer contact surface traversal.

Only use for a single isolated, consistently outward-wound rubber mesh.
Object rays retain nearest-hit semantics. Distances are metres.
"""
from __future__ import annotations
import math
import torch

SURFACE_REFERENCE_VERSION = "outer_forward_v1"

def raycast_outer_surface_forward(
    ray_starts, ray_directions, *, mesh_ids_wp, max_dist,
    mesh_positions_w=None, mesh_orientations_w=None,
    step_epsilon=1.0e-7, max_hits=32, raycast_fn=None,
):
    """Return (points, distances, normals, valid), with no first-hit fallback.

    Traverse from inside to outside and keep the last outward crossing.
    Exhausted traversal is an error, not a guessed reference surface.
    The caller must isolate the rubber contact mesh and validate max_dist
    spans its exterior. This routine does not redefine calibrated markers.
    """
    if not math.isfinite(max_dist) or max_dist <= 0:
        raise ValueError("max_dist must be finite and positive")
    if not math.isfinite(step_epsilon) or not 0 < step_epsilon < max_dist:
        raise ValueError("step_epsilon must lie within the ray interval")
    if max_hits < 2:
        raise ValueError("max_hits must be at least two")
    if ray_starts.shape != ray_directions.shape or ray_starts.shape[-1] != 3:
        raise ValueError("Ray starts/directions must have matching [...,3] shapes")
    if not torch.isfinite(ray_starts).all() or not torch.isfinite(ray_directions).all():
        raise ValueError("Nonfinite reference rays")
    lengths = torch.linalg.vector_norm(ray_directions, dim=-1, keepdim=True)
    if bool((lengths <= 1e-9).any()):
        raise ValueError("Zero reference ray direction")
    directions = ray_directions / lengths
    if raycast_fn is None:
        from isaaclab.utils.warp import raycast_dynamic_meshes
        raycast_fn = raycast_dynamic_meshes
    shape = ray_starts.shape[:-1]
    offset = torch.zeros(shape, dtype=ray_starts.dtype, device=ray_starts.device)
    best = torch.full_like(offset, torch.inf)
    points = torch.zeros_like(ray_starts)
    normals = torch.zeros_like(ray_starts)
    found = torch.zeros(shape, dtype=torch.bool, device=ray_starts.device)
    active = torch.ones_like(found)
    for _ in range(max_hits):
        current = ray_starts + directions * offset.unsqueeze(-1)
        hits, distance, normal, _, _ = raycast_fn(
            current, directions, mesh_ids_wp=mesh_ids_wp, max_dist=float(max_dist),
            mesh_positions_w=mesh_positions_w, mesh_orientations_w=mesh_orientations_w,
            return_distance=True, return_normal=True, return_mesh_id=False,
        )
        if distance is None or normal is None:
            raise RuntimeError("Outer surface traversal requires distance and normals")
        total = offset + distance
        hit = (active & torch.isfinite(distance) & (distance >= 0)
               & torch.isfinite(hits).all(-1) & torch.isfinite(normal).all(-1)
               & (total >= 0) & (total <= max_dist))
        outward = hit & ((normal * directions).sum(-1) > 0)
        best = torch.where(outward, total, best)
        points = torch.where(outward.unsqueeze(-1), hits, points)
        normals = torch.where(outward.unsqueeze(-1), normal, normals)
        found |= outward
        next_offset = torch.where(hit, total + step_epsilon, torch.full_like(total, max_dist))
        # Guarantee progress even if world-scale rounding swallows the epsilon.
        next_offset = torch.maximum(next_offset, torch.nextafter(total, torch.full_like(total, torch.inf)))
        active = hit & (next_offset < max_dist)
        offset = torch.where(active, next_offset, torch.full_like(offset, max_dist))
        if not bool(active.any()):
            break
    if bool(active.any()):
        raise RuntimeError("Outer surface traversal hit its limit; reference was not accepted")
    return points, best, normals, found
