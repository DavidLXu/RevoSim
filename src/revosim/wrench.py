"""Six-axis contact wrench, including friction, about the pad's link origin."""

import numpy as np


def _contact_indices(counts, starts):
    counts = counts.cpu().numpy().copy()
    starts = starts.cpu().numpy().copy()
    return np.concatenate(
        [np.arange(int(s), int(s + c)) for s, c in zip(starts.ravel(), counts.ravel())]
    ), int(counts.sum())


def read_wrench(sensor, dt, origin, rotation):
    view = sensor.contact_physx_view
    fn, pn, nn, _, counts, starts = view.get_contact_data(dt)
    indices, count = _contact_indices(counts, starts)
    if count >= len(fn):
        raise RuntimeError("Contact detail buffer capacity exceeded")
    # Gather on the device before copying: most of the generously sized PhysX
    # buffer is empty. Keep every active contact, including cancelling forces.
    ids = indices.tolist()
    force = fn[ids].cpu().numpy().copy() * nn[ids].cpu().numpy().copy()
    points = pn[ids].cpu().numpy().copy()
    total = force.sum(0)
    torque = np.cross(points - origin, force).sum(0)
    expected = view.get_contact_force_matrix(dt).cpu().numpy().sum(axis=(0, 1))
    if not np.allclose(total, expected, rtol=2e-3, atol=2e-3):
        raise RuntimeError("Contact normal detail disagrees with force matrix")
    ft, pt, counts_t, starts_t = view.get_friction_data(dt)
    indices_t, count_t = _contact_indices(counts_t, starts_t)
    if count_t >= len(ft):
        raise RuntimeError("Contact detail buffer capacity exceeded")
    ids_t = indices_t.tolist()
    friction = ft[ids_t].cpu().numpy().copy()
    points_t = pt[ids_t].cpu().numpy().copy()
    total += friction.sum(0)
    torque += np.cross(points_t - origin, friction).sum(0)
    return np.r_[total @ rotation, torque @ rotation].astype(np.float32), count > 0
