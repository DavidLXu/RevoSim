"""Project marker vectors into the calibrated camera plane for diagnostics.

This is an orthographic metric view, not distorted image optical flow. Keep the
original marker index ordering; the supplied point coordinates place each dot.
"""
import numpy as np


def project_marker_camera_xy(displacement_w, link_quat_wxyz, points_camera_m, camera_rotation_link):
    """Return camera-plane origins and displacement vectors in metres.

    ``camera_rotation_link`` maps camera coordinates into link coordinates. The
    left-hand calibration contains a mirrored basis (det=-1); use the matrix
    directly instead of converting it into a quaternion.
    """
    vectors = np.asarray(displacement_w, dtype=np.float64)
    points = np.asarray(points_camera_m, dtype=np.float64)
    camera_to_link = np.asarray(camera_rotation_link, dtype=np.float64)
    q = np.asarray(link_quat_wxyz, dtype=np.float64)
    if vectors.shape != points.shape or vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError('Marker points and vectors must both have shape [M,3]')
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('Expected a finite, nonzero wxyz quaternion')
    if camera_to_link.shape != (3, 3) or not np.allclose(camera_to_link.T@camera_to_link, np.eye(3), atol=1e-5):
        raise ValueError('Camera calibration must be an orthogonal 3x3 basis')
    w, x, y, z = q / np.linalg.norm(q)
    link_to_world = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    camera_vectors = vectors @ link_to_world @ camera_to_link
    return points[:, :2].copy(), camera_vectors[:, :2]
