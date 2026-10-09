import importlib.util
import numpy as np
from revosim.resources import ASSETS

spec = importlib.util.spec_from_file_location(
    "registration", ASSETS.parent / "tactile/surface_registration.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_embedded_taxel_projects_to_surface_without_changing_tangential_position():
    v = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float)
    f = np.array([[0, 1, 2], [0, 2, 3]])
    p = np.array([[0.2, 0.3, -0.001], [2, 2, -0.001]])
    n = np.array([[0, 0, 1], [0, 0, 1]])
    out, valid = m.ray_surface_points(p, n, v, f)
    np.testing.assert_allclose(out[0], [0.2, 0.3, 0])
    assert valid.tolist() == [True, False]
