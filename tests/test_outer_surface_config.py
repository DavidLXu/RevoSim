"""Check routing with real calibration resources, without starting Isaac Sim."""
import ast
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import numpy as np
import pytest
from revosim.resources import ASSETS, sensor_paths, model_manifest


@pytest.mark.parametrize('enabled', [True, False])
@pytest.mark.parametrize('side', ['left', 'right'])
def test_outer_reference_only_changes_depth_surface(monkeypatch, enabled, side):
    fake = ModuleType('isaaclab.sensors.ray_caster')
    fake.patterns = SimpleNamespace(GridPatternCfg=lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    backend = ModuleType('outer_config_test.backends.sharpa_tacmap_link_surface')
    class RayCfg(SimpleNamespace):
        RaycastTargetCfg = staticmethod(lambda **kw: SimpleNamespace(**kw))
        OffsetCfg = staticmethod(lambda **kw: SimpleNamespace(**kw))
    backend.SharpaTacmapLinkSurfaceCfg = RayCfg
    backend.SharpaTacmapLinkSurface = object
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    path = ASSETS.parent / 'tactile/backend_cfg.py'
    tree = ast.parse(path.read_text())
    # Execute the real builder; replace only Isaac's configuration classes.
    nodes = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    ns = dict(__package__='outer_config_test', Path=Path, np=np,
              json=__import__('json'), sensor_paths=sensor_paths, model_manifest=model_manifest)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    cfg = SimpleNamespace(asset_root=ASSETS/'sensors', prim_path='/World/Hand',
        target_prim_paths=('/World/Object',), pressure=False, pressure_diffusion=True,
        depth=True, marker=True, rgb=True, object_sample_mode='poisson', object_sample_count=1024,
        outer_surface_reference=enabled)
    children, _, _ = ns['build_hand'](cfg, side)
    assert len(children) == 20
    for name, child in children.items():
        outer = enabled and name.endswith('_tacmap_surface')
        assert child.surface_reference_mode == ('outer_forward' if outer else 'legacy')
        if outer:
            assert child.mesh_prim_paths[0].prim_expr == child.prim_path
        if name.endswith('_object'):
            assert child.ray_hit_index == 1


def test_outer_reference_enabled_by_default():
    tree = ast.parse((ASSETS.parent/'tactile/tactile_cfg.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    field = next(n for n in cls.body if isinstance(n, ast.AnnAssign)
                 and n.target.id == 'outer_surface_reference')
    assert ast.literal_eval(field.value) is True
