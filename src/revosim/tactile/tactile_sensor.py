"""Isaac Lab sensor facade over the four copied tactile computation paths."""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
import math
import weakref
import torch

from isaaclab.sensors import SensorBase

from .cache import DemandCache
from .tactile_data import TactileSensorData


class _Scene(dict):
    def __init__(self, sensors):
        super().__init__()
        self.sensors = sensors


class TactileSensor(SensorBase):
    """Lazy outputs: pressure[N,H,247], depth[N,H,5,240,320],
    marker displacement[N,H,5,100,3] in world-frame meters, RGB[N,H,5,3,240,320].

    H follows cfg.hands. Fingers follow middle,index,ring,little,thumb.
    Pressure is the donor's normalized, diffused response (not Pascals).
    Output tensors are borrowed read-only buffers; clone them before mutation.
    Call update(dt) once per observation interval (InteractiveScene does this).
    """

    def __init__(self, cfg):
        if not cfg.hands or len(set(cfg.hands)) != len(cfg.hands) or set(cfg.hands) - {'left', 'right'}:
            raise ValueError('hands must be a nonempty, unique subset of left/right')
        if len(cfg.target_prim_paths) != 1:
            raise ValueError('This backend supports one rigid target per environment; specify exactly one target path')
        if cfg.marker_history not in ('on_demand', 'continuous'):
            raise ValueError('marker_history must be on_demand or continuous')
        if cfg.update_period != 0 or cfg.history_length != 0:
            raise ValueError('Use update_period=0 and history_length=0; drive the sampling rate through update(dt)')
        super().__init__(cfg)
        self._cache = DemandCache(name for name in ('pressure', 'depth', 'marker', 'rgb') if getattr(cfg, name))
        self._data = TactileSensorData(self)
        self._children = {}
        self._components = {}
        self._contexts = {}
        self._kernel_calls = Counter()
        self._last_marker = {}
        self._preserved = {}
        from .backend_cfg import build_hand
        for side in cfg.hands:
            configs, component, target = build_hand(self.cfg, side)
            self._components[side] = component
            self._target_path = target
            children = {}
            for name, child_cfg in configs.items():
                child = child_cfg.class_type(child_cfg)
                original = child._update_buffers_impl
                owner = weakref.ref(self)
                def counted(ids, _original=original, _name=name, _owner=owner):
                    parent = _owner()
                    if parent is not None:
                        parent._kernel_calls[_name] += 1
                    return _original(ids)
                child._update_buffers_impl = counted
                children[name] = child
            self._children[side] = children

    @property
    def data(self):
        return self._data

    @property
    def diagnostics(self):
        return {'step': self._cache.step, 'getters_computed': dict(self._cache.calls),
                'backend_updates': dict(self._kernel_calls)}

    def _initialize_impl(self):
        super()._initialize_impl()
        self._cache.invalidate()
        self._preserved.clear()
        import omni.physics.tensors.impl.api as physx
        import isaaclab.sim as sim_utils
        from pxr import Usd, UsdPhysics
        self._physics = physx.create_simulation_view(self._backend)
        self._physics.set_subspace_roots('/')
        first = sim_utils.find_first_matching_prim(self.cfg.prim_path, stage=self.stage)
        roots = [p for p in Usd.PrimRange(first) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(roots) != 1:
            raise RuntimeError('Expected one articulation root under the configured robot')
        suffix = str(roots[0].GetPath())[len(str(first.GetPath())):]
        expression = self.cfg.prim_path + suffix
        self._robot_view = self._physics.create_articulation_view(expression.replace('.*', '*'))
        self._object_view = self._physics.create_rigid_body_view(self._target_path.replace('.*', '*'))
        if self._robot_view.count != self._num_envs or self._object_view.count != self._num_envs:
            raise RuntimeError('Robot/target paths must resolve to exactly one articulation/rigid body per environment')
        robot_name = self.cfg.prim_path.rsplit('/', 1)[1]
        robot_roots = [path.split('/'+robot_name)[0] for path in self._robot_view.prim_paths]
        target_roots = [path.rsplit('/', 1)[0] for path in self._object_view.prim_paths]
        if robot_roots != target_roots:
            raise RuntimeError('Robot and target view environment ordering differs')
        self._reset_mask = torch.ones(self._num_envs, dtype=torch.bool, device=self.device)
        for side in self.cfg.hands:
            scene = _Scene(self._children[side])
            scene['robot'] = SimpleNamespace(body_names=self._robot_view.shared_metatype.link_names, data=SimpleNamespace())
            scene['object'] = SimpleNamespace(cfg=SimpleNamespace(prim_path=self._target_path), data=SimpleNamespace())
            self._contexts[side] = SimpleNamespace(
                scene=scene, num_envs=self._num_envs, device=self.device, common_step_counter=0,
                episode_length_buf=torch.zeros(self._num_envs, device=self.device, dtype=torch.long))
            self._last_marker[side] = torch.full((self._num_envs,), -100, device=self.device, dtype=torch.long)

    def _invalidate_initialize_callback(self, event):
        super()._invalidate_initialize_callback(event)
        self._cache.invalidate()
        self._preserved.clear()
        self._contexts.clear()
        self._robot_view = self._object_view = self._physics = None

    def update(self, dt, force_recompute=False):
        if not math.isfinite(dt) or dt < 0:
            raise ValueError('dt must be finite and nonnegative')
        if not self.is_initialized:
            raise RuntimeError('Initialize the simulation before updating tactile sensors')
        self._cache.advance()
        self._preserved.clear()
        self._reset_mask.zero_()
        for side, children in self._children.items():
            context = self._contexts[side]
            context.common_step_counter = self._cache.step
            context.episode_length_buf += 1
            for child in children.values():
                child.update(dt, force_recompute=False)
        if self.cfg.marker and self.cfg.marker_history == 'continuous':
            self.get_marker_displacement()
        # InteractiveScene may pass force_recompute when lazy_sensor_update=False.
        # Keep this composite sensor demand-driven: it has four independent caches.

    def reset(self, env_ids=None):
        if not self.is_initialized:
            return
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._reset_mask[ids] = True
        self._preserved.update(self._cache.values)
        self._cache.invalidate()
        from .backends.observations import invalidate_ours_tactile_cache_on_reset
        for side, context in self._contexts.items():
            context.episode_length_buf[ids] = 0
            self._last_marker[side][ids] = -100
            invalidate_ours_tactile_cache_on_reset(context, env_ids)
            for child in self._children[side].values():
                child.reset(env_ids)

    def _refresh_poses(self):
        body = self._robot_view.get_link_transforms()
        body = torch.cat((body[..., :3], body[..., 6:7], body[..., 3:6]), dim=-1)
        target = self._object_view.get_transforms()
        for context in self._contexts.values():
            context.scene['robot'].data.body_link_state_w = body
            context.scene['object'].data.root_pos_w = target[:, :3]
            context.scene['object'].data.root_quat_w = target[:, [6, 3, 4, 5]]

    def _compute_marker(self, side, obs):
        context = self._contexts[side]
        last = self._last_marker[side]
        context.episode_length_buf = torch.where(last < self._cache.step - 1, 0, 1)
        # Preserve unaffected environments when a partial reset triggers another
        # read within the same frame. Only reset environments should change history.
        repeated = (last == self._cache.step).repeat_interleave(5)
        adapter = getattr(context, '_brainco_rl_hydroshear_adapter', None)
        saved = {}
        if adapter is not None and repeated.any():
            for name, value in vars(adapter).items():
                if name.startswith('_batch_') and isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == len(repeated):
                    saved[name] = value[repeated].clone()
        value = obs.ours_rl_hydroshear_obs(context, self._components[side])
        if saved:
            for name, previous in saved.items():
                getattr(adapter, name)[repeated] = previous
        self._last_marker[side].fill_(self._cache.step)
        return value.reshape(self._num_envs, 5, 100, 3)

    def _read(self, modality):
        if not self.is_initialized:
            raise RuntimeError('Call sim.reset() before reading tactile observations')
        def compute():
            from .backends import observations as obs
            self._refresh_poses()
            results = []
            for side in self.cfg.hands:
                ctx, component = self._contexts[side], self._components[side]
                if modality == 'pressure':
                    value = obs.ours_rl_pressure_obs(ctx, component)
                elif modality == 'depth':
                    value = obs.ours_rl_tacmap_policy_obs(ctx, component).reshape(self._num_envs, 5, 240, 320)
                elif modality == 'marker':
                    value = self._compute_marker(side, obs)
                else:
                    value = obs.ours_rl_taxim_rgb_obs(ctx, component).reshape(self._num_envs, 5, 3, 240, 320)
                if not torch.isfinite(value).all():
                    raise RuntimeError(f'Non-finite tactile {modality} data for {side}')
                results.append(value)
            result = torch.stack(results, dim=1)
            previous = self._preserved.get(modality)
            if previous is not None:
                result[~self._reset_mask] = previous[~self._reset_mask]
            return result
        return self._cache.get(modality, compute)

    def get_pressure(self):
        return self._read('pressure')

    def get_depth(self):
        return self._read('depth')

    def get_marker_displacement(self):
        return self._read('marker')

    def get_rgb(self):
        return self._read('rgb')

    def _update_buffers_impl(self, env_ids):
        # There is no eager composite observation; each getter owns its cache.
        pass
