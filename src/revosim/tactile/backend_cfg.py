"""Adapt local calibration resources to the standalone Revo3 link names."""
import json
from pathlib import Path
import numpy as np
from revosim.resources import sensor_paths
from revosim.resources import model_manifest

FINGERS = ('middle', 'index', 'ring', 'pinky', 'thumb')
STEMS = ('mid', 'index', 'ring', 'pinky', 'thumb')


def build_hand(cfg, side):
    root = Path(cfg.asset_root)
    paths = sensor_paths(side, root)
    mapping = json.loads(paths['mapping'].read_text())['hands'][side]['links']
    manifest = model_manifest()
    pressure = manifest['pressure_presets']
    vtac = manifest['vtac_presets']
    diffusion = pressure['diffusion']
    robot = cfg.prim_path
    namespace = robot.rsplit('/', 1)[0]
    target = cfg.target_prim_paths[0].replace('{ENV_REGEX_NS}', namespace)
    children = {}
    components = dict(pressure_sensor_names=[], pressure_taxel_counts=[],
                      pressure_diffusion_enabled=cfg.pressure_diffusion,
                      pressure_diffusion_sigma_m=diffusion['sigma_m'], pressure_diffusion_blend=diffusion['blend'],
                      pressure_diffusion_radius_sigma=diffusion['radius_sigma'], pressure_diffusion_normal_power=diffusion['normal_power'])
    if cfg.pressure:
        from .backends.urdf_pressure_layout import load_pressure_pad_specs_from_urdf
        from .backends.warp_sdf_tactile_cfg import WarpSdfTactileSensorCfg
        specs = load_pressure_pad_specs_from_urdf(paths['pressure'], require_files=False)
        # The left snapshot is already mirrored, including origins and normals.
        for spec in specs:
            grid = spec.to_taxel_map()
            normals = np.asarray(grid.normals_l).copy()
            # The inherited MCP/PIP mesh normals point into the finger. Keep the
            # original resources intact and orient their runtime sampling offset outward.
            if '_hand_touch_link' not in spec.link_name:
                normals *= -1
            from .surface_registration import register_pad
            points = register_pad(robot, mapping[spec.link_name], np.asarray(grid.points_l), normals)
            name = mapping[spec.link_name] + '_pressure'
            children[name] = WarpSdfTactileSensorCfg(
                prim_path=robot, elastomer_prim_paths=[f'{robot}/{mapping[spec.link_name]}'],
                num_rows=grid.image_shape[0], num_cols=grid.image_shape[1],
                taxel_points_l=points.tolist(), taxel_normals_l=normals.tolist(),
                normal_axis=spec.normal_axis, normal_sign=spec.normal_sign, normal_offset=0.0,
                target_mesh_prim_path=target, mesh_max_dist=0.2, mesh_use_signed_distance=True,
                mesh_signed_distance_method='winding', stiffness=spec.calibration.stiffness,
                damping=spec.calibration.damping, max_force=spec.calibration.max_force,
                pressure_gain=spec.calibration.gain, pressure_bias=spec.calibration.bias,
                pressure_gamma=spec.calibration.gamma, pressure_threshold=spec.calibration.threshold,
                taxel_area=spec.calibration.area, normalize_forces=True, store_debug_fields=False)
            components['pressure_sensor_names'].append(name)
            components['pressure_taxel_counts'].append(int(np.prod(grid.image_shape)))
    if cfg.depth or cfg.marker or cfg.rgb:
        from isaaclab.sensors.ray_caster import patterns
        from .backends.sharpa_tacmap_link_surface import SharpaTacmapLinkSurface, SharpaTacmapLinkSurfaceCfg as RayCfg
        grids = json.loads(paths['defaults'].read_text())
        layout = paths['vtac']/'marker_positions.npz'
        ray_rows, ray_cols = vtac['depth_ray_shape']
        reference_rows, reference_cols = vtac['reference_shape']
        marker_rows, marker_cols = vtac['marker_shape']
        local_rows, local_cols = vtac['local_shape']
        components.update(
            tacmap_sensor_names=[], tacmap_surface_sensor_names=[], hydroshear_marker_sensor_names=[],
            hydroshear_marker_surface_sensor_names=[], hydroshear_marker_finger_link_names=[],
            tacmap_rows=ray_rows, tacmap_cols=ray_cols, local_tacmap_enabled=True,
            local_tacmap_reference_rows=reference_rows, local_tacmap_reference_cols=reference_cols,
            local_tacmap_rows=local_rows, local_tacmap_cols=local_cols, local_tacmap_contact_threshold_m=vtac['contact_threshold_m'],
            local_tacmap_roi_margin_m=vtac['roi_margin_m'], local_tacmap_penetration_deadband_m=vtac['penetration_deadband_m'],
            local_tacmap_max_distance_m=vtac['max_distance_m'], tacmap_policy_full_depth_image=True,
            taxim_rgb_render_rows=reference_rows, taxim_rgb_render_cols=reference_cols,
            taxim_rgb_background_path=str(paths['background']),
            hydroshear_render_rows=reference_rows, hydroshear_render_cols=reference_cols,
            hydroshear_marker_rows=marker_rows, hydroshear_marker_cols=marker_cols,
            hydroshear_marker_layout_path=str(layout), hydroshear_render_debug_marker_images=False,
            hydroshear_object_sample_mode=cfg.object_sample_mode, hydroshear_object_sample_count=cfg.object_sample_count,
            hydroshear_object_sample_roi_count=vtac['object_sample_roi_count'], hydroshear_use_object_surface_samples=True)
        for finger, stem in zip(FINGERS, STEMS):
            source_name = f'{side}_{stem}dip_roll_rubber_link'
            link = mapping[source_name]
            components['hydroshear_marker_finger_link_names'].append(link)
            params = grids[source_name.replace('left_', 'right_', 1)].copy()
            if side == 'left':
                params['grid_center'][1] *= -1
                for axis in ('ray_axis', 'grid_u_axis', 'grid_v_axis'):
                    if params[axis].endswith('y'):
                        params[axis] = ('-' if params[axis].startswith('+') else '+')+'y'
            for group, rows, cols, prefix in [('tacmap', ray_rows, ray_cols, finger)] + (
                    [('hydroshear_marker', marker_rows, marker_cols, finger+'_marker')] if cfg.marker else []):
                for kind in ('object', 'surface'):
                    name = f'{link}_{group}_{kind}'
                    component_key = group + ('_sensor_names' if kind == 'object' else '_surface_sensor_names')
                    components[component_key].append(name)
                    children[name] = RayCfg(
                        class_type=SharpaTacmapLinkSurface, prim_path=f'{robot}/{link}',
                        mesh_prim_paths=[RayCfg.RaycastTargetCfg(prim_expr=target if kind == 'object' else f'{robot}/{link}',
                                                               track_mesh_transforms=True)],
                        pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=(0.5, 0.5)),
                        offset=RayCfg.OffsetCfg(convention='world'), resolution_step=1,
                        # This backend uses explicit NPZ rays; dense surface NPYs are unnecessary.
                        points_npy='', normals_npy='',
                        data_types=['distance_along_normal', 'distance_along_normal_raw'],
                        max_distance=0.05, image_height=rows, image_width=cols,
                        ray_layout_npz=str(layout), ray_layout_prefix=prefix,
                        surface_reference_mode=('outer_forward' if cfg.outer_surface_reference and kind == 'surface' and group == 'tacmap' else 'legacy'),
                        ray_hit_index=1 if kind == 'object' else 2,
                        use_ray_hit_index_layout=kind == 'surface', use_first_hit_fallback=kind == 'surface',
                        debug_viz_rays=False, **{k: v for k, v in params.items() if k != 'max_distance'})
    return children, components, target
