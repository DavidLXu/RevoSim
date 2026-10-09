"""Small public API for tactile hands. Import only after AppLauncher starts."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from revosim.scene import create_hand, create_object
from revosim.resources import (
    SIDES,
    FINGERS,
    BACKEND_ORDER,
    sensor_paths,
    pressure_regions,
)
from revosim.tactile import make_tactile_cfg, TactileSensor
from revosim.wrench import read_wrench


class World:
    """Two sensor-equipped hands observing one or more rigid targets.

    ``step`` advances one 1/60-s observation interval (four 240-Hz physics steps).
    ``observe`` returns owned NumPy arrays; data are safe to retain or modify.
    Object motion uses applied forces, except explicit ``reset_object`` calls.
    """

    def __init__(
        self,
        shape="sphere",
        device="cuda:0",
        physics_hz=240,
        observation_hz=60,
        hand_spacing=0.26,
    ):
        if physics_hz % observation_hz:
            raise ValueError("physics_hz must be divisible by observation_hz")
        self.device = device
        self.dt = 1 / physics_hz
        self.obs_dt = 1 / observation_hz
        self.decimation = physics_hz // observation_hz
        shapes = (shape,) if isinstance(shape, str) else tuple(shape)
        if not shapes or len(set(shapes)) != len(shapes):
            raise ValueError("Specify one or more unique object shapes")
        self.shape = " + ".join(shapes)
        self.hand_spacing = hand_spacing
        self.time = 0.0
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=self.dt,
                render_interval=self.decimation,
                device=device,
                use_fabric=True,
            )
        )
        self.hands = {}
        self.sensors = {}
        self.wrench_sensors = {}
        self.physics_manifest = {}
        self.calibration = {}
        for side in SIDES:
            self.hands[side], self.physics_manifest[side] = create_hand(
                self.sim.stage,
                side,
                translation=(
                    (-1 if side == "left" else 1) * hand_spacing / 2,
                    -0.12,
                    0.18,
                ),
            )
        paths = [
            "/World/Object" if len(shapes) == 1 else f"/World/Object_{s}"
            for s in shapes
        ]
        self.objects = {}
        self.meshes = {}
        for i, (name, path) in enumerate(zip(shapes, paths)):
            self.objects[name], self.meshes[name] = create_object(
                self.sim.stage, name, path, position=(i * 0.06, 0.20, 0.50)
            )
        self.object = self.objects[shapes[0]]
        self.mesh = self.meshes[shapes[0]]
        self._targets = {name: None for name in shapes}
        for side in SIDES:
            path = f"/World/{side.capitalize()}"
            self.sensors[side] = (
                TactileSensor(
                    make_tactile_cfg(
                        robot_prim_path=path,
                        target_prim_paths=(paths[0],),
                        hands=(side,),
                        object_sample_count=8192,
                    )
                )
                if len(shapes) == 1
                else self._multi_sensor(path, side, paths)
            )
            self.wrench_sensors[side] = [
                ContactSensor(
                    ContactSensorCfg(
                        prim_path=f"{path}/{side}_{f}_DIP_rubber_link",
                        filter_prim_paths_expr=paths,
                        max_contact_data_count_per_prim=4096,
                    )
                )
                for f in FINGERS
            ]
            with np.load(sensor_paths(side)["vtac"] / "marker_positions.npz") as npz:
                self.calibration[side] = {k: npz[k].copy() for k in npz.files}
        light = sim_utils.DomeLightCfg(intensity=225, visible_in_primary_ray=False)
        light.func("/World/Ambient", light)
        ground = sim_utils.CuboidCfg(
            size=(3, 3, 0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.055, 0.07, 0.09), roughness=0.95
            ),
        )
        ground.func("/World/Ground", ground, translation=(0, 0, -0.01))
        self.sim.reset()
        for hand in self.hands.values():
            hand.reset()
            q = hand.data.default_joint_pos.clone()
            hand.write_joint_state_to_sim(q, torch.zeros_like(q))
            hand.set_joint_position_target(q)
        for obj in self.objects.values():
            obj.reset()
        self.target = None
        for _ in range(8):
            self.step()
        self.initial_poses = {
            s: h.data.body_link_state_w[0, :, :7].cpu().numpy().copy()
            for s, h in self.hands.items()
        }
        self.body_ids = {
            s: [h.body_names.index(f"{s}_{f}_DIP_rubber_link") for f in FINGERS]
            for s, h in self.hands.items()
        }

    @staticmethod
    def _multi_sensor(path, side, targets):
        from revosim.tactile.multi_target import MultiTargetTactile

        return MultiTargetTactile(path, side, targets)

    @property
    def target(self):
        return self._targets[next(iter(self.objects))]

    @target.setter
    def target(self, value):
        self._targets[next(iter(self.objects))] = value

    def reset_object(self, position=(0, 0, 0.5), velocity=(0, 0, 0), object_name=None):
        name = object_name or next(iter(self.objects))
        obj = self.objects[name]
        pose = torch.tensor(
            [[*position, 1, 0, 0, 0]], device=self.device, dtype=torch.float32
        )
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(
            torch.tensor(
                [[*velocity, 0, 0, 0]], device=self.device, dtype=torch.float32
            )
        )
        for s in self.sensors.values():
            s.reset()
        self._targets[name] = None

    def drive_object(self, position, object_name=None):
        """Attach a virtual spring to the target point, for a reproducible probe demo."""
        self._targets[object_name or next(iter(self.objects))] = np.asarray(
            position, dtype=np.float32
        )

    def release_object(self, velocity=None):
        self.target = None
        if velocity is not None:
            self.object.write_root_velocity_to_sim(
                torch.tensor(
                    [[*velocity, 0, 0, 0]], device=self.device, dtype=torch.float32
                )
            )

    def set_joints(self, side, values):
        """Set selected named joint targets in radians (PhysX PD drives)."""
        hand = self.hands[side]
        q = hand.data.joint_pos.clone()
        for name, value in values.items():
            i = hand.joint_names.index(name)
            lo, hi = hand.data.joint_pos_limits[0, i]
            if not float(lo) <= value <= float(hi):
                raise ValueError(f"{name} outside limits")
            q[0, i] = value
        hand.set_joint_position_target(q)

    def step(self, render=False):
        for _ in range(self.decimation):
            for name, obj in self.objects.items():
                force = torch.zeros((1, 1, 3), device=self.device)
                target = self._targets[name]
                if target is not None:
                    error = (
                        torch.as_tensor(target, device=self.device)
                        - obj.data.root_pos_w[0]
                    )
                    f = (
                        180.0 * error
                        - 3.8 * obj.data.root_lin_vel_w[0]
                        + torch.tensor([0, 0, 0.02 * 9.81], device=self.device)
                    )
                    force[0, 0] = f * torch.clamp(
                        3.0 / f.norm().clamp_min(1e-6), max=1.0
                    )
                obj.instantaneous_wrench_composer.set_forces_and_torques(
                    forces=force, torques=torch.zeros_like(force), is_global=True
                )
                obj.write_data_to_sim()
            for hand in self.hands.values():
                hand.write_data_to_sim()
            self.sim.step(render=False)
            for hand in self.hands.values():
                hand.update(self.dt)
            for obj in self.objects.values():
                obj.update(self.dt)
        for side in SIDES:
            self.sensors[side].update(self.obs_dt)
            for sensor in self.wrench_sensors[side]:
                sensor.update(self.obs_dt)
        self.time += self.obs_dt
        if render:
            self.sim.render()

    def surface_point(self, side, finger="index", region="tip"):
        """Return an actual registered sample position and outward normal in world metres."""
        hand = self.hands[side]
        if region == "tip":
            prefix = "pinky" if finger == "little" else finger
            d = self.calibration[side]
            valid = (
                d["distortion_valid"]
                & (d[prefix + "_method"] == "ray_hit")
                & d[prefix + "_marker_ray_valid"]
            )
            points = d[prefix + "_points_link_m"][valid]
            normals = d[prefix + "_normals_link"][valid]
            i = np.argmin(np.linalg.norm(points - points.mean(0), axis=1))
            point = points[i]
            normal = normals[i]
            link = f"{side}_{finger}_DIP_rubber_link"
        else:
            key = (
                f"{side}_plam_touch_link_pressure"
                if region == "palm"
                else f"{side}_{finger}_{region.upper()}_touch_link_pressure"
            )
            child = self.sensors[side]._children[side][key]
            points = child._points_local_per_sensor[0].cpu().numpy()
            normals = np.array(child.cfg.taxel_normals_l)
            # The palm is C-shaped: its centroid lies beside the hard thumb
            # recess. Choose the interior of its broad upper pad, so a finite
            # probe contacts skin instead of bridging across that recess.
            if region == "palm":
                distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
                density = np.exp(-0.5 * (distances / 0.007) ** 2).sum(axis=1)
                i = int(np.argmax(density))
            else:
                i = np.argmin(np.linalg.norm(points - points.mean(0), axis=1))
            point = points[i]
            normal = normals[i]
            link = child.cfg.elastomer_prim_paths[0].split("/")[-1]
        state = (
            hand.data.body_link_state_w[0, hand.body_names.index(link), :7]
            .cpu()
            .numpy()
        )
        rot = Rotation.from_quat(state[[4, 5, 6, 3]]).as_matrix()
        normal = rot @ normal
        normal /= np.linalg.norm(normal)
        return state[:3] + rot @ point, normal

    def observe(self):
        values = {
            k: []
            for k in (
                "rgb",
                "depth",
                "marker",
                "pressure",
                "wrench",
                "contact",
                "pad_pose",
                "joint_pos",
            )
        }
        for side in SIDES:
            sensor = self.sensors[side]
            hand = self.hands[side]
            values["pressure"].append(sensor.get_pressure()[0, 0].cpu().numpy().copy())
            values["depth"].append(
                sensor.get_depth()[0, 0, list(BACKEND_ORDER)].cpu().numpy().copy()
            )
            values["rgb"].append(
                sensor.get_rgb()[0, 0, list(BACKEND_ORDER)].cpu().numpy().copy()
            )
            marker = (
                sensor.get_marker_displacement()[0, 0, list(BACKEND_ORDER)]
                .cpu()
                .numpy()
                .copy()
            )
            poses = (
                hand.data.body_link_state_w[0, self.body_ids[side], :7]
                .cpu()
                .numpy()
                .copy()
            )
            rot = Rotation.from_quat(poses[:, [4, 5, 6, 3]]).as_matrix()
            values["marker"].append(
                np.einsum("fni,fij->fnj", marker, rot).astype(np.float32)
            )
            values["pad_pose"].append(poses)
            w = []
            c = []
            for i, contact_sensor in enumerate(self.wrench_sensors[side]):
                wi, ci = read_wrench(contact_sensor, self.dt, poses[i, :3], rot[i])
                w.append(wi)
                c.append(ci)
            values["wrench"].append(w)
            values["contact"].append(c)
            values["joint_pos"].append(hand.data.joint_pos[0].cpu().numpy().copy())
        values = {k: np.asarray(v) for k, v in values.items()}
        values["time"] = self.time
        values["object_pose"] = (
            self.object.data.root_state_w[0, :7].cpu().numpy().copy()
        )
        values["object_poses"] = np.stack(
            [
                obj.data.root_state_w[0, :7].cpu().numpy().copy()
                for obj in self.objects.values()
            ]
        )
        if len(self.objects) > 1:
            owners = np.stack(
                [
                    self.sensors[side]
                    ._owners()[0, 0, list(BACKEND_ORDER)]
                    .cpu()
                    .numpy()
                    for side in SIDES
                ]
            )
            values["optical_object_index"] = np.where(
                values["depth"].max(axis=(-2, -1)) > 1e-5, owners, -1
            )
        for key, v in values.items():
            if not np.isfinite(v).all():
                raise RuntimeError(f"Nonfinite {key}")
        return values

    def sensor_layout(self):
        """JSON-compatible channel locations, link names, joint order and limits.

        Pressure positions are the runtime surface-registered sampling points,
        in the indicated link frame; channel indices match ``observe`` exactly.
        """
        layout = {"sides": list(SIDES), "fingers": list(FINGERS), "hands": {}}
        for side in SIDES:
            hand = self.hands[side]
            channels = []
            for name, sl in pressure_regions().items():
                link = (
                    f"{side}_plam_touch_link"
                    if name == "palm"
                    else f"{side}_{name.rsplit('_', 1)[0]}_{name.rsplit('_', 1)[1].upper()}_touch_link"
                )
                child = self.sensors[side]._children[side][link + "_pressure"]
                points = child._points_local_per_sensor[0].cpu().numpy()
                normals = np.asarray(child.cfg.taxel_normals_l)
                channels.append(
                    {
                        "region": name,
                        "start": sl.start,
                        "stop": sl.stop,
                        "link": child.cfg.elastomer_prim_paths[0].split("/")[-1],
                        "points_local_m": points.tolist(),
                        "normals_local": normals.tolist(),
                    }
                )
            layout["hands"][side] = {
                "joint_names": list(hand.joint_names),
                "joint_limits_rad": hand.data.joint_pos_limits[0]
                .cpu()
                .numpy()
                .tolist(),
                "pressure_regions": channels,
                "optical_links": [f"{side}_{f}_DIP_rubber_link" for f in FINGERS],
            }
        return layout
