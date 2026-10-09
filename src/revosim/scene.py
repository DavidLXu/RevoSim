"""Build two standalone articulated hands; no arm, planner, task or benchmark imports."""

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from revosim.resources import ASSETS


def create_hand(stage, side, translation=None, orientation=(0.5, -0.5, -0.5, -0.5)):
    path = f"/World/{side.capitalize()}"
    cfg = sim_utils.UsdFileCfg(
        usd_path=str(ASSETS / f"hands/revo3_{side}_bionic_silver.usda")
    )
    cfg.func(
        path,
        cfg,
        translation=translation
        if translation is not None
        else (-0.13 if side == "left" else 0.13, -0.12, 0.18),
        orientation=orientation,
    )
    root = stage.GetPrimAtPath(path)
    roots = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    ar = roots[0] if roots else root
    UsdPhysics.ArticulationRootAPI.Apply(ar)
    api = PhysxSchema.PhysxArticulationAPI.Apply(ar)
    api.CreateEnabledSelfCollisionsAttr(False)
    api.CreateSolverPositionIterationCountAttr(16)
    api.CreateSolverVelocityIterationCountAttr(4)
    # Soft pads and hard backing use the same high-fidelity geometry, with explicit
    # SDF shells and convex pad approximations. Source USDs are never edited.
    regularized = []
    collisions = []
    for prim in list(Usd.PrimRange(root)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0)
            mass = UsdPhysics.MassAPI(prim)
            if not mass or not mass.GetMassAttr().HasAuthoredValueOpinion():
                mass = UsdPhysics.MassAPI.Apply(prim)
                mass.CreateMassAttr(1e-6)
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-9))
                regularized.append(prim.GetName())
        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(5.0)
            drive.CreateDampingAttr(0.15)
            drive.CreateMaxForceAttr(2.0)
        if prim.IsA(UsdGeom.Mesh):
            name = str(prim.GetPath())
            soft = "_rubber_link/" in name or "_tubber_link/" in name
            shell = any(
                token in name
                for token in ("_plam_link/", "_MCP_link/", "_PIP_link/", "_DIP_link/")
            )
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                "sdf"
                if shell
                else "convexHull"
                if soft and "plam_" not in name
                else "convexDecomposition"
            )
            if shell:
                PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(
                    prim
                ).CreateSdfResolutionAttr(256)
            col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            col.CreateContactOffsetAttr(0.0002)
            col.CreateRestOffsetAttr(
                -0.0001
                if "DIP" in name
                else -0.0017
                if soft or "_touch_link/" in name
                else 0.0
            )
            mat_path = "/World/PhysicsMaterials/" + ("Soft" if soft else "Rigid")
            if not stage.GetPrimAtPath(mat_path):
                mat = sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.7,
                    dynamic_friction=0.55,
                    restitution=0.1,
                    compliant_contact_stiffness=280.0 if soft else 0.0,
                    compliant_contact_damping=20.0 if soft else 0.0,
                )
                mat.func(mat_path, mat)
            sim_utils.bind_physics_material(str(prim.GetPath()), mat_path, stage=stage)
            collisions.append(name)
    robot = Articulation(
        ArticulationCfg(
            prim_path=path,
            spawn=None,
            actuators={
                "fingers": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=5.0,
                    damping=0.15,
                    effort_limit_sim=2.0,
                    velocity_limit_sim=3.0,
                    armature=0.00005,
                )
            },
        )
    )
    return robot, dict(collisions=collisions, mass_regularized=regularized)


def create_object(stage, shape, path="/World/Object", position=(0, 0, 0.5)):
    import trimesh

    if shape == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.007)
    elif shape == "cube":
        mesh = trimesh.creation.box(extents=(0.014, 0.014, 0.014))
    elif shape == "polyhedron":
        vertices = (
            np.array(
                [
                    [-1, -0.7, -0.6],
                    [1, -1, -0.8],
                    [0.7, 1, -1],
                    [-0.8, 0.9, -0.5],
                    [-0.6, -0.7, 0.9],
                    [0.8, -0.6, 1],
                    [1, 0.8, 0.5],
                    [-1, 1, 0.8],
                ]
            )
            * 0.008
        )
        mesh = trimesh.convex.convex_hull(vertices)
    else:
        raise ValueError(shape)
    root = stage.DefinePrim(path, "Xform")
    UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(*position))
    geom = UsdGeom.Mesh.Define(stage, path + "/Mesh")
    geom.CreatePointsAttr(mesh.vertices.tolist())
    geom.CreateFaceVertexIndicesAttr(mesh.faces.ravel().tolist())
    geom.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    geom.CreateSubdivisionSchemeAttr("none")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(0.02)
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr(
        "convexHull"
    )
    col = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
    col.CreateContactOffsetAttr(0.0001)
    col.CreateRestOffsetAttr(0)
    mat = sim_utils.PreviewSurfaceCfg(
        diffuse_color={
            "sphere": (0.09, 0.44, 0.8),
            "cube": (0.8, 0.27, 0.07),
            "polyhedron": (0.22, 0.65, 0.3),
        }[shape],
        metallic=0.2,
        roughness=0.25,
    )
    mat.func(path + "/Material", mat)
    sim_utils.bind_visual_material(str(geom.GetPath()), path + "/Material")
    return RigidObject(RigidObjectCfg(prim_path=path, spawn=None)), mesh
