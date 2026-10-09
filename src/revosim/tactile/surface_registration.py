"""Register embedded pressure channels to the actual outer high-fidelity skin.

Source channel centres lie below the visible elastomer. Cast from outside along
an outward normal onto that same pad; preserve channel order and tangential
placement. This avoids an arbitrary global thickness shift for curved pads.
"""
import numpy as np


def ray_surface_points(points, normals, vertices, faces, reach=.02):
    triangles=vertices[faces];v0=triangles[:,0];e1=triangles[:,1]-v0;e2=triangles[:,2]-v0
    projected=[];valid=[]
    for p,n in zip(points,normals):
        n=n/np.linalg.norm(n);origin=p+reach*n;direction=-n
        h=np.cross(np.broadcast_to(direction,e2.shape),e2);det=np.einsum('ij,ij->i',e1,h)
        ok=np.abs(det)>1e-12;inv=np.zeros_like(det);inv[ok]=1/det[ok]
        s=origin-v0;u=np.einsum('ij,ij->i',s,h)*inv;q=np.cross(s,e1);v=q@direction*inv;t=np.einsum('ij,ij->i',e2,q)*inv
        ok&=(u>=-1e-7)&(v>=-1e-7)&(u+v<=1+1e-7)&(t>0)&(t<2*reach)
        hit=bool(ok.any());projected.append(origin+direction*t[ok].min() if hit else p);valid.append(hit)
    return np.asarray(projected),np.asarray(valid)


def register_pad(robot_path, link_name, points, normals):
    import isaaclab.sim as sim_utils
    from pxr import Usd,UsdGeom,Gf
    root=sim_utils.find_first_matching_prim(robot_path)
    source=root.GetStage().GetPrimAtPath(str(root.GetPath())+'/'+link_name)
    skin_name=link_name.replace('_touch_link','_rubber_link')
    skin=root.GetStage().GetPrimAtPath(str(root.GetPath())+'/'+skin_name)
    if not skin.IsValid():
        skin_name=skin_name.replace('_rubber_link','_tubber_link')
        skin=root.GetStage().GetPrimAtPath(str(root.GetPath())+'/'+skin_name)
    if not skin.IsValid():raise ValueError('Missing outer skin '+skin_name)
    cache=UsdGeom.XformCache();inv=cache.GetLocalToWorldTransform(source).GetInverse();verts=[];faces=[];offset=0
    for prim in Usd.PrimRange(skin):
        if not prim.IsA(UsdGeom.Mesh):continue
        mesh=UsdGeom.Mesh(prim);transform=cache.GetLocalToWorldTransform(prim)*inv
        vs=np.array([list(transform.Transform(Gf.Vec3d(*map(float,p)))) for p in mesh.GetPointsAttr().Get()]);idx=list(mesh.GetFaceVertexIndicesAttr().Get());start=0;tri=[]
        for count in mesh.GetFaceVertexCountsAttr().Get():
            ids=idx[start:start+count];tri.extend([[ids[0],ids[i],ids[i+1]] for i in range(1,count-1)]);start+=count
        verts.append(vs);faces.append(np.array(tri)+offset);offset+=len(vs)
    projected,valid=ray_surface_points(np.asarray(points),np.asarray(normals),np.concatenate(verts),np.concatenate(faces))
    if not valid.all():
        raise RuntimeError(f'{skin_name}: pressure channels outside skin: {np.flatnonzero(~valid).tolist()}')
    print('PRESSURE_REGISTRATION',link_name,'channels',len(points),'offset_mm',np.round(np.linalg.norm(projected-points,axis=1).min()*1000,3),np.round(np.linalg.norm(projected-points,axis=1).max()*1000,3),flush=True)
    return projected+np.asarray(normals)*.0002
