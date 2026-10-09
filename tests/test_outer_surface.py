"""Forward reference traversal regressions; no simulator required."""
import importlib.util
from pathlib import Path
import torch
import pytest
PATH=Path(__file__).resolve().parents[1]/"src/revosim/tactile/backends/outer_surface.py"
spec=importlib.util.spec_from_file_location("outer_surface_test",PATH)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
solve=module.raycast_outer_surface_forward

def layer_caster(layers, signs):
    def cast(starts,directions,**kwargs):
        z=torch.tensor(layers,dtype=starts.dtype,device=starts.device)
        distances=(z-starts[...,2,None])/directions[...,2,None]
        distances=torch.where((distances>=0)&(distances<=kwargs["max_dist"]),distances,torch.inf)
        distance,index=distances.min(-1)
        normal=torch.zeros_like(starts)
        normal[...,2]=torch.tensor(signs,dtype=starts.dtype,device=starts.device)[index]
        points=starts+directions*distance.unsqueeze(-1)
        return points,distance,normal,None,None
    return cast

def rays(z=0.0):
    return torch.tensor([[[0.,0.,z]]]),torch.tensor([[[0.,0.,1.]]])

def test_four_crossings_select_outer_surface_not_second():
    p,d=rays()
    hits,depth,n,valid=solve(p,d,mesh_ids_wp=None,max_dist=.05,
        raycast_fn=layer_caster([.001,.002,.003,.012],[-1,1,-1,1]))
    assert valid.all()
    torch.testing.assert_close(depth,torch.tensor([[.012]]))
    torch.testing.assert_close(hits[...,2],depth)
    assert (n[...,2]>0).all()

def test_inside_origin_single_exit_is_valid():
    p,d=rays(.005)
    _,depth,_,valid=solve(p,d,mesh_ids_wp=None,max_dist=.05,
        raycast_fn=layer_caster([.001,.012],[-1,1]))
    assert valid.all()
    torch.testing.assert_close(depth,torch.tensor([[.007]]))

def test_missing_surface_is_invalid_and_not_first_hit_fallback():
    p,d=rays(.020)
    hits,depth,n,valid=solve(p,d,mesh_ids_wp=None,max_dist=.05,
        raycast_fn=layer_caster([.001,.012],[-1,1]))
    assert not valid.any() and torch.isinf(depth).all()
    assert not hits.any() and not n.any()

def test_inward_only_hit_is_not_accepted_as_outer_contact():
    p,d=rays()
    _,depth,_,valid=solve(p,d,mesh_ids_wp=None,max_dist=.05,
        raycast_fn=layer_caster([.001],[-1]))
    assert not valid.any() and torch.isinf(depth).all()

def test_thin_extra_layers_are_not_skipped_by_old_ten_micron_step():
    p,d=rays()
    _,depth,_,valid=solve(p,d,mesh_ids_wp=None,max_dist=.05,
        raycast_fn=layer_caster([.001,.002,.002002,.002004],[-1,1,-1,1]))
    assert valid.all()
    assert abs(depth.item()-.002004)<1e-8

def test_iteration_limit_fails_instead_of_accepting_inner_surface():
    p,d=rays()
    with pytest.raises(RuntimeError,match="hit its limit"):
        solve(p,d,mesh_ids_wp=None,max_dist=.05,max_hits=2,
            raycast_fn=layer_caster([.001,.002,.003,.012],[-1,1,-1,1]))

def test_mixed_batch_does_not_overwrite_finished_rays():
    p,d=rays();p=p.expand(2,1,3).clone();d=d.expand(2,1,3).clone();p[1,0,2]=.020
    _,depth,_,valid=solve(p,d,mesh_ids_wp=None,max_dist=.05,
        raycast_fn=layer_caster([.001,.002,.003,.012],[-1,1,-1,1]))
    assert valid[:,0].tolist()==[True,False]
    assert abs(depth[0,0].item()-.012)<1e-8 and torch.isinf(depth[1,0])

def test_zero_direction_rejected():
    p,d=rays()
    with pytest.raises(ValueError,match="Zero"):
        solve(p,d*0,mesh_ids_wp=None,max_dist=.05)
