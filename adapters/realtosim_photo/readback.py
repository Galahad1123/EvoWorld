"""Read actual PhysX traces independently; no action or sensor-stream reinterpretation."""
import json
from pathlib import Path
import sys
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

def validate_trace(t,p,v,w):
    assert t.shape==(121,) and p.shape==(121,7) and v.shape==w.shape==(121,3)
    assert all(np.isfinite(x).all() for x in [t,p,v,w])
    assert t[0]==0 and np.allclose(np.diff(t),1/60,atol=1e-7,rtol=0) and abs(t[-1]-2)<1e-5
    assert np.allclose(np.linalg.norm(p[:,3:],axis=1),1,atol=1e-5)
    drop=float(p[0,2]-p[-1,2]);assert .01<drop<.04,('Drop not observed / support lost',drop)
    speed=float(np.linalg.norm(v[-61:],axis=1).max())
    span=float(np.linalg.norm(np.ptp(p[-61:,:3],axis=0)))
    rotation=Rotation.from_quat(p[:,[4,5,6,3]])
    degrees=float(np.rad2deg((rotation[-61].inv()*rotation[-61:]).magnitude()).max())
    assert speed<.01 and span<.002 and degrees<3,('Unstable full final second',speed,span,degrees)
    assert np.max(np.linalg.norm(p[:,:2]-p[0,:2],axis=1))<.02
    return {'drop_m':drop,'last_second_peak_speed_m_s':speed,'last_second_center_span_m':span,'last_second_rotation_degrees':degrees,'tail_seconds':float(t[-1]-t[-61])}

if __name__=='__main__':
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    candidate=Path(sys.argv[1]).resolve();out=candidate/'physics'
    ex=json.loads((out/'execution.json').read_text());assert ex['completed'] and not (out/'error.json').exists()
    log=(out/'process.log').read_text();assert 'PHOTO_ASSET_PHYSICS_COMPLETE' in log and 'Traceback (most recent call last)' not in log
    assert ex['timing']['steps']==ex['timing']['callback_count']==120 and abs(ex['timing']['callback_seconds']-2)<1e-5
    stage=Usd.Stage.Open(str(candidate/'relocated/scene.usda'))
    transforms=UsdGeom.XformCache(Usd.TimeCode.Default())
    body_names={str(p.GetPath()).split('/')[-1] for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    assert body_names=={'milk','tissue'}==set(ex['dynamic_ids'])
    result={}
    with h5py.File(out/'states.hdf5','r') as f:
        assert f.attrs['schema']=='evoworld.photo_asset_physics.v1' and f.attrs['quaternion']=='wxyz' and f.attrs['up_axis']=='Z'
        assert set(f.keys())==body_names|{'time'}
        assert json.loads(f.attrs['timing'])==ex['timing']
        t=f['time'][:]
        for name in sorted(body_names):
            p,v,w=[f[name+'/'+k][:] for k in ['pose_world','linear_velocity_world','angular_velocity_world']]
            r=validate_trace(t,p,v,w);assert np.allclose(p[0,:3],ex['initial_positions'][name],atol=1e-6)
            body=stage.GetPrimAtPath('/World/'+name)
            mass=UsdPhysics.MassAPI(body).GetMassAttr().Get();assert abs(mass-ex['actual_mass_kg'][name])<1e-5
            vertices=[]
            for child in Usd.PrimRange(body):
                if child.HasAPI(UsdPhysics.CollisionAPI) and child.IsA(UsdGeom.Cube):
                    size=UsdGeom.Cube(child).GetSizeAttr().Get()
                    transform=transforms.GetLocalToWorldTransform(child)*transforms.GetLocalToWorldTransform(body).GetInverse()
                    for x in [-.5,.5]:
                        for y in [-.5,.5]:
                            for z in [-.5,.5]:vertices.append(list(transform.Transform(Gf.Vec3d(x*size,y*size,z*size))))
            assert vertices,'Need explicit cube collision support for this fixture'
            rotation=Rotation.from_quat(p[:,[4,5,6,3]])
            corners=np.einsum('nij,kj->nki',rotation.as_matrix(),np.asarray(vertices))+p[:,:3,None].transpose(0,2,1)
            bottoms=corners[:,:,2].min(axis=1)
            assert abs(bottoms[-1])<.004 and bottoms.min()>-.004,('Table support failure',name,bottoms[-1],bottoms.min())
            r.update(final_collision_bottom_z_m=float(bottoms[-1]),minimum_collision_bottom_z_m=float(bottoms.min()),mass_kg_designed=float(mass))
            result[name]=r
    report={'passed':True,'states':121,'steps':120,'seconds':float(t[-1]),'objects':result,
            'scope':'Real PhysX rigid proxies; mass/friction uncalibrated, tissue is not deformable. No robot actions or synchronized images.'}
    (candidate/'physics-readback.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
