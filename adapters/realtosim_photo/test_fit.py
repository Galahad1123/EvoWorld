import unittest
import numpy as np
from fit import fit_camera, project

class CameraFitTest(unittest.TestCase):
    def test_recovers_visible_nonplanar_anchor(self):
        points = np.array([[x,y,z] for x in [-.0275,.0275] for y in [-.019,.019] for z in [.001,.116]])
        position = np.array([.12,-.5,.31])
        center = np.array([0.,0.,.06])
        forward = center-position; forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
        down=np.cross(forward,right)
        R=np.array([right,down,forward]); t=-R@position
        observed=project(points,R,t,1300.,[1706,1279])
        view={'image_size_wh':[1706,1279], 'focal_length_px':1200., 'camera_position_m':[.1,-.55,.35],
              'look_at_m':[0,0,.06], 'points':[{'world_xyz':p.tolist(),'pixel_xy':uv.tolist(),'label':str(i)} for i,(p,uv) in enumerate(zip(points,observed))]}
        fit=fit_camera(view)
        self.assertLess(fit['rms_reprojection_px'],.05)
        self.assertTrue(np.allclose(np.asarray(fit['world_T_camera_cv'])[:3,3],position,atol=.01))

    def test_rejects_nonfinite_or_insufficient_landmarks(self):
        with self.assertRaises(ValueError):
            fit_camera({'image_size_wh':[100,100], 'points':[]})

    def test_explicit_focal_prior_is_not_refitted_from_ambiguous_panel(self):
        xyz=np.array([[-.0275,-.019,0],[.0275,-.019,0],[.0275,-.019,.112],[-.0275,-.019,.112]])
        position=np.array([0.,-.5,.35]);forward=np.array([0,0,.05])-position;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);R=np.array([right,np.cross(forward,right),forward])
        uv=project(xyz,R,-R@position,1400.,[1706,1279]);uv[0,0]+=2
        view={'image_size_wh':[1706,1279],'focal_length_px':1200.,'focal_length_fixed_px':1400.,
              'camera_position_m':position.tolist(),'look_at_m':[0,0,.05],
              'points':[{'world_xyz':p.tolist(),'pixel_xy':u.tolist()} for p,u in zip(xyz,uv)]}
        fit=fit_camera(view)
        self.assertEqual(fit['focal_length_px'],1400.)
        self.assertEqual(fit['focal_constraint'],'assumed_fixed_not_calibrated')

if __name__=='__main__':unittest.main()
