import unittest
import numpy as np
from usd_visual import atlas_uv
class AtlasTests(unittest.TestCase):
 def test_pixel_centers_never_cross_tiles(self):
  for tile in range(6):
   uv=atlas_uv(np.array([[0,0],[1,1],[.5,.5]]),tile,6)
   y=(1-uv[:,1])*(6*256-1)
   np.testing.assert_allclose(y,[tile*256+255,tile*256,tile*256+127.5],atol=1e-10)
 def test_dark_flat_material_does_not_mix_white_neighbor(self):
  atlas=np.ones(6*256);atlas[:256]=.014
  uv=atlas_uv(np.full((1,2),.5),0,6)
  y=(1-uv[0,1])*(len(atlas)-1);lo=int(np.floor(y));hi=int(np.ceil(y))
  sampled=atlas[lo]*(1-(y-lo))+atlas[hi]*(y-lo)
  self.assertAlmostEqual(sampled,.014)
if __name__=='__main__':unittest.main()
