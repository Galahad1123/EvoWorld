import unittest
import numpy as np
from geometry import cube_triangles,triangulate
class GeometryTests(unittest.TestCase):
 def test_cube_size_and_volume(self):
  v,f=cube_triangles(2)
  edges=[tuple(sorted((int(a),int(b)))) for tri in f for a,b in zip(tri,np.roll(tri,-1))]
  self.assertTrue(all(edges.count(e)==2 for e in set(edges)))
  volume=np.einsum("ij,ij->i",v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])).sum()/6
  self.assertAlmostEqual(abs(volume),8)
  np.testing.assert_allclose(np.ptp(v,axis=0),[2,2,2])
 def test_face_indices(self):
  f=triangulate([4,3],[0,1,2,3,4,5,6])
  np.testing.assert_array_equal(f,[[0,1,2],[0,2,3],[4,5,6]])
if __name__=='__main__':unittest.main()
