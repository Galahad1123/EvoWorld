"""Exact analytic-cube triangles and polygon fan indices for the USD bridge."""
import numpy as np

def triangulate(counts,indices):
 result=[];cursor=0
 for n in counts:
  if n<3:raise ValueError('Degenerate face')
  face=indices[cursor:cursor+n];cursor+=n
  if len(face)!=n:raise ValueError('Truncated face')
  result.extend([face[0],face[i],face[i+1]] for i in range(1,n-1))
 if cursor!=len(indices):raise ValueError('Unused indices')
 return np.asarray(result,dtype=np.int32)

def cube_triangles(size):
 if size<=0:raise ValueError('Cube size must be positive')
 vertices=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],dtype=float)*size/2
 faces=triangulate([4]*6,[0,3,2,1,4,5,6,7,0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7])
 return vertices,faces
