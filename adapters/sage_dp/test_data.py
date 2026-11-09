import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from prepare import inspect_data, split_episodes


class DataTests(unittest.TestCase):
    def make(self, path):
        with h5py.File(path, 'w') as f:
            for i in range(4):
                g = f.create_group(f'data/demo_{i}')
                a = np.zeros((20, 8), dtype='f'); a[:, 0] = i / 10; a[:, 3] = 1; a[:, 7] = 1
                g['actions'] = a
                for prefix in ['obs', 'next_obs']:
                    for key in ['ee_pose', 'joint_qpos']:
                        g[f'{prefix}/{key}'] = a
                    for key in ['rgb_left', 'rgb_right']:
                        g[f'{prefix}/{key}'] = np.zeros((20, 128, 128, 3), dtype='uint8')

    def test_split_whole_episodes_and_deterministic(self):
        names=[f'demo_{i}' for i in range(20)]
        a, b, c = split_episodes(names, 42)
        self.assertFalse(set(a) & set(b)); self.assertEqual(len(b), 2)
        self.assertFalse(set(a) & set(c)); self.assertFalse(set(b) & set(c))
        self.assertEqual(len(a),16);self.assertEqual(len(c),2)
        self.assertEqual((a,b,c), split_episodes(names, 42))

    def test_reject_too_few(self):
        with self.assertRaises(ValueError): split_episodes(['demo_0','demo_1'], 42)

    def test_reject_nonfinite_and_duplicate(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/'x.hdf5'; self.make(p)
            self.assertEqual(len(inspect_data(p)), 4)
            with h5py.File(p,'r+') as f: f['data/demo_1/actions'][0,0] = np.nan
            with self.assertRaises(ValueError): inspect_data(p)
            self.make(p)
            with h5py.File(p,'r+') as f: f['data/demo_1/actions'][:] = f['data/demo_0/actions'][:]
            with self.assertRaises(ValueError): inspect_data(p)

    def test_reject_bad_shape_and_quaternion(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.hdf5'; self.make(p)
            with h5py.File(p,'r+') as f: f['data/demo_0/actions'][:,3:7] = 0
            with self.assertRaises(ValueError): inspect_data(p)

    def test_reject_temporal_shift(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.hdf5';self.make(p)
            with h5py.File(p,'r+') as f:f['data/demo_0/next_obs/ee_pose'][3,0]=.01
            with self.assertRaisesRegex(ValueError,'temporal mismatch'):inspect_data(p)
            self.make(p)
            with h5py.File(p,'r+') as f:
                del f['data/demo_0/obs/rgb_left'];f['data/demo_0/obs/rgb_left']=np.zeros((19,128,128,3),'uint8')
            with self.assertRaises(ValueError): inspect_data(p)


if __name__ == '__main__': unittest.main()
