import tempfile,unittest
from pathlib import Path
import h5py,numpy as np
from export_lerobot import export_dataset, verify_dataset

class ExportTests(unittest.TestCase):
    def fixture(self,p):
        with h5py.File(p,'w') as f:
            for ep in range(2):
                g=f.create_group(f'data/demo_{ep}');n=16
                a=np.zeros((n,8),'float32');a[:,0]=np.arange(n)/100+ep/10;a[:,3]=1;a[:,7]=1
                g['actions']=a;g['rewards']=np.zeros(n,'float32');g['dones']=np.zeros(n,'bool')
                g['obs/joint_qpos']=a;g['obs/ee_pose']=a
                g['obs/point_cloud']=np.arange(n*30,dtype='float32').reshape(n,10,3)
                g['obs/depth_left']=np.arange(n*16,dtype='uint8').reshape(n,4,4,1)
                rgb=np.zeros((n,32,32,3),'uint8')
                for t in range(n):rgb[t]=[t*8,ep*80,100]
                g['obs/rgb_left']=rgb
            f['mask/train']=np.asarray(['demo_0'],dtype='S');f['mask/test']=np.asarray(['demo_1'],dtype='S')
    def test_official_writer_roundtrip_and_boundary(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'source.hdf5';self.fixture(p);out=Path(t)/'dataset'
            export_dataset(p,out,'local/evoworld-test',50,'Pick up the cube.')
            report=verify_dataset(out)
            self.assertEqual(report['episodes'],2);self.assertEqual(report['frames'],32)
            self.assertTrue(report['numeric_exact']);self.assertTrue(report['action_chunk_boundaries'])
            self.assertEqual(report['explicit_split_episode_counts'],{'train':1,'test':1})
            with self.assertRaises(FileExistsError):export_dataset(p,out,'local/evoworld-test',50,'Pick up the cube.')
    def test_bad_frequency_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'source.hdf5';self.fixture(p);out=Path(t)/'dataset'
            with self.assertRaises(ValueError):export_dataset(p,out,'local/evoworld-test',0,'Pick up the cube.')
            with h5py.File(p,'r+') as f:f['data/demo_0/actions'][0,0]=np.nan
            with self.assertRaises(ValueError):export_dataset(p,out,'local/evoworld-test',50,'Pick up the cube.')
    def test_rejects_inconsistent_observation_keys(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'source.hdf5';self.fixture(p)
            with h5py.File(p,'r+') as f:f['data/demo_1/obs/extra']=np.zeros((16,2),'float32')
            with self.assertRaisesRegex(ValueError,'Observation keys'):export_dataset(p,Path(t)/'dataset','local/test',50,'Pick.')
if __name__=='__main__':unittest.main()
