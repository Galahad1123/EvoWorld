import sys,json,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/rl/scripts'))

class VideoCacheTests(unittest.TestCase):
 def test_unverified_cache_does_not_replace_original(self):
  from recap_video_cache import activate_cache
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);raw=root/'raw';raw.mkdir();view=root/'run/data/rollout';view.mkdir(parents=True);(view/'videos').symlink_to(raw)
   with self.assertRaises(RuntimeError):activate_cache(root/'cache',root/'run')
   self.assertEqual((view/'videos').resolve(),raw)
 def test_verified_cache_changes_video_links_only(self):
  from recap_video_cache import activate_cache
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cache=root/'cache';cache.mkdir()
   (cache/'verified.json').write_text(json.dumps({'verified':True,'files':2}))
   for name,split in [('rollout','train'),('eval','eval')]:
    view=root/'run/data'/name;view.mkdir(parents=True);(view/'meta').mkdir();(view/'meta/sentinel').write_text('original')
    videos=cache/f'libero10_task0_{split}/videos';videos.mkdir(parents=True)
   activate_cache(cache,root/'run')
   for name,split in [('rollout','train'),('eval','eval')]:
    view=root/'run/data'/name
    self.assertEqual((view/'videos').resolve(),cache/f'libero10_task0_{split}/videos')
    self.assertEqual((view/'meta/sentinel').read_text(),'original')
 def test_manifest_hash_includes_timing_and_pixels(self):
  from recap_video_cache import canonical_frames
  text='#format: frame checksums\n#tb 0: 1/10\n0, 0, 0, 1, 196608, abc\n'
  self.assertEqual(canonical_frames(text),['#tb 0: 1/10','0, 0, 0, 1, 196608, abc'])
