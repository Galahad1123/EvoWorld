import unittest
from validation_loss import balanced_indices
class ValidationTests(unittest.TestCase):
 def test_balanced_unique_cover_all_episodes(self):
  ids,groups=balanced_indices([395,392,392,390,392],64)
  self.assertEqual(len(ids),320);self.assertEqual(len(set(ids)),320)
  self.assertEqual([groups.count(i) for i in range(5)],[64]*5)
  offset=0
  for ep,length in enumerate([395,392,392,390,392]):
   picked=[i for i,g in zip(ids,groups) if g==ep]
   self.assertGreaterEqual(min(picked),offset);self.assertLess(max(picked),offset+length)
   self.assertLess(min(picked)-offset,5);self.assertLess(offset+length-max(picked),6)
   offset+=length
 def test_small_episode_rejected(self):
  with self.assertRaises(ValueError):balanced_indices([5],64)
if __name__=='__main__':unittest.main()
