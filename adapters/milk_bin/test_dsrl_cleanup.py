import os,subprocess,unittest
from dsrl_cleanup import cleanup
class CleanupTests(unittest.TestCase):
 def test_stops_only_this_run_processes(self):
  marker='/tmp/evoworld-cleanup-test-'+str(os.getpid())
  own=subprocess.Popen(['sleep','60'],env={**os.environ,'EVOWORLD_DSRL_AUDIT_DIR':marker})
  other=subprocess.Popen(['sleep','60'])
  try:
   result=cleanup(marker,grace=.2)
   self.assertIn(own.pid,result['matched']);own.wait(timeout=3)
   self.assertIsNone(other.poll());self.assertEqual(result['remaining'],[])
  finally:
   for p in [own,other]:
    if p.poll() is None:p.kill()
    p.wait()
if __name__=='__main__':unittest.main()
