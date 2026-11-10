import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'research/rl/scripts'
sys.path.insert(0, str(SCRIPTS))

class BudgetTests(unittest.TestCase):
    def budget(self):
        import recap_budget
        return recap_budget

    def test_resume_preserves_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'budget.json'
            first = self.budget().load_budget(p, 15, now=100)
            second = self.budget().load_budget(p, 15, now=200)
            self.assertEqual(first, second)
            self.assertEqual(second['deadline_unix'], 54100)
            with self.assertRaises(ValueError):
                self.budget().load_budget(p, 10, now=200)

    def test_calibration_caps_steps_and_reserves_save_eval(self):
        b = self.budget()
        full = b.choose_policy_budget(10000, 1.5, deadline=30000, now=1000)
        self.assertEqual(full['steps'], 10000)
        self.assertEqual(full['warmup_steps'], 1500)
        short = b.choose_policy_budget(10000, 3., deadline=10000, now=1000)
        self.assertLess(short['steps'], 10000)
        self.assertLessEqual(short['steps'] * 3 * 1.15 + 2400, 9000)
        self.assertLessEqual(short['warmup_steps'], int(short['steps'] * .15))
        with self.assertRaises(TimeoutError):
            b.choose_policy_budget(10000, 3, deadline=2000, now=1000)
        for bad in [float('nan'),float('inf'),0,-1]:
            with self.assertRaises(ValueError):
                b.choose_policy_budget(10000,bad,deadline=30000,now=1000)

    def test_actual_checkpoint_requires_completed_save(self):
        with tempfile.TemporaryDirectory() as d:
            stage = Path(d)
            (stage/'training_status.json').write_text(json.dumps({'global_step':317,'stop_reason':'deadline'}))
            ckpt = stage/'checkpoints/global_step_317/actor'
            (ckpt/'model_state_dict').mkdir(parents=True)
            (ckpt/'model_state_dict/full_weights.pt').write_bytes(b'weights')
            with self.assertRaises(RuntimeError): self.budget().completed_checkpoint(stage)
            (ckpt/'dcp_checkpoint').mkdir()
            (ckpt/'dcp_checkpoint/.metadata').write_bytes(b'metadata')
            with self.assertRaises(RuntimeError): self.budget().completed_checkpoint(stage)
            (ckpt.parent/'checkpoint_complete.json').write_text(json.dumps({'global_step':317,'full_weights_bytes':7}))
            self.assertEqual(self.budget().completed_checkpoint(stage), ckpt)
            (ckpt/'model_state_dict/full_weights.pt').write_bytes(b'part')
            with self.assertRaises(RuntimeError): self.budget().completed_checkpoint(stage)

    def test_effective_schedule_frozen_across_resume(self):
        import yaml
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'provenance').mkdir();(out/'config').mkdir()
            config=out/'config/cfg.yaml'
            config.write_text(yaml.safe_dump({'runner':{'max_steps':10000},'actor':{'optim':{}}}))
            calibration={'step_seconds':[2.]*100}
            first=self.budget().finalize_policy_config(out,calibration,30000,10000,now=1000)
            second=self.budget().finalize_policy_config(out,calibration,30000,10000,now=2000)
            self.assertEqual(first,second)
            config.write_text('{}')
            with self.assertRaises(ValueError):self.budget().finalize_policy_config(out,calibration,30000,10000,now=3000)

    def test_cleanup_kills_grandchild_after_leader_exits(self):
        import os,signal,subprocess,time
        with tempfile.TemporaryDirectory() as d:
            pidfile=Path(d)/'child.pid'
            child_code="import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"
            parent_code=("import subprocess,sys,time;from pathlib import Path;"
                         f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
                         f"Path({str(pidfile)!r}).write_text(str(p.pid));time.sleep(60)")
            proc=subprocess.Popen([sys.executable,'-c',parent_code],start_new_session=True)
            child=None
            try:
                for _ in range(100):
                    if pidfile.exists():break
                    time.sleep(.01)
                child=int(pidfile.read_text());time.sleep(.1)
                self.budget().terminate_process_group(proc,grace_seconds=.1)
                for _ in range(100):
                    stat=Path(f'/proc/{child}/stat')
                    if not stat.exists() or stat.read_text().split()[2]=='Z':break
                    time.sleep(.01)
                else:self.fail('Grandchild survived cleanup')
            finally:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()

    def test_sigterm_enters_exception_cleanup(self):
        import subprocess,time,os,signal
        code=("import sys,time,os;sys.path.insert(0,"+repr(str(SCRIPTS))+");"
              "from recap_budget import install_termination_handler;install_termination_handler();"
              "print('ready',flush=True);time.sleep(60)")
        proc=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            self.assertEqual(proc.stdout.readline().strip(),'ready')
            proc.send_signal(signal.SIGTERM)
            _,stderr=proc.communicate(timeout=5)
            self.assertIn('InterruptedError',stderr)
        finally:
            if proc.poll() is None:proc.kill()
            proc.communicate()
