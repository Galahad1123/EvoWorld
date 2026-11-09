import unittest
from training_guards import run_epoch_guarded

class SchedulerTests(unittest.TestCase):
    def test_validation_does_not_advance_scheduler(self):
        class Model:
            steps=0
            def on_gradient_step(self):self.steps+=1
        def original(*args,**kw):
            kw['model'].on_gradient_step();kw['model'].on_gradient_step();return 'ok'
        m=Model()
        self.assertEqual(run_epoch_guarded(original,model=m,validate=True),'ok')
        self.assertEqual(m.steps,0)
        run_epoch_guarded(original,model=m,validate=False);self.assertEqual(m.steps,2)
    def test_restores_callback_after_error(self):
        class Model:
            steps=0
            def on_gradient_step(self):self.steps+=1
        def bad(**kw):raise ValueError('failed validation')
        m=Model()
        with self.assertRaises(ValueError):run_epoch_guarded(bad,model=m,validate=True)
        m.on_gradient_step();self.assertEqual(m.steps,1)
if __name__=='__main__':unittest.main()
