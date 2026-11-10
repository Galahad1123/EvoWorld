import tempfile
import unittest
from pathlib import Path
import numpy as np
from evaluate import adapt_actions, cleanup_scratch, prepare_scratch

class Contracts(unittest.TestCase):
    def test_gripper_and_motion(self):
        a = np.array([[[.1, .2, .3, .4, .5, .6, 0.], [.1, .2, .3, .4, .5, .6, 1.]]])
        b = adapt_actions(a)
        np.testing.assert_array_equal(b[..., :6], a[..., :6])
        np.testing.assert_array_equal(b[..., -1], [[1., -1.]])
        np.testing.assert_array_equal(a[..., -1], [[0., 1.]])

    def test_cleanup_requires_marker_and_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            scratch = root / 'scratch/libero/test'
            scratch.mkdir(parents=True)
            (scratch / 'large.tmp').write_bytes(b'x' * 100)
            with self.assertRaises(ValueError):
                cleanup_scratch(root, 'test', passed=True)
            (scratch / '.evoworld-owned').write_text('test')
            with self.assertRaises(ValueError):
                cleanup_scratch(root, 'test', passed=False)
            (root / 'keep').write_text('important')
            self.assertGreaterEqual(cleanup_scratch(root, 'test', passed=True), 100)
            self.assertFalse(scratch.exists())
            self.assertTrue((root / 'keep').exists())

    def test_cleanup_rejects_escape(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                cleanup_scratch(Path(d), '../outside', passed=True)

    def test_prepare_refuses_unowned_directory(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / 'scratch/libero/test'
            p.mkdir(parents=True)
            (p / 'important').write_text('keep')
            with self.assertRaises(ValueError):
                prepare_scratch(root, 'test')
            self.assertFalse((p / '.evoworld-owned').exists())

    def test_prepare_reuses_owned_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = prepare_scratch(root, 'test')
            self.assertEqual(prepare_scratch(root, 'test'), p)
            (p / '.evoworld-owned').write_text('different')
            with self.assertRaises(ValueError):
                prepare_scratch(root, 'test')

if __name__ == '__main__':
    unittest.main()
