import json,unittest
import numpy as np
from specs import action_record
class RecordingTests(unittest.TestCase):
    def test_float32_action_is_json_serializable(self):
        action=np.array([.4,0,.5,1,0,0,0,1],dtype=np.float32)
        saved=json.loads(json.dumps(action_record(action),allow_nan=False));np.testing.assert_allclose(saved,action)
if __name__=='__main__':unittest.main()
