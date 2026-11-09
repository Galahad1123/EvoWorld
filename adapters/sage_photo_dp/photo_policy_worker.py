"""Private local pipe worker: original robomimic policy in its own environment."""
import contextlib
import os
import pickle
import struct
import sys
from pathlib import Path

output = sys.stdout.buffer
sys.stdout = sys.stderr
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'  # only our locally trained checkpoint
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'upstream/sage/robomimic'))
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sage_dp'))
from bootstrap import disable_unused_language_encoder
disable_unused_language_encoder()
from robomimic.utils.file_utils import policy_from_checkpoint
torch.set_num_threads(2);torch.manual_seed(208);np.random.seed(208)
policy, ckpt = policy_from_checkpoint(ckpt_path=sys.argv[1],device=torch.device('cuda:0'))

def send(value):
    payload=pickle.dumps(value,protocol=5);output.write(struct.pack('!Q',len(payload)));output.write(payload);output.flush()

shape=ckpt['shape_metadata'];shape=shape[0] if isinstance(shape,list) else shape
assert 'lang_emb' not in shape['all_obs_keys']
send({'ready':True,'obs_keys':shape['all_obs_keys']})
while True:
    header=sys.stdin.buffer.read(8)
    if not header:break
    n=struct.unpack('!Q',header)[0];msg=pickle.loads(sys.stdin.buffer.read(n))
    if msg.get('reset'):
        seed=int(msg.get('seed',208));torch.manual_seed(seed);np.random.seed(seed)
        policy.start_episode();send({'reset':True});continue
    with torch.inference_mode():action=policy(msg['obs'])
    send({'action':action})
