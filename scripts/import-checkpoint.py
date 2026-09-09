"""Convert the supplied checkpoint to a weights-only local inference artifact."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser()
p.add_argument('checkpoint',type=Path)
args=p.parse_args()
torch.serialization.add_safe_globals([(np._core.multiarray.scalar,'numpy.core.multiarray.scalar'),np.dtype,np.dtypes.Float64DType,np.dtypes.Float32DType])
checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
root=Path(__file__).resolve().parents[1]
target=root/'models/balanced_ddpf.pt'
target.parent.mkdir(exist_ok=True)
torch.save({'model_state_dict':checkpoint['model_state_dict'],'config':checkpoint['config'],'epoch':checkpoint['epoch'],'best_conf_threshold':float(checkpoint.get('best_conf_threshold',.25)),'source_filename':args.checkpoint.name},target)
with args.checkpoint.open('rb') as handle:source_hash=hashlib.file_digest(handle,'sha256').hexdigest()
with target.open('rb') as handle:output_hash=hashlib.file_digest(handle,'sha256').hexdigest()
manifest={'source':args.checkpoint.name,'source_sha256':source_hash,'inference_sha256':output_hash,'epoch':checkpoint['epoch'],'architecture':'DDPFNetLastHope / supplied training.py cells 6–9','normalization':'RGB/255, ImageNet mean/std','input':512,'density_scale':100,'offset':'0.5*tanh; signed','config':checkpoint['config'],'checkpoint_metrics_note':'Stored training metrics are not independent verification.'}
(target.parent/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(target)
