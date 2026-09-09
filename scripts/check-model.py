import json
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import torch
from backend.balanced_ddpf import BalancedPredictor

torch.set_num_threads(2)
model=BalancedPredictor()
image=cv2.imread(sys.argv[1])
if image is None:
    raise ValueError('Image cannot be read')
started=time.perf_counter()
output=model.export_sequence([image],'verification',1)
result={'checkpoint_sha256':model.sha256,'strict_load':True,'device':str(model.device),'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,'finite':bool(torch.from_numpy(output.density).isfinite().all()),'count':float(output.density.sum()),'padded_count':model.last_padded_count,'padding_excluded':model.last_padding_count,'head_locations':len(model.last_points),'density_shape':list(output.density.shape),'feature_shape':list(output.visual_features.shape),'seconds':time.perf_counter()-started,'note':'Inference smoke test on one image; not an accuracy evaluation.'}
print(json.dumps(result,indent=2),flush=True)
Path('artifacts').mkdir(exist_ok=True)
Path('artifacts/model-verification.json').write_text(json.dumps(result,indent=2))
