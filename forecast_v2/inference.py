"""Research inference on raw regional tokens; deliberately not dashboard promotion."""
from pathlib import Path
import numpy as np
import torch
from .common import read,sha,sources
from .data import corrected
from .train import make_model


class Predictor:
    def __init__(self, run, device=None):
        run=Path(run);done=read(run/'COMPLETE.json');contract=read(run/'contract.json')
        if sha(run/'best.pt')!=done['best_sha256'] or contract['sources']!=sources():
            raise ValueError('Restore frozen checkpoint/source files before inference')
        self.payload=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        self.config=self.payload['config'];self.cal=self.payload['calibration']
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model=make_model(self.config,self.cal,self.payload['input_dim']).to(self.device)
        self.model.load_state_dict(self.payload['model']);self.model.eval()
        path=run/'development-evaluation/interval-calibration.json'
        self.interval=read(path) if path.exists() else None
        if self.interval and self.interval['best_sha256']!=done['best_sha256']:raise ValueError('Interval calibration is stale')

    @torch.no_grad()
    def predict(self, tokens, observed):
        tokens=np.asarray(tokens,dtype=np.float32);observed=np.asarray(observed,dtype=np.float32)
        expected=(self.config['context'],9,self.payload['input_dim'])
        if tokens.shape!=expected or observed.shape!=(expected[0],9,2):raise ValueError('Input shape does not match frozen token schema/context')
        if not np.isfinite(tokens).all() or not np.isfinite(observed).all():raise ValueError('Nonfinite inputs')
        x=np.clip((tokens-self.cal['x_mean'])/np.array(self.cal['x_std']),-12,12).astype('float32')
        anchor=corrected(observed,self.cal)
        out=self.model(torch.from_numpy(x[None]).to(self.device),torch.from_numpy(anchor[None]).to(self.device),torch.from_numpy(observed[None]).to(self.device))
        result={k:v[0].cpu().numpy() for k,v in out.items()}
        if self.interval:
            result['lower']=np.maximum(result['lower']-self.interval['radius'],0)
            result['upper']=result['upper']+self.interval['radius']
        result.update(horizons=self.config['horizons'],temporal_unit='annotation_steps',pressure_semantics='perception-derived proxy',deployment_approved=False,interval_role='development sequence calibration' if self.interval else 'uncalibrated learned quantiles')
        return result
