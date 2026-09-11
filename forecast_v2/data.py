"""Small, portable regional tokens from the already-computed DDPF cache."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .common import read, write, sha, verify_data


def region_masks():
    yy, xx = np.indices((32, 32))
    ids = np.minimum(yy * 3 // 32, 2) * 3 + np.minimum(xx * 3 // 32, 2)
    return np.stack([ids == n for n in range(9)])


def prepare(study, out):
    study, out = Path(study), Path(out)
    if out.resolve().is_relative_to(study.resolve()):
        raise ValueError('New data must be outside the completed study')
    source = read(study / 'split-manifest.json')
    if not (study / 'EVALUATION_COMPLETE.json').exists():
        raise ValueError('Complete the original study before starting a revision')
    out.mkdir(parents=True, exist_ok=True)
    plan = {'source_manifest_sha256': sha(study / 'split-manifest.json'), 'schema': 'regional-v2-mean256-state7-v1'}
    if (out / 'prepare.json').exists() and read(out / 'prepare.json') != plan:
        raise ValueError('Output belongs to another source/schema')
    write(out / 'prepare.json', plan)
    # Original validation is development data already; reserve eight clips for interval calibration.
    val = sorted(source['partitions']['val'], key=lambda s: __import__('hashlib').sha256(f'v2-cal:{s}'.encode()).hexdigest())
    splits = {'train': source['partitions']['train'], 'val': val[8:], 'calibration': val[:8]}
    masks = region_masks().reshape(9, -1).astype(np.float32)
    normalized = masks / masks.sum(-1, keepdims=True)
    files = {}; summaries = []
    for split, ids in splits.items():
        for sid in ids:
            target_path = out / f'{sid}.npz'; meta_path = out / f'{sid}.json'
            if target_path.exists() and meta_path.exists() and read(meta_path).get('sha256') == sha(target_path):
                files[target_path.name] = sha(target_path); continue
            folder = study / 'cache' / sid
            meta = read(folder / 'complete.json')
            for key in ('visual_features', 'flow', 'counts', 'density'):
                if sha(folder / f'{key}.npy') != meta['cache_sha256'][key]:
                    raise ValueError(f'Corrupt original cache: {sid}/{key}')
            visual = np.load(folder / 'visual_features.npy', mmap_mode='r', allow_pickle=False)
            chunks = []
            for start in range(0, len(visual), 16):
                chunks.append(np.einsum('nq,tcq->tnc', normalized, np.asarray(visual[start:start+16], np.float32).reshape(-1, 256, 1024), optimize=True))
            pooled = np.concatenate(chunks)
            observed = np.load(folder / 'observed.npy', allow_pickle=False)
            target = np.load(folder / 'target.npy', allow_pickle=False)
            # Signed flow residual removes only the global median translation proxy.
            flow = np.load(folder / 'flow.npy', mmap_mode='r', allow_pickle=False)
            xy = np.asarray(flow[:, :2], np.float32).reshape(-1, 2, 1024)
            med = np.median(xy, axis=-1)
            residual = np.einsum('nq,tcq->tnc', normalized, xy - med[:, :, None], optimize=True)
            quality = np.broadcast_to(med[:, None, :], (len(visual), 9, 2))
            x = np.concatenate([pooled, observed[:, :, :5], residual, quality], axis=-1).astype(np.float32)
            if not all(np.isfinite(a).all() for a in (x, observed, target)):
                raise ValueError('Nonfinite source tokens')
            # Only selected target fields are labels. No current/future manual counts in x.
            with target_path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, x=x, observed=observed[:, :, [0, 6]], y=target[:, :, [0, 6]], frames=np.array([r['frame_index'] for r in source['sequences'][sid]['frames']]))
            target_path.with_suffix('.tmp').replace(target_path)
            digest = sha(target_path); files[target_path.name] = digest
            write(meta_path, {'sha256': digest, 'source_cache': meta, 'split': split})
            print(f'Prepared {split}/{sid}', flush=True)
    arrays = []
    for sid in splits['train']:
        with np.load(out / f'{sid}.npz', allow_pickle=False) as data:
            arrays.append({k: data[k] for k in ('x', 'observed', 'y')})
    x = np.concatenate([a['x'] for a in arrays])
    observed = np.concatenate([a['observed'] for a in arrays])
    truth = np.concatenate([a['y'] for a in arrays])
    mean, std = x.mean((0, 1)), np.maximum(x.std((0, 1)), .01)
    # Region-specific affine count correction with global-count context. Fit training frames only.
    design = np.stack([observed[..., 0], np.broadcast_to(observed[..., 0].mean(-1, keepdims=True), observed[..., 0].shape), np.ones(observed.shape[:2])], axis=-1)
    coefficients = []
    for n in range(9):
        a = design[:, n].astype(np.float64); b = truth[:, n, 0]
        coefficients.append(np.linalg.solve(a.T @ a + np.diag([1., 1., 0.]), a.T @ b))
    thresholds = read(study / 'target-calibration.json')['pressure_thresholds']
    calibration = {'x_mean': mean.tolist(), 'x_std': std.tolist(), 'count_coefficients': np.asarray(coefficients).tolist(), 'count_scale': float(max(np.quantile(truth[..., 0], .95), 1)), 'pressure_thresholds': thresholds, 'source_split': 'train', 'source_sequences': splits['train']}
    write(out / 'calibration.json', calibration); files['calibration.json'] = sha(out / 'calibration.json')
    write(out / 'manifest.json', {**plan, 'splits': splits, 'excluded_exposed_test': source['partitions']['test'], 'input_dim': x.shape[-1], 'regions': 9, 'files': files, 'evaluation_role': 'development_only', 'temporal_unit': 'annotation_steps', 'ddpf_sha256': source['ddpf_sha256'], 'limitations': ['Region tokens discard within-region spatial structure.', 'Location independence and original DDPF training provenance unverified.', 'Pressure is the original training-calibrated perception proxy.']})
    verify_data(out)


def corrected(observed, calibration):
    count = observed[..., 0]
    design = np.stack([count, np.broadcast_to(count.mean(-1, keepdims=True), count.shape), np.ones_like(count)], -1)
    return np.maximum(np.einsum('...nf,nf->...n', design, np.array(calibration['count_coefficients'])), 0).astype(np.float32)


class Windows(Dataset):
    def __init__(self, root, split, context=32, horizons=(5, 15, 30), stride=5):
        root = Path(root); manifest = read(root / 'manifest.json')
        self.cal = read(root / 'calibration.json'); self.context = context; self.horizons = list(horizons)
        self.arrays = {}; self.index = []
        for sid in manifest['splits'][split]:
            with np.load(root / f'{sid}.npz', allow_pickle=False) as file:
                self.arrays[sid] = {k: file[k] for k in file.files}
            a = self.arrays[sid]; a['anchor'] = corrected(a['observed'], self.cal)
            if np.any(np.diff(a['frames']) != 1): raise ValueError('Irregular cadence requires explicit resampling protocol')
            self.index.extend((sid, t) for t in range(context - 1, len(a['x']) - max(horizons), stride))
        if not self.index: raise ValueError('No valid temporal windows')

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        sid, t = self.index[i]; a = self.arrays[sid]; window = slice(t-self.context+1, t+1)
        x = np.clip((a['x'][window] - np.array(self.cal['x_mean'])) / np.array(self.cal['x_std']), -12, 12).astype(np.float32)
        return {'x': torch.from_numpy(x), 'anchor': torch.from_numpy(a['anchor'][window].copy()), 'observed': torch.from_numpy(a['observed'][window].copy()), 'current': torch.from_numpy(a['y'][t].copy()), 'target': torch.from_numpy(a['y'][[t+h for h in self.horizons]].copy()), 'scene': sid, 'frame': t}
