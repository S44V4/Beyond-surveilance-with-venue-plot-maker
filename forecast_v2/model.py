"""Compact temporal model with optional count-preserving regional transport."""
import torch
from torch import nn
from torch.nn import functional as F


def adjacency():
    return torch.tensor([[abs(i//3-j//3)+abs(i%3-j%3) <= 1 for j in range(9)] for i in range(9)])


class TemporalBlock(nn.Module):
    def __init__(self, dim, dilation, dropout):
        super().__init__(); self.pad = 2*dilation
        self.conv = nn.Conv1d(dim, dim*2, 3, dilation=dilation)
        self.out = nn.Conv1d(dim, dim, 1); self.norm = nn.LayerNorm(dim); self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.conv(F.pad(x.transpose(1, 2), (self.pad, 0)))
        a, b = y.chunk(2, dim=1)
        y = self.out(torch.tanh(a)*torch.sigmoid(b)).transpose(1, 2)
        return self.norm(x + self.dropout(y))


class RegionalForecaster(nn.Module):
    def __init__(self, input_dim=265, dim=64, horizons=(5, 15, 30), mode='transport', dropout=.1, count_scale=50., thresholds=(.2,.4,.6)):
        super().__init__()
        if mode not in ('transport', 'direct', 'state_only'): raise ValueError(mode)
        self.mode = mode; self.horizons = tuple(horizons); self.count_scale = float(count_scale)
        if not self.horizons or min(self.horizons)<=0 or list(self.horizons)!=sorted(set(self.horizons)):
            raise ValueError('Horizons must be positive and strictly increasing')
        self.register_buffer('thresholds', torch.tensor(thresholds, dtype=torch.float32))
        self.register_buffer('adj', adjacency())
        self.register_buffer('boundary', torch.tensor([i != 4 for i in range(9)], dtype=torch.float32))
        self.project = nn.Linear(input_dim if mode != 'state_only' else input_dim-256, dim)
        self.temporal = nn.Sequential(*[TemporalBlock(dim, d, dropout) for d in (1, 2, 4, 8)])
        self.spatial = nn.MultiheadAttention(dim, 4, dropout=dropout, batch_first=True) if mode=='transport' else None
        self.norm = nn.LayerNorm(dim) if mode=='transport' else None
        self.horizon = nn.Parameter(torch.randn(len(horizons), dim)*.02)
        self.decode = nn.Sequential(nn.Linear(dim*2+1, dim), nn.GELU(), nn.LayerNorm(dim))
        self.current = nn.Linear(dim, 1); self.delta = nn.Linear(dim, 1) if mode!='transport' else None
        self.transport = nn.Linear(dim, 9) if mode=='transport' else None
        self.exchange = nn.Linear(dim, 2) if mode=='transport' else None
        self.width = nn.Linear(dim, 2); self.pressure = nn.Linear(dim, 2)
        for layer in (self.current, self.delta):
            if layer is None: continue
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        if self.transport is not None:
            nn.init.zeros_(self.transport.weight); nn.init.zeros_(self.transport.bias)
            nn.init.zeros_(self.exchange.weight); nn.init.constant_(self.exchange.bias, -7.)
        nn.init.zeros_(self.pressure.weight); nn.init.zeros_(self.pressure.bias)

    def forward(self, x, anchor, observed):
        b, t, n, _ = x.shape
        if n != 9: raise ValueError('This model uses a 3x3 image grid, not arbitrary venue topology')
        if self.mode == 'state_only': x = x[..., 256:]
        z = self.project(x).permute(0,2,1,3).reshape(b*n, t, -1)
        z = self.temporal(z)
        z = z[:, -1].reshape(b,n,-1)
        if self.mode == 'transport':
            attended, _ = self.spatial(z,z,z, attn_mask=~self.adj, need_weights=False)
            z = self.norm(z+attended)
        current = (anchor[:, -1].float() + self.current(z).squeeze(-1).float()*self.count_scale).clamp_min(0)
        count = current; outputs = []; lowers = []; uppers = []; pressures = []; probabilities = []; balances = []
        previous_h = 0
        for hi, horizon in enumerate(self.horizons):
            emb = self.horizon[hi].expand(b,n,-1)
            h = self.decode(torch.cat([z, emb, (count/self.count_scale).unsqueeze(-1).to(z.dtype)], -1))
            if self.mode == 'transport':
                logits = self.transport(h).float().masked_fill(~self.adj, -1e4)
                logits = logits + torch.eye(n, device=x.device)*7.
                matrix = logits.softmax(-1)
                moved = torch.einsum('bn,bnm->bm', count, matrix)
                rate = self.exchange(h).float(); step = (horizon-previous_h)/30.
                leave = moved*(-torch.expm1(-F.softplus(rate[...,0])*step))*self.boundary
                enter = F.softplus(rate[...,1])*self.count_scale*self.boundary*step
                next_count = moved-leave+enter
                balances.append(next_count.sum(-1)-count.sum(-1)+leave.sum(-1)-enter.sum(-1))
                count = next_count
            else:
                count = (current + self.delta(h).squeeze(-1).float()*self.count_scale).clamp_min(0)
                balances.append(torch.zeros(b, device=x.device))
            width = F.softplus(self.width(h).float())*self.count_scale
            lowers.append((count-width[...,0]).clamp_min(0)); uppers.append(count+width[...,1]); outputs.append(count)
            # One ordered logistic distribution drives both scalar and categorical proxy outputs.
            p = self.pressure(h).float()
            mean = torch.sigmoid(torch.logit(observed[:,-1,:,1].float().clamp(.001,.999))+p[...,0])
            scale = .01+.1*F.softplus(p[...,1])
            cdf = torch.sigmoid((self.thresholds-mean[...,None])/scale[...,None])
            prob = torch.cat([cdf[...,:1], cdf[...,1:]-cdf[...,:-1], 1-cdf[...,-1:]], -1).clamp_min(1e-7)
            probabilities.append(prob); pressures.append(mean); previous_h = horizon
        return {'count': torch.stack(outputs,1), 'lower': torch.stack(lowers,1), 'upper': torch.stack(uppers,1), 'pressure': torch.stack(pressures,1), 'probabilities': torch.stack(probabilities,1), 'current': current, 'balance_residual': torch.stack(balances,1)}


def objective(out, batch, scale, thresholds):
    truth = batch['target'].float(); y = truth[...,0]; p = truth[...,1]
    count = (out['count']-y).abs().mean()/scale
    total = (out['count'].sum(-1)-y.sum(-1)).abs().mean()/(scale*3)
    now = (out['current']-batch['current'][...,0]).abs().mean()/scale
    intervals = 0.
    for q,key in ((.1,'lower'),(.9,'upper')):
        error = y-out[key]; intervals = intervals+torch.maximum(q*error,(q-1)*error).mean()/scale
    labels = torch.bucketize(p.contiguous(), thresholds)
    ordinal = F.nll_loss(out['probabilities'].log().reshape(-1,4), labels.reshape(-1))
    pressure = (out['pressure']-p).abs().mean()
    return count+.25*total+.25*now+.1*intervals+.2*pressure+.02*ordinal
