import numpy as np
import pytest
import torch
from forecast_v2.common import write, sha, verify_data
from forecast_v2.model import RegionalForecaster, objective
from forecast_v2.train import train
from forecast_v2.data import Windows


def tiny_data(root):
    root.mkdir(); rng=np.random.default_rng(7); files={}
    for sid in ('a','b','c',*[f'd{i}' for i in range(8)]):
        x=rng.normal(size=(14,9,265)).astype('float32')
        observed=np.stack([rng.uniform(2,10,(14,9)),rng.uniform(.1,.7,(14,9))],-1).astype('float32')
        y=observed.copy();y[...,0]*=1.2
        np.savez(root/f'{sid}.npz',x=x,observed=observed,y=y,frames=np.arange(14))
        files[f'{sid}.npz']=sha(root/f'{sid}.npz')
    write(root/'calibration.json',{'x_mean':[0.]*265,'x_std':[1.]*265,'count_coefficients':[[1.,0.,0.]]*9,'count_scale':10.,'pressure_thresholds':[.2,.4,.6]})
    files['calibration.json']=sha(root/'calibration.json')
    write(root/'manifest.json',{'files':files,'splits':{'train':['a','b'],'val':['c'],'calibration':[f'd{i}' for i in range(8)]},'excluded_exposed_test':['never'],'input_dim':265})


def test_constraints_and_gradients():
    torch.set_num_threads(2);torch.manual_seed(5)
    m=RegionalForecaster(dim=16);x=torch.randn(2,8,9,265)
    a=torch.rand(2,8,9)*20;o=torch.rand(2,8,9,2)
    out=m(x,a,o)
    assert out['count'].shape==(2,3,9)
    assert (out['lower']<=out['count']).all() and (out['count']<=out['upper']).all()
    assert out['count'].min()>=0
    assert out['balance_residual'].abs().max()<1e-4
    torch.testing.assert_close(out['probabilities'].sum(-1),torch.ones(2,3,9))
    batch={'target':torch.rand(2,3,9,2),'current':torch.rand(2,9,2)}
    loss=objective(out,batch,20,m.thresholds);loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_labels_never_enter_inputs(tmp_path):
    tiny_data(tmp_path/'data');d=Windows(tmp_path/'data','train',4,[1,2],1)
    before=d[0]['x'].clone();sid,_=d.index[0];d.arrays[sid]['y'][:]=9999
    assert torch.equal(before,d[0]['x'])


def test_exposed_test_rejected(tmp_path):
    tiny_data(tmp_path/'data')
    from forecast_v2.common import read
    p=tmp_path/'data/manifest.json';m=read(p);m['excluded_exposed_test']=['a'];write(p,m)
    with pytest.raises(ValueError,match='forbidden'):verify_data(tmp_path/'data')


def test_pause_resume_matches_uninterrupted(tmp_path):
    data=tmp_path/'data';tiny_data(data)
    cfg={'seed':17,'mode':'direct','epochs':1,'min_epochs':1,'patience':2,'context':4,'horizons':[1,2],'stride':1,'dim':16,'micro_batch':2,'accumulation':2,'checkpoint_steps':1,'precision':'fp32','threads':2}
    full=tmp_path/'full';resumed=tmp_path/'resumed'
    train(data,full,cfg,device_name='cpu')
    result=train(data,resumed,cfg,max_steps=2,device_name='cpu')
    assert result['paused'] and not (resumed/'COMPLETE.json').exists()
    train(data,resumed,cfg,device_name='cpu')
    a=torch.load(full/'last.pt',weights_only=False);b=torch.load(resumed/'last.pt',weights_only=False)
    assert a['state']['global_step']==b['state']['global_step']
    for name in a['model']:torch.testing.assert_close(a['model'][name],b['model'][name],rtol=0,atol=0)
    with pytest.raises(ValueError,match='contract changed'):
        train(data,resumed,{**cfg,'learning_rate':.003},device_name='cpu')


def test_metric_total_aggregates_before_absolute():
    from forecast_v2.evaluate import metrics
    y=np.zeros((1,2,2));y[...,0]=10
    value=metrics(y,np.array([[11.,9.]]),np.zeros((1,2)),[.2,.4,.6])
    assert value['zone_mae']==1 and value['total_mae']==0


def test_long_horizon_cannot_remove_more_than_available():
    m=RegionalForecaster(dim=16,horizons=(15,60,120))
    with torch.no_grad():m.exchange.bias.fill_(15)
    out=m(torch.randn(1,8,9,265),torch.ones(1,8,9),torch.rand(1,8,9,2))
    assert torch.isfinite(out['count']).all() and out['count'].min()>=0


def test_complete_evaluation_inference_and_stress(tmp_path):
    from forecast_v2.evaluate import evaluate
    from forecast_v2.stress import stress
    from forecast_v2.inference import Predictor
    from forecast_v2.common import read
    data=tmp_path/'data';tiny_data(data);run=tmp_path/'run'
    cfg={'seed':21,'mode':'transport','epochs':1,'min_epochs':1,'patience':2,'context':8,'horizons':[1,2],'stride':1,'dim':16,'micro_batch':2,'accumulation':2,'checkpoint_steps':1,'precision':'fp32','threads':2}
    train(data,run,cfg,device_name='cpu');evaluate(data,run,'cpu');stress(data,run,'cpu')
    assert (run/'development-evaluation/REPORT.md').exists()
    assert read(run/'development-evaluation/metrics.json')['independent_test_required']
    predictor=Predictor(run,'cpu')
    with np.load(data/'c.npz') as f:out=predictor.predict(f['x'][:8],f['observed'][:8])
    assert out['count'].shape==(2,9) and not out['deployment_approved']
    assert np.all(out['lower']<=out['upper'])
