import numpy as np
import pytest
from research.metrics import summarize,cluster_interval,pressure_scores
from research.protocol import PROTOCOL

def test_count_metrics_use_sum_before_total_absolute_error():
    truth=np.array([[[2.,.1],[3.,.8]],[[1.,.2],[4.,.7]]])
    pred=truth.copy();pred[:,:,0]+=[1.,-1.]
    result=summarize(truth,pred,[.25,.5,.75])
    assert result['zone_count_mae']==1
    assert result['zone_count_rmse']==1
    assert result['total_count_mae']==0
    assert result['pressure_macro_f1']==.75 # absent class has F1 zero

def test_paired_cluster_constant_improvement_ci():
    baseline=np.array([[3.,8.,14.],[4.,9.,15.]])
    result=cluster_interval(baseline-2,baseline,replicates=1000)
    assert result['delta']==pytest.approx(-2)
    assert result['delta_ci95']==pytest.approx([-2.,-2.])
    assert result['simultaneous_upper_one_sided95']==pytest.approx(-2)

def test_rmse_aggregates_squared_error_not_scene_rmse():
    values=np.array([[1.,9.]])
    result=cluster_interval(values,values,rmse=True,replicates=100)
    assert result['learned']==pytest.approx(np.sqrt(5))
    assert result['delta']==0

def test_cluster_uncertainty_is_invariant_to_duplicate_training_seeds():
    a=np.array([[1.,4.,2.]]);b=np.array([[2.,2.,3.]])
    one=cluster_interval(a,b,replicates=100)
    three=cluster_interval(np.repeat(a,3,axis=0),np.repeat(b,3,axis=0),replicates=100)
    assert one['delta_ci95']==pytest.approx(three['delta_ci95'])

def test_model_contract_forward_and_gradients():
    import torch
    from research.training import temporal,graph,topology
    torch.set_num_threads(2)
    model=temporal();g=graph();adj,static=topology(torch.device('cpu'))
    data={'visual_features':torch.zeros(1,8,256,8,8),'density':torch.zeros(1,8,1,8,8),'localization_logits':torch.zeros(1,8,1,8,8),'flow':torch.zeros(1,8,4,8,8),'zone_masks':torch.ones(9,8,8),'sensors':torch.zeros(1,8,0),'sensor_mask':torch.zeros(1,8,0)}
    out=model(**data)
    result=g(current_zone_features=out['current_zone_features'],future_zone_features=out['future_zone_features'],current_state_prior=out['current_zone_state'],future_state_prior=out['future_zone_state'],adjacency=adj,node_static=static)
    assert result['future_zone_state'].shape==(1,3,9,7)
    result['future_zone_state'].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

def test_protocol_has_full_training_and_three_seeds():
    assert len(PROTOCOL['seeds'])==3
    assert PROTOCOL['strfe_epochs']==40 and PROTOCOL['graph_epochs']==40
    assert PROTOCOL['horizons']==[5,15,30]
    assert PROTOCOL['test_sequences']==22

def test_training_checkpoint_and_handoff_lifecycle(tmp_path,monkeypatch):
    """Artificial tensors verify execution plumbing; never a model result."""
    import json
    import torch
    import research.training as training
    torch.set_num_threads(2)
    monkeypatch.setattr(training,'STUDY',tmp_path)
    monkeypatch.setattr(training,'PROTOCOL',{**PROTOCOL,'strfe_epochs':2,'graph_epochs':2,'min_epochs':1})
    (tmp_path/'target-calibration.json').write_text(json.dumps({'state_scale':[1]*7,'pressure_thresholds':[.25,.5,.75]}))
    (tmp_path/'protocol.json').write_text('{}')
    (tmp_path/'split-manifest.json').write_text(json.dumps({'ddpf_sha256':'unit-test-only'}))
    folder=tmp_path/'cache'/'unit';folder.mkdir(parents=True)
    targets=np.ones((40,9,7),np.float32);targets[...,6]=.5
    np.save(folder/'target.npy',targets)
    second=tmp_path/'cache'/'unit2';second.mkdir()
    np.save(second/'target.npy',targets)
    class UnitWindows:
        def __init__(self,split):self.index=[('unit',7),('unit2',8)]
        def __len__(self):return 2
        def __getitem__(self,i):
            return {'visual_features':torch.zeros(8,256,8,8),'density':torch.ones(8,1,8,8),'localization_logits':torch.zeros(8,1,8,8),'flow':torch.zeros(8,4,8,8),'zone_masks':torch.ones(9,8,8),'sensors':torch.zeros(8,0),'sensor_mask':torch.zeros(8,0),'current_target':torch.from_numpy(targets[7+i].copy()),'future_target':torch.from_numpy(targets[[7+i+h for h in PROTOCOL['horizons']]].copy()),'observed':torch.from_numpy(targets[7+i].copy()),'scene':self.index[i][0],'frame':7+i}
    monkeypatch.setattr(training,'Windows',UnitWindows)
    training.train_one('strfe',99,torch.device('cpu'))
    training.export_handoffs(99,torch.device('cpu'))
    training.train_one('graph',99,torch.device('cpu'))
    assert (tmp_path/'seed-99/strfe/best.pt').is_file()
    assert (tmp_path/'seed-99/graph/best.pt').is_file()
    history=json.loads((tmp_path/'seed-99/graph/history.json').read_text())
    assert len(history)==2
    training.train_one('graph',99,torch.device('cpu'))
    assert json.loads((tmp_path/'seed-99/graph/history.json').read_text())==history
    import research.evaluation as evaluation
    monkeypatch.setattr(evaluation,'STUDY',tmp_path)
    monkeypatch.setattr(evaluation,'PROTOCOL',{**PROTOCOL,'seeds':[99],'bootstrap_replicates':100})
    monkeypatch.setattr(evaluation,'Windows',UnitWindows)
    from research.protocol import sha
    (tmp_path/'CHECKPOINTS_SEALED.json').write_text(json.dumps({'checkpoints':{f'99/{kind}':sha(tmp_path/f'seed-99/{kind}/best.pt') for kind in ['strfe','graph']},'protocol_sha256':sha(tmp_path/'protocol.json'),'manifest_sha256':sha(tmp_path/'split-manifest.json')}))
    evaluation.evaluate()
    decision=json.loads((tmp_path/'evaluation/gate-decision.json').read_text())
    assert not decision['gates']['strfe_graph']['pass']
    assert len(json.loads((tmp_path/'evaluation/scene-horizon-metrics.json').read_text()))==18
