import pytest
from backend.defaults import default_venue
from backend.schemas import ScenarioRequest,Venue
from backend.simulator import compare


def request(**kwargs):
    return ScenarioRequest(run_id="test",snapshot_index=0,idempotency_key="test-key-123",sensitivity=False,**kwargs)


def scene():
    v=Venue.model_validate(default_venue())
    return v,{z.id:20.0 for z in v.zones}


def test_no_action_is_identical():
    v,n=scene();r=compare(v,n,request())
    assert r['baseline']==r['scenario']
    assert r['scenario']['metrics']['conservation_error']<1e-8


def test_all_exits_closed_conserves_people():
    v,n=scene();r=compare(v,n,request(gates={p.id:{'open':False} for p in v.portals if p.kind=='exit'}))
    metrics=r['scenario']['metrics']
    assert metrics['departed']==0
    assert metrics['remaining']==sum(n.values())
    assert metrics['unreachable']==sum(n.values())
    assert metrics['clearance_seconds'] is None
    assert r['baseline']['metrics']['departed']>0


def test_closed_entrance_retains_arrivals():
    v,n=scene();r=compare(v,n,request(mode='continuous',inflow_per_second=2,gates={'I-W':{'open':False}},duration=30))
    m=r['scenario']['metrics']
    assert m['outside_queue']==pytest.approx(60)
    assert m['arrivals']==pytest.approx(60)
    assert m['conservation_error']<1e-7


def test_timed_closure_and_reopening():
    v,n=scene();r=compare(v,n,request(gates={'E-N':{'open':False}},start=10,end=20,duration=30))
    ts=r['scenario']['trajectory']
    assert ts[9]['gate_open']['E-N'] is True
    assert ts[10]['gate_open']['E-N'] is False
    assert ts[20]['gate_open']['E-N'] is True
    assert ts[20]['gate_totals']['E-N']==pytest.approx(ts[10]['gate_totals']['E-N'])


def test_receiving_space_does_not_destroy_people():
    v,n=scene();n={z:1000.0 for z in n};r=compare(v,n,request(mode='continuous',inflow_per_second=5,duration=60))
    assert r['scenario']['metrics']['conservation_error']<1e-7
    assert all(c>=0 for t in r['scenario']['trajectory'] for c in t['counts'].values())


def test_portal_capacity_bounds():
    v,n=scene();r=compare(v,n,request(duration=30))
    ts=r['scenario']['trajectory']
    for a,b in zip(ts,ts[1:]):
        for p in v.portals:
            assert b['gate_totals'][p.id]-a['gate_totals'][p.id]<=p.capacity*(b['time']-a['time'])+1e-8


def test_unknown_portal_rejected():
    v,n=scene()
    with pytest.raises(ValueError,match='Unknown portals'):
        compare(v,n,request(gates={'missing':{'open':False}}))


def test_empty_venue_clears_at_zero():
    v,n=scene();r=compare(v,{z:0 for z in n},request())
    assert r['scenario']['metrics']['clearance_seconds']==0


def test_sensitivity_is_paired():
    v,n=scene();r=compare(v,n,ScenarioRequest(run_id='t',snapshot_index=0,idempotency_key='testing12',sensitivity=True,duration=30))
    assert len(r['sensitivity'])==2
    assert all(x['remaining_delta']==0 for x in r['sensitivity'])


def test_geometry_and_units_validation():
    payload=default_venue();payload['geometry_kind']='calibrated'
    with pytest.raises(ValueError,match='Calibrated'):
        Venue.model_validate(payload)
    payload=default_venue();payload['portals'][0]['target']='invalid'
    with pytest.raises(ValueError):Venue.model_validate(payload)

