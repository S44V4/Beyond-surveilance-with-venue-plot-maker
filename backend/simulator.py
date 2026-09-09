"""Conserving, directed, finite-capacity zone-flow scenario engine.

Outputs are conditional simulations, not learned causal or emergency predictions.
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict

from backend.schemas import ScenarioRequest, Venue


def simulate(venue: Venue, initial: dict[str,float], request: ScenarioRequest, changed: bool, factor=1.0, cancelled=lambda:False):
    ids=[z.id for z in venue.zones]
    counts={z:max(0.0,float(initial[z])) for z in ids}
    capacities={z.id:z.capacity for z in venue.zones}
    entrances=[p for p in venue.portals if p.kind=="entrance"]
    queues={p.id:0.0 for p in entrances}
    exited=arrived=0.0
    initial_total=sum(counts.values())
    history=[]
    cumulative={p.id:0.0 for p in venue.portals}
    peak=sum(counts.values())
    peak_queue=0.0
    peak_zone=max(counts.values(),default=0)
    clearance=None
    max_residual=0.0
    dt=0.5
    steps=round(request.duration/dt)

    for step in range(steps+1):
        if cancelled():
            raise InterruptedError("Scenario cancelled")
        t=step*dt
        active=changed and request.start<=t and (request.end is None or t<request.end)
        edges=[]
        portal_caps={}
        for p in venue.portals:
            change=request.gates.get(p.id) if active else None
            opened=change.open if change else p.open
            cap=(change.capacity if change and change.capacity is not None else p.capacity)*factor if opened else 0.0
            portal_caps[p.id]=cap
            if p.kind!="entrance" and cap>0:
                edges.append((p.source,p.target,p,cap))
                if p.bidirectional:
                    edges.append((p.target,p.source,p,cap))
        # Shortest paths include occupancy/service delay; closed gates disappear.
        reverse=defaultdict(list)
        for a,b,p,cap in edges:
            reverse[b].append((a,p.travel_seconds+counts[a]/max(cap,1e-8)))
        dist={"outside":0.0}
        heap=[(0.0,"outside")]
        while heap:
            cost,b=heapq.heappop(heap)
            if cost!=dist[b]:
                continue
            for a,w in reverse[b]:
                if cost+w<dist.get(a,math.inf):
                    dist[a]=cost+w
                    heapq.heappush(heap,(cost+w,a))
        unreachable=sum(counts[z] for z in ids if z not in dist)
        residual=abs(initial_total+arrived-sum(counts.values())-sum(queues.values())-exited)
        max_residual=max(max_residual,residual)
        peak=max(peak,sum(counts.values()))
        peak_queue=max(peak_queue,sum(queues.values()))
        peak_zone=max(peak_zone,max(counts.values(),default=0))
        if sum(counts.values())<0.1 and request.mode=="evacuation" and clearance is None:
            clearance=t
        if step%2==0:
            history.append({"time":t,"counts":{z:round(counts[z],5) for z in ids},"total":sum(counts.values()),"exited":exited,"outside_queue":sum(queues.values()),"unreachable":unreachable,"gate_open":{p.id:portal_caps[p.id]>0 for p in venue.portals},"gate_totals":dict(cumulative),"conservation_residual":residual})
        if step==steps:
            break
        if request.mode=="continuous" and entrances:
            demand=request.inflow_per_second*dt/len(entrances)
            for p in entrances:
                queues[p.id]+=demand
                arrived+=demand
        proposed=[]
        for a in ids:
            choices=[(b,p,cap) for src,b,p,cap in edges if src==a and dist.get(b,math.inf)<dist.get(a,math.inf)]
            weights=[1/(p.travel_seconds+dist.get(b,0)+1) for b,p,cap in choices]
            denom=sum(weights)
            for (b,p,cap),w in zip(choices,weights):
                # Travel/service rate bounds: no instantaneous zone transfer.
                amount=min(cap*dt,counts[a]*dt/p.travel_seconds*w/denom)
                proposed.append([a,b,p.id,amount])
        # Shared physical capacity for opposing flows through one portal.
        shared=defaultdict(float)
        for a,b,p,n in proposed:
            shared[p]+=n
        for flow in proposed:
            flow[3]*=min(1.0,portal_caps[flow[2]]*dt/max(shared[flow[2]],1e-12))
        out=defaultdict(float)
        for a,b,p,n in proposed:
            out[a]+=n
        for flow in proposed:
            flow[3]*=min(1.0,counts[flow[0]]/max(out[flow[0]],1e-12))
        for p in entrances:
            proposed.append(["outside",p.target,p.id,min(queues[p.id],portal_caps[p.id]*dt)])
        received=defaultdict(float)
        for a,b,p,n in proposed:
            if b!="outside":
                received[b]+=n
        # Conservative receiving space: simultaneous outflow space is not borrowed.
        for flow in proposed:
            b=flow[1]
            if b!="outside":
                flow[3]*=min(1.0,max(0,capacities[b]-counts[b])/max(received[b],1e-12))
        for a,b,p,n in proposed:
            if a=="outside":
                queues[p]-=n
            else:
                counts[a]-=n
            if b=="outside":
                exited+=n
            else:
                counts[b]+=n
            cumulative[p]+=n
        if min(counts.values(),default=0)<-1e-7:
            raise ArithmeticError("Population conservation failed")
    return {"trajectory":history,"metrics":{"initial":initial_total,"remaining":sum(counts.values()),"departed":exited,"arrivals":arrived,"outside_queue":sum(queues.values()),"peak_queue":peak_queue,"peak_occupancy":peak,"peak_zone_count":peak_zone,"clearance_seconds":clearance,"unreachable":history[-1]["unreachable"],"conservation_error":max_residual,"gate_totals":cumulative}}


def compare(venue: Venue, initial: dict[str,float], request: ScenarioRequest, cancelled=lambda:False):
    unknown=set(request.gates)-{p.id for p in venue.portals}
    if unknown:
        raise ValueError(f"Unknown portals: {', '.join(sorted(unknown))}")
    if set(initial)!={z.id for z in venue.zones} or any(not math.isfinite(v) or v<0 for v in initial.values()):
        raise ValueError("Every zone requires a finite nonnegative initial count")
    baseline=simulate(venue,initial,request,False,cancelled=cancelled)
    scenario=simulate(venue,initial,request,True,cancelled=cancelled)
    sensitivity=[]
    if request.sensitivity:
        for factor in [0.8,1.2]:
            b=simulate(venue,initial,request,False,factor,cancelled)
            s=simulate(venue,initial,request,True,factor,cancelled)
            sensitivity.append({"capacity_factor":factor,"baseline_remaining":b["metrics"]["remaining"],"scenario_remaining":s["metrics"]["remaining"],"remaining_delta":s["metrics"]["remaining"]-b["metrics"]["remaining"]})
    return {"engine":"conserving-zone-flow-v1","evidence":"assumption_based_simulation","baseline":baseline,"scenario":scenario,"sensitivity":sensitivity,"delta":{k:scenario["metrics"][k]-baseline["metrics"][k] for k in ["remaining","departed","outside_queue","peak_queue","peak_zone_count","unreachable"]},"assumptions":["Portal capacities and zone capacities are configured assumptions unless independently measured.","People follow available lower-cost routes with zone travel-time limits; individual pedestrian trajectories are not modelled.","Closed entrances retain new demand in an outside queue.","Sensitivity varies all portal capacities by ±20%; this is not a calibrated confidence interval."]}
