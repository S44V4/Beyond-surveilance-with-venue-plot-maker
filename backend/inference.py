"""Accepted local checkpoints, bounded temporal context, explicit forecast provenance."""
from __future__ import annotations

import hashlib
import time
from collections import deque

import cv2
import numpy as np
import torch

from backend import store
from backend.schemas import Venue
from backend.balanced_ddpf import BalancedPredictor
from crowd_twin.data.strfe import CachedDDPFMetadata, CachedDDPFSequence, optical_flow_sequence
from crowd_twin.strfe_inference import STRFEGraphPredictor


class Pipeline:
    def __init__(self):
        torch.set_num_threads(2)
        self.ddpf=BalancedPredictor()
        self.temporal=None
        self.temporal_error=None
        try:
            # These checkpoints require retraining against the authoritative DDPF features.
            self.temporal=STRFEGraphPredictor(store.ROOT/"configs/balanced_temporal.yaml",store.ROOT/"models/balanced_strfe.pt",store.ROOT/"models/balanced_graph.pt")
            if self.temporal.strfe.input_channels!=256 or self.temporal.strfe_checkpoint.get("ddpf_sha256")!=self.ddpf.sha256:
                raise ValueError("Temporal checkpoint was not trained against the supplied DDPF features")
            if self.temporal.strfe_checkpoint.get("feature_preprocessing")!="valid-image-features-v1" or not self.temporal.strfe_checkpoint.get("deployment_approved"):
                raise ValueError("Temporal candidate requires evaluation and explicit deployment approval")
            if self.temporal.graph.checkpoint.get("source_strfe_sha256")!=self.temporal.strfe_sha256:
                raise ValueError("Graph checkpoint must match the accepted temporal checkpoint")
            if "synthetic" in str(self.temporal.strfe_checkpoint.get("checkpoint_kind")) or "synthetic" in str(self.temporal.graph.checkpoint.get("checkpoint_kind")):
                raise ValueError("Synthetic forecast checkpoints are blocked")
        except (FileNotFoundError,TypeError,ValueError,RuntimeError) as exc:
            self.temporal=None
            self.temporal_error="Compatible STRFE/graph weights must be trained against the supplied 256-channel DDPF features. Persistence forecasts are available."
        store.put("system","models",{"ready":True,"device":str(self.ddpf.device),"ddpf":{"kind":self.ddpf.checkpoint["checkpoint_kind"],"sha256":self.ddpf.sha256,"datasets":self.ddpf.checkpoint.get("source_datasets",[]),"image_size":self.ddpf.image_size},"temporal":{"ready":self.temporal is not None,"error":self.temporal_error,"venue_id":self.temporal.venue.venue_id if self.temporal else None,"unit":"normalized_frame_steps","sha256":self.temporal.strfe_sha256 if self.temporal else None,"graph_sha256":self.temporal.graph.checkpoint_sha256 if self.temporal else None},"verified_at":time.time()})

    def reset(self, venue:Venue):
        self.venue=venue
        self.window=deque(maxlen=self.temporal.context_frames if self.temporal else 8)
        self.previous=None
        self.previous_time=None
        self.previous_velocity={}
        h,w=self.ddpf.feature_size
        masks=[]
        for zone in venue.zones:
            mask=np.zeros((h,w),np.float32)
            polygon=zone.image_polygon or zone.polygon
            pts=np.round(np.array(polygon)*[w/1000,h/1000]).astype(np.int32)
            cv2.fillPoly(mask,[pts],1.0)
            masks.append(mask if zone.observed else mask*0)
        masks=np.stack(masks)
        total=masks.sum(axis=0)
        self.masks=masks/np.maximum(total,1)[None]
        self.compatible=self.temporal is not None and venue.id==self.temporal.venue.venue_id and [z.id for z in venue.zones]==list(self.temporal.venue.zone_ids)
        if self.compatible:
            for i,z in enumerate(venue.zones):
                r,c=divmod(i,3)
                expected=np.array([[c*1000/3,r*1000/3],[(c+1)*1000/3,r*1000/3],[(c+1)*1000/3,(r+1)*1000/3],[c*1000/3,(r+1)*1000/3]])
                if not z.observed or len(z.image_polygon or [])!=4 or not np.allclose(z.image_polygon,expected,atol=.1):
                    self.compatible=False
        if self.compatible:
            yy,xx=np.indices((h,w))
            cells=np.minimum(yy*3//h,2)*3+np.minimum(xx*3//w,2)
            self.masks=np.stack([(cells==i).astype(np.float32) for i in range(9)])

    def analyze(self, bgr, timestamp, idx, run_id):
        started=time.perf_counter()
        sample=self.ddpf.export_sequence([bgr],run_id,fps=1,inference_batch_size=1)
        rgb=cv2.cvtColor(cv2.resize(bgr,self.ddpf.image_size[::-1]),cv2.COLOR_BGR2RGB)
        dt=timestamp-self.previous_time if self.previous_time is not None else None
        cut=self.previous is not None and float(np.abs(rgb.astype(float)-self.previous).mean())>65
        if cut or (dt is not None and dt>8):
            self.window.clear()
            self.previous=None
            self.previous_velocity={}
        flow=optical_flow_sequence(np.stack([self.previous,rgb]),self.ddpf.feature_size)[1] if self.previous is not None else np.zeros_like(sample.flow[0])
        valid_motion=self.previous is not None and dt is not None and dt>0
        self.window.append((sample,flow,timestamp,idx))
        density=sample.density[0,0]
        count=float(density.sum())
        zones=[]
        for i,z in enumerate(self.venue.zones):
            mask=self.masks[i]
            n=float((density*mask).sum()) if z.observed else None
            speed=float((flow[2]*mask).sum()/max(mask.sum(),1)/dt) if valid_motion and z.observed else None
            direction=[float((flow[c]*mask).sum()/max(mask.sum(),1)/dt*self.ddpf.image_size[1-c]/self.ddpf.feature_size[1-c]) for c in [0,1]] if speed is not None else None
            acceleration=(speed-self.previous_velocity[z.id])/dt if speed is not None and self.previous_velocity.get(z.id) is not None else None
            self.previous_velocity[z.id]=speed
            zones.append({"id":z.id,"count":n,"density":n/z.area_m2 if n is not None and z.area_m2 and self.venue.geometry_kind=="calibrated" else None,"velocity":speed,"direction":direction,"acceleration":acceleration,"occupancy_ratio":n/z.capacity if n is not None else None,"pressure":None})
        learned=None
        learned_status="Custom venue: persistence baseline; compatible training required"
        if self.compatible:
            learned_status="Collecting distinct frames for temporal context"
        if self.compatible and len(self.window)==self.window.maxlen:
            entries=list(self.window)
            gaps=np.diff([x[2] for x in entries])
            if np.max(np.abs(gaps-gaps.mean()))<gaps.mean()*.25:
                sequence=CachedDDPFSequence(metadata=CachedDDPFMetadata(schema_version="1.0",sequence_id=run_id,source_sequence=run_id,split="test",venue_id=self.temporal.venue.venue_id,zone_ids=self.temporal.venue.zone_ids,fps=1/float(gaps.mean()),frame_indices=tuple(x[3] for x in entries),timestamps_ms=tuple(x[2]*1000 for x in entries),feature_channels=sample.metadata.feature_channels),visual_features=np.concatenate([x[0].visual_features for x in entries]),density=np.concatenate([x[0].density for x in entries]),localization_logits=np.concatenate([x[0].localization_logits for x in entries]),flow=np.stack([x[1] for x in entries]),zone_masks=self.masks)
                pred=self.temporal.predict_sequence(sequence)
                learned={"forecasts":[{"horizon":f["horizon_value"],"unit":f["horizon_unit"],"count":f["total_count"],"zones":[{"id":z["id"],"count":z["count"],"pressure":z["risk"]} for z in f["zones"]]} for f in pred["forecasts"]],"provenance":pred["pipeline"],"pressure_label":"ordinal pressure proxy; not emergency probability"}
                for zone,pz in zip(zones,pred["zones"]):
                    zone["pressure"]=pz["risk"]
                learned_status="STRFE + graph · research checkpoint · frame-step horizons"
            else:
                learned_status="Irregular frame timing: persistence baseline"
        baseline=[{"horizon":h,"unit":"seconds","count":sum(z["count"] or 0 for z in zones),"zones":[{"id":z["id"],"count":z["count"]} for z in zones]} for h in [5,15,30]]
        heat=cv2.resize(density,(48,30),interpolation=cv2.INTER_AREA)
        scale=max(float(np.percentile(heat,98)),1e-8)
        heat=np.clip(heat/scale,0,1)
        logits=sample.localization_logits[0,0]
        prob=1/(1+np.exp(-np.clip(logits,-30,30)))
        peaks=(prob==cv2.dilate(prob,np.ones((3,3),np.uint8))) & (prob>.5)
        ys,xs=np.where(peaks)
        points=self.ddpf.last_points
        quality=[]
        if not valid_motion:
            quality.append("Motion unavailable: initial frame or scene cut")
        if cut:
            quality.append("Scene cut detected; temporal history reset")
        if float(rgb.mean())<30:
            quality.append("Low-light input: estimates may be unreliable")
        if valid_motion and np.linalg.norm(np.median(flow[:2].reshape(2,-1),axis=1))>1.5:
            quality.append("Global image motion detected; fixed-camera interpretation may be invalid")
        self.previous,self.previous_time=rgb,timestamp
        return {"index":idx,"timestamp":timestamp,"count":count,"mapped_count":sum(z["count"] or 0 for z in zones),"unmapped_count":max(0,count-sum(z["count"] or 0 for z in zones)),"padding_count_excluded":self.ddpf.last_padding_count,"zones":zones,"heatmap":np.round(heat,3).tolist(),"heatmap_scale":"per-frame relative intensity","points":points,"forecasts":baseline,"forecast_method":"persistence_baseline","learned":learned,"learned_status":learned_status,"quality":quality,"evidence":"ddpf_video_estimate","units":{"count":"estimated people","velocity":"model-input pixels / second","acceleration":"model-input pixels / second²","density":"people / m² only when calibrated","pressure":"ordinal proxy / 100"},"latency_ms":(time.perf_counter()-started)*1000}
