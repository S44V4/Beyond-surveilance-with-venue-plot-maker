"""DDPF-Net Last Hope, restored from the user's training.py (cells 6–9).

Only model definitions are transcribed: no environment installation or training
side effects are executed. Parameter names are checked strictly against supplied weights.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import maximum_filter

from backend import store

if (store.ROOT/".python-deps").exists():
    sys.path.insert(0,str(store.ROOT/".python-deps"))
import timm

IMAGE_SIZE=512
DENSITY_SCALE=100.0


def norm_layer(channels):
    groups=32
    while channels%groups!=0 and groups>1:
        groups//=2
    return nn.GroupNorm(groups,channels)


class ConvGNAct(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size=3,stride=1,padding=1):
        super().__init__()
        self.block=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),norm_layer(out_channels),nn.SiLU(inplace=True))

    def forward(self,x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv1=ConvGNAct(channels,channels)
        self.conv2=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1,bias=False),norm_layer(channels))
        self.act=nn.SiLU(inplace=True)

    def forward(self,x):
        return self.act(self.conv2(self.conv1(x))+x)


class SwinTinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=timm.create_model("swin_tiny_patch4_window7_224",pretrained=False,features_only=True,out_indices=(0,1,2,3),img_size=IMAGE_SIZE)

    def forward(self,x):
        return [f.permute(0,3,1,2).contiguous() if f.shape[-1] in [96,192,384,768] else f for f in self.backbone(x)]


class SwinUNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False)
        self.conv1=ConvGNAct(1152,384)
        self.up2=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False)
        self.conv2=ConvGNAct(576,192)
        self.up3=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False)
        self.conv3=ConvGNAct(288,256)
        self.hires_stem=nn.Sequential(ConvGNAct(3,32),ConvGNAct(32,32))
        self.up4=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False)
        self.conv4=ConvGNAct(288,256)
        self.up5=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False)
        self.conv5=ConvGNAct(288,256)
        self.shared_conv=ResidualBlock(256)

    def forward(self,c1,c2,c3,c4,image):
        x=self.conv1(torch.cat([self.up1(c4),c3],dim=1))
        x=self.conv2(torch.cat([self.up2(x),c2],dim=1))
        x=self.conv3(torch.cat([self.up3(x),c1],dim=1))
        hires=self.hires_stem(image)
        x=self.conv4(torch.cat([self.up4(x),F.avg_pool2d(hires,2)],dim=1))
        x=self.conv5(torch.cat([self.up5(x),hires],dim=1))
        return self.shared_conv(x)


class Head(nn.Module):
    def __init__(self,channels=1,activation="identity"):
        super().__init__()
        self.head=nn.Sequential(ConvGNAct(256,128),ConvGNAct(128,64),nn.Conv2d(64,channels,1))
        self.activation=activation

    def forward(self,x):
        out=self.head(x)
        return F.softplus(out) if self.activation=="softplus" else .5*torch.tanh(out) if self.activation=="offset" else out


class DDPFNetLastHope(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=SwinTinyBackbone()
        self.decoder=SwinUNetDecoder()
        self.density_head=Head(activation="softplus")
        self.heatmap_head=Head()
        self.offset_head=Head(2,"offset")

    def forward(self,x):
        features=self.encoder(x)
        shared=self.decoder(*features,x)
        # Export a bounded-size representation without changing learned weights.
        return {"density":self.density_head(shared),"heatmap_logits":self.heatmap_head(shared),"offset":self.offset_head(shared),"features":F.adaptive_avg_pool2d(shared,(32,32))}


def preprocess(bgr):
    h,w=bgr.shape[:2]
    scale=IMAGE_SIZE/max(h,w)
    nw,nh=max(1,int(w*scale)),max(1,int(h*scale))
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    resized=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_LINEAR)
    padded=cv2.copyMakeBorder(resized,0,IMAGE_SIZE-nh,0,IMAGE_SIZE-nw,cv2.BORDER_CONSTANT,value=(0,0,0))
    values=(padded.astype(np.float32)/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return torch.from_numpy(values.transpose(2,0,1)).unsqueeze(0),nw,nh


class BalancedPredictor:
    def __init__(self):
        self.path=store.ROOT/"models/balanced_ddpf.pt"
        checkpoint=torch.load(self.path,map_location="cpu",weights_only=True)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model=DDPFNetLastHope()
        self.model.load_state_dict(checkpoint["model_state_dict"],strict=True)
        self.model.to(self.device).eval()
        with self.path.open("rb") as handle:
            self.sha256=hashlib.file_digest(handle,"sha256").hexdigest()
        self.checkpoint={"checkpoint_kind":"user_supplied_balanced_ddpf","source_datasets":["ShanghaiTech Part B (training.py configuration; not independently re-evaluated)"],"epoch":checkpoint["epoch"]}
        self.threshold=float(checkpoint.get("best_conf_threshold",.25))
        self.image_size=(512,512)
        self.feature_size=(32,32)

    @torch.inference_mode()
    def export_sequence(self,frames,sequence_id,fps,inference_batch_size=1):
        if len(frames)!=1:
            raise ValueError("Balanced DDPF uses one-frame bounded inference")
        tensor,nw,nh=preprocess(frames[0])
        # FP16 overflows on this checkpoint. BF16 preserves range on supported GPUs;
        # all other devices use FP32, including older CUDA cards.
        with torch.autocast(device_type=self.device.type,dtype=torch.bfloat16,enabled=self.device.type=="cuda" and torch.cuda.is_bf16_supported()):
            out=self.model(tensor.to(self.device))
        if any(not torch.isfinite(v).all().item() for v in out.values()):
            raise ValueError("DDPF produced non-finite outputs; no prediction was saved")
        raw_density=out["density"][0,0].float().cpu().numpy()/DENSITY_SCALE
        # Padding must never be assigned to real venue zones.
        valid=raw_density[:nh,:nw]
        density=cv2.resize(valid,self.feature_size[::-1],interpolation=cv2.INTER_AREA)
        density*=float(valid.sum())/max(float(density.sum()),1e-12)
        logits=out["heatmap_logits"][0,0].float().cpu().numpy()[:nh,:nw]
        offsets=out["offset"][0].float().cpu().numpy()[:,:nh,:nw]
        probabilities=1/(1+np.exp(-np.clip(logits,-30,30)))
        ys,xs=np.where((probabilities==maximum_filter(probabilities,size=4)) & (probabilities>self.threshold))
        pts=[]
        for x,y in zip(xs,ys):
            px,py=x+float(offsets[0,y,x]),y+float(offsets[1,y,x])
            if 0<=px<nw and 0<=py<nh:
                pts.append([px/nw*1000,py/nh*1000])
        # Align feature cells to the same valid image coordinates as density.
        # grid_sample accounts for fractional letterbox boundaries at 32x32.
        fy=(torch.arange(32,device=self.device)+.5)/32*(nh/512)*2-1
        fx=(torch.arange(32,device=self.device)+.5)/32*(nw/512)*2-1
        gy,gx=torch.meshgrid(fy,fx,indexing="ij")
        grid=torch.stack([gx,gy],-1)[None]
        features=F.grid_sample(out["features"].float(),grid,align_corners=False,padding_mode="border").cpu().numpy()
        self.last_points=pts[:2000]
        self.last_padded_count=float(raw_density.sum())
        self.last_padding_count=float(raw_density.sum()-valid.sum())
        return SimpleNamespace(density=density[None,None],visual_features=features,localization_logits=cv2.resize(logits,self.feature_size[::-1])[None,None],flow=np.zeros((1,4,*self.feature_size),np.float32),metadata=SimpleNamespace(feature_channels=256))
