from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend import store
from backend.defaults import default_venue
from backend.schemas import RunRequest, ScenarioRequest, Venue


@asynccontextmanager
async def lifespan(app):
    store.initialize()
    if not store.listing("venue"):
        demo=default_venue()
        store.put("venue",demo["id"],demo)
        reference=default_venue()
        reference.update(id="dronecrowd-image-grid-3x3-v1",name="Camera grid · 9 zones",description="3×3 camera partition. Map arrangement and gates are illustrative assumptions; compatible learned forecast weights require evaluation.")
        store.put("venue",reference["id"],reference)
    reference=store.get("venue","dronecrowd-image-grid-3x3-v1")
    if reference and reference["name"]=="Camera grid · trained forecast":
        reference.update(name="Camera grid · 9 zones",version=reference["version"]+1,description="3×3 camera partition. Map and gates are assumptions; compatible learned forecasts require evaluation.")
        store.put("venue",reference["id"],reference)
    proc=None
    if os.environ.get("CROWD_NO_WORKER")!="1":
        logs=(store.DATA/"worker.log").open("a",encoding="utf-8")
        proc=subprocess.Popen([sys.executable,"-u","-m","backend.worker"],cwd=store.ROOT,stdout=logs,stderr=logs,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        app.state.worker=proc
    yield
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        logs.close()


app=FastAPI(title="Beyond Surveillance",version="1.0.0",lifespan=lifespan)


@app.middleware("http")
async def local_origin(request:Request,call_next):
    origin=request.headers.get("origin")
    if request.method not in {"GET","HEAD","OPTIONS"} and origin and origin not in {"http://127.0.0.1:5173","http://localhost:5173","http://127.0.0.1:8010","http://localhost:8010"}:
        return JSONResponse({"detail":"This local workspace does not accept requests from external websites"},403)
    response=await call_next(request)
    response.headers["X-Content-Type-Options"]="nosniff"
    response.headers["Referrer-Policy"]="same-origin"
    return response


def require(kind,id):
    value=store.get(kind,id)
    if value is None:
        raise HTTPException(404,f"{kind.replace('_',' ').capitalize()} not found")
    return value


def require_job(id):
    value=store.get_job(id)
    if value is None:
        raise HTTPException(404,"Run not found")
    return value


@app.get("/api/health")
def health():
    models=store.get("system","models")
    return {"status":"ready","worker_alive":getattr(app.state,"worker",None) is not None and app.state.worker.poll() is None,"checkpoints_present":{name:(store.ROOT/f"models/{filename}.pt").is_file() for name,filename in [("ddpf","balanced_ddpf"),("strfe","balanced_strfe"),("graph","balanced_graph")]},"models":models,"mode":"local research application","version":"1.0.0"}


@app.get("/api/venues")
def venues():
    return store.listing("venue",100)


@app.post("/api/venues")
def save_venue(venue:Venue):
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row=db.execute("SELECT body FROM objects WHERE kind='venue' AND id=?",(venue.id,)).fetchone()
        old=json.loads(row[0]) if row else None
        if old and venue.version!=old["version"]:
            raise HTTPException(409,"Venue changed since you opened it. Reload before saving.")
        value=venue.model_dump()
        value["version"]=(old["version"]+1) if old else 1
        db.execute("INSERT OR REPLACE INTO objects VALUES ('venue',?,?,?)",(venue.id,json.dumps(value,allow_nan=False),time.time()))
    return value


@app.post("/api/venues/validate")
def validate_venue(venue:Venue):
    return venue.model_dump()


@app.post("/api/assets")
async def upload_plan(file:UploadFile=File(...)):
    raw=await file.read(10*1024*1024+1)
    if len(raw)>10*1024*1024:
        raise HTTPException(413,"Floor plan must be under 10 MB")
    img=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if img is None or img.size>80_000_000:
        raise HTTPException(422,"Use a PNG or JPEG floor plan under 25 megapixels")
    id=store.uid()
    folder=store.DATA/"assets"
    folder.mkdir(exist_ok=True)
    cv2.imwrite(str(folder/f"{id}.jpg"),img)
    return {"url":f"/api/assets/{id}"}


@app.get("/api/assets/{id}")
def asset(id:str):
    if not id.isalnum() or not (store.DATA/"assets"/f"{id}.jpg").is_file():
        raise HTTPException(404,"Floor plan not found")
    return FileResponse(store.DATA/"assets"/f"{id}.jpg",media_type="image/jpeg")


@app.post("/api/videos")
async def upload_video(file:UploadFile=File(...)):
    suffix=Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4",".mov",".avi",".mkv",".webm",".m4v"}:
        raise HTTPException(422,"Use MP4, MOV, AVI, MKV, M4V, or WebM video")
    id=store.uid()
    folder=store.DATA/"videos"
    folder.mkdir(exist_ok=True)
    path=folder/f"{id}{suffix}"
    size=0
    digest=hashlib.sha256()
    try:
        with path.open("wb") as out:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>1024*1024*1024:
                    raise HTTPException(413,"Video must be under 1 GB")
                out.write(chunk)
                digest.update(chunk)
        cap=cv2.VideoCapture(str(path))
        fps=cap.get(cv2.CAP_PROP_FPS)
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok,frame=cap.read()
        cap.release()
        if not ok or not np.isfinite(fps) or fps<=0 or frame_count<1:
            raise HTTPException(422,"Could not decode a video with valid timing metadata. Re-encode as MP4.")
        duration=frame_count/fps
        if duration>3600:
            raise HTTPException(422,"Use a video of 60 minutes or less")
        poster=folder/f"{id}.jpg"
        cv2.imwrite(str(poster),frame)
        value={"id":id,"name":Path(file.filename or "video").name[:150],"file":f"videos/{id}{suffix}","poster":f"/api/videos/{id}/poster","url":f"/api/videos/{id}/media","size":size,"sha256":digest.hexdigest(),"fps":fps,"duration":duration,"frame_count":frame_count,"width":frame.shape[1],"height":frame.shape[0],"created":time.time()}
        store.put("video",id,value)
        return value
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@app.get("/api/videos/{id}/{kind}")
def media(id:str,kind:str):
    video=require("video",id)
    if kind=="poster":
        return FileResponse(store.DATA/"videos"/f"{id}.jpg",media_type="image/jpeg")
    if kind=="media":
        return FileResponse(store.DATA/video["file"])
    raise HTTPException(404,"Media not found")


@app.post("/api/runs")
def run(request:RunRequest):
    require("video",request.video_id)
    venue=require("venue",request.venue_id)
    if not (store.ROOT/"models/balanced_ddpf.pt").is_file():
        raise HTTPException(503,"DDPF weights are missing. Run scripts/import-models.ps1 or configure the accepted checkpoint.")
    body=request.model_dump(exclude={"idempotency_key"})
    body["venue"]=venue
    try:
        return store.create_job("analysis",body,request.idempotency_key)
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc


@app.get("/api/runs")
def runs(limit:int=Query(30,ge=1,le=100),offset:int=Query(0,ge=0)):
    return [{**j,"video":store.get("video",j["body"].get("video_id","")),"result":store.get("run_result",j["id"])} for j in store.jobs("analysis",limit,offset)]


@app.get("/api/runs/{id}")
def get_run(id:str):
    value=require_job(id)
    return {**value,"video":store.get("video",value["body"].get("video_id","")),"result":store.get("run_result",id)}


@app.post("/api/jobs/{id}/cancel")
def cancel(id:str):
    value=require_job(id)
    if value["status"] in {"queued","running"}:
        store.update_job(id,"cancelled" if value["status"]=="queued" else "cancelling")
    return require_job(id)


@app.post("/api/jobs/{id}/retry")
def retry(id:str):
    value=require_job(id)
    if value["status"] not in {"failed","cancelled"}:
        raise HTTPException(409,"Only failed or cancelled jobs can be retried")
    return store.create_job(value["kind"],value["body"],store.uid())


@app.get("/api/runs/{id}/timeline")
def timeline(id:str,offset:int=Query(0,ge=0),limit:int=Query(300,ge=1,le=1000)):
    require_job(id)
    return store.frames(id,offset,limit,True)


@app.get("/api/runs/{id}/snapshots/{idx}")
def snapshot(id:str,idx:int):
    value=store.frame(id,idx)
    if value is None:
        raise HTTPException(404,"Snapshot not available")
    return value


@app.get("/api/runs/{id}/frames/{idx}.jpg")
def frame_image(id:str,idx:int):
    require_job(id)
    if not id.isalnum() or idx<0 or not (store.DATA/"runs"/id/f"{idx}.jpg").is_file():
        raise HTTPException(404,"Frame not available")
    return FileResponse(store.DATA/"runs"/id/f"{idx}.jpg",media_type="image/jpeg")


@app.post("/api/scenarios")
def scenario(request:ScenarioRequest):
    run=require_job(request.run_id)
    if run["kind"]!="analysis":
        raise HTTPException(422,"Scenarios must start from an analysis snapshot")
    snapshot=store.frame(request.run_id,request.snapshot_index)
    if snapshot is None:
        raise HTTPException(422,"Choose an available snapshot")
    venue=Venue.model_validate(run["body"]["venue"])
    unknown=set(request.gates)-{p.id for p in venue.portals}
    if unknown:
        raise HTTPException(422,f"Unknown portal: {', '.join(unknown)}")
    if request.mode=="continuous" and request.inflow_per_second>0 and not any(p.kind=="entrance" for p in venue.portals):
        raise HTTPException(422,"Add an entrance before specifying outside arrivals")
    observations={z["id"]:z["count"] for z in snapshot["zones"]}
    initial={}
    for zone in venue.zones:
        value=observations.get(zone.id)
        if value is None:
            value=request.unobserved_counts.get(zone.id)
        if value is None:
            raise HTTPException(422,f"Specify an initial count for unobserved zone {zone.name}")
        initial[zone.id]=float(value)
    body={"request":request.model_dump(),"venue":venue.model_dump(),"initial":initial,"source_timestamp":snapshot["timestamp"],"source_evidence":snapshot["evidence"]}
    try:
        return store.create_job("scenario",body,request.idempotency_key)
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc


@app.get("/api/scenarios")
def scenarios(limit:int=Query(30,ge=1,le=100),offset:int=Query(0,ge=0)):
    return store.jobs("scenario",limit,offset)


@app.get("/api/scenarios/{id}")
def get_scenario(id:str):
    return {**require_job(id),"result":store.get("scenario_result",id)}


@app.post("/api/examples")
def example():
    venue=require("venue","atrium-study")
    run=store.create_job("analysis",{"venue":venue,"venue_id":venue["id"],"sample_fps":1,"camera_name":"Illustrative scenario","example":True},f"illustrative-scenario-v{venue['version']}",status="completed")
    weights=[18,32,22,41,72,28,17,36,20]
    counts=[weights[i%len(weights)] for i in range(len(venue["zones"]))]
    # One explicit hypothetical state, never an invented video timeline.
    store.add_frame(run["id"],0,{"index":0,"timestamp":0,"count":sum(counts),"mapped_count":sum(counts),"unmapped_count":0,"zones":[{"id":z["id"],"count":counts[i],"density":None,"velocity":None,"direction":None,"acceleration":None,"occupancy_ratio":counts[i]/z["capacity"],"pressure":None} for i,z in enumerate(venue["zones"])],"heatmap":[],"points":[],"forecasts":[],"learned":None,"learned_status":"Illustrative input; no model prediction","quality":["Hypothetical initial population. Gate locations and capacities are assumptions."],"evidence":"hypothetical_scenario_input","units":{"count":"assumed people"},"latency_ms":0})
    store.update_job(run["id"],"completed",1)
    return get_run(run["id"])


@app.get("/api/runs/{id}/export")
def export(id:str):
    run=get_run(id)
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("run.json",json.dumps(run,indent=2))
        z.writestr("venue.json",json.dumps(run["body"]["venue"],indent=2))
        with z.open("timeline.jsonl","w") as timeline:
            offset=0
            while True:
                chunk=store.frames(id,offset,300)
                for f in chunk["items"]:
                    timeline.write((json.dumps(f)+"\n").encode())
                offset+=len(chunk["items"])
                if offset>=chunk["total"]:
                    break
        for scenario in store.jobs("scenario",10000):
            if scenario["body"]["request"]["run_id"]==id:
                z.writestr(f"scenarios/{scenario['id']}.json",json.dumps({"job":scenario,"result":store.get("scenario_result",scenario["id"])},indent=2))
        z.writestr("README.txt","Beyond Surveillance research export. Video estimates, persistence/learned forecasts, and assumption-based simulations are distinct evidence types. Frame-step model horizons are not seconds. This export contains no validated emergency probabilities. See run.json for checkpoint and venue provenance.")
    return Response(buffer.getvalue(),media_type="application/zip",headers={"Content-Disposition":f'attachment; filename="beyond-surveillance-{id[:8]}.zip"'})


if (store.ROOT/"dist").is_dir():
    app.mount("/",StaticFiles(directory=store.ROOT/"dist",html=True),name="dashboard")
