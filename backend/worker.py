from __future__ import annotations

import time
import traceback

import cv2

from backend import store
from backend.schemas import ScenarioRequest, Venue
from backend.simulator import compare


def cancelled(id):
    return store.get_job(id)["status"] in {"cancelling","cancelled"}


def process_video(job,pipeline):
    id,body=job["id"],job["body"]
    video=store.get("video",body["video_id"])
    venue=Venue.model_validate(body["venue"])
    pipeline.reset(venue)
    cap=cv2.VideoCapture(str(store.DATA/video["file"]))
    if not cap.isOpened():
        raise ValueError("Video can no longer be decoded; upload it again")
    folder=store.DATA/"runs"/id
    folder.mkdir(parents=True,exist_ok=True)
    with store.connect() as db:
        db.execute("DELETE FROM frames WHERE run_id=?",(id,))
    frame_index=idx=0
    next_time=0.0
    last_timestamp=-1.0
    sampled_timestamp=-1.0
    fps=video["fps"]
    last_frame=None
    started=time.perf_counter()
    try:
        while True:
            if cancelled(id):
                raise InterruptedError("Analysis cancelled")
            ok,frame=cap.read()
            if not ok:
                break
            raw=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            timestamp=raw if raw>=0 and raw>last_timestamp else frame_index/fps
            if timestamp<=last_timestamp:
                timestamp=last_timestamp+1/fps
            last_timestamp=timestamp
            last_frame=frame
            if timestamp+1e-5>=next_time:
                result=pipeline.analyze(frame,timestamp,idx,id)
                result["frame_url"]=f"/api/runs/{id}/frames/{idx}.jpg"
                preview=cv2.resize(frame,(min(960,frame.shape[1]),round(frame.shape[0]*min(960,frame.shape[1])/frame.shape[1])))
                cv2.imwrite(str(folder/f"{idx}.jpg"),preview,[cv2.IMWRITE_JPEG_QUALITY,82])
                store.add_frame(id,idx,result)
                sampled_timestamp=timestamp
                idx+=1
                next_time=timestamp+1/body["sample_fps"]
                store.progress(id,min(.99,timestamp/max(video["duration"],.1)))
            frame_index+=1
        if last_frame is None:
            raise ValueError("Video contains no readable frames")
        if sampled_timestamp<last_timestamp-1e-4:
            result=pipeline.analyze(last_frame,last_timestamp,idx,id)
            result["frame_url"]=f"/api/runs/{id}/frames/{idx}.jpg"
            cv2.imwrite(str(folder/f"{idx}.jpg"),last_frame,[cv2.IMWRITE_JPEG_QUALITY,82])
            store.add_frame(id,idx,result)
            idx+=1
        if cancelled(id):
            raise InterruptedError("Analysis cancelled")
        if frame_index < video["frame_count"]*.98:
            raise ValueError("Decoder stopped before the expected end; partial results retained. Re-encode and retry.")
        store.put("run_result",id,{"sample_count":idx,"decoded_frames":frame_index,"processed_seconds":last_timestamp,"wall_seconds":time.perf_counter()-started,"models":store.get("system","models"),"venue_version":venue.version,"quality_note":"Camera motion and out-of-domain imagery can reduce accuracy; image motion is not calibrated physical speed."})
    finally:
        cap.release()


def process_scenario(job):
    body=job["body"]
    request=ScenarioRequest.model_validate(body["request"])
    venue=Venue.model_validate(body["venue"])
    initial=body["initial"]
    result=compare(venue,initial,request,lambda:cancelled(job["id"]))
    result.update({"id":job["id"],"name":request.name,"snapshot_index":request.snapshot_index,"source_timestamp":body["source_timestamp"],"source_evidence":body["source_evidence"],"run_id":request.run_id,"venue":body["venue"],"request":body["request"]})
    store.put("scenario_result",job["id"],result)


def main():
    store.initialize()
    # A single local worker owns the queue. A hard restart explicitly requeues unfinished work.
    with store.connect() as db:
        db.execute("UPDATE jobs SET status='queued',progress=0 WHERE status='running'")
        db.execute("UPDATE jobs SET status='cancelled' WHERE status='cancelling'")
    pipeline=None
    while True:
        store.put("system","worker",{"heartbeat":time.time()})
        job=store.next_job()
        if not job:
            time.sleep(.5)
            continue
        try:
            if job["kind"]=="analysis":
                if pipeline is None:
                    from backend.inference import Pipeline
                    pipeline=Pipeline()
                process_video(job,pipeline)
            elif job["kind"]=="scenario":
                process_scenario(job)
            else:
                raise ValueError("Unsupported job")
            if cancelled(job["id"]):
                raise InterruptedError("Job cancelled")
            store.update_job(job["id"],"completed",1)
        except InterruptedError:
            store.update_job(job["id"],"cancelled")
        except Exception as exc:
            traceback.print_exc()
            store.update_job(job["id"],"failed",error=str(exc)[:1000])


if __name__=="__main__":
    main()

