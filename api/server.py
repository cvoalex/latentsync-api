"""latentsync-api — internal HTTP API for LatentSync 1.6 lipsync jobs.

Same shape as ltx-api/longcat-api (submit -> poll -> fetch), but this service
has NO local GPU path: LatentSync was never installed on prod, so DISPATCH is
runpod-only by design. The server is a thin dispatcher — uploads land on
local disk, jobs go to the combo-latentsync RunPod serverless endpoint via R2
(renderjobs/latentsync/<job>/…), the lipsynced mp4 comes back to local disk.

  GET    /healthz
  POST   /v1/uploads                (multipart "file" -> {"path": ...})
  POST   /v1/jobs/lipsync           ({video_path, audio_path, ...} -> JobRecord)
  GET    /v1/jobs [/{id}] [/{id}/video]
  DELETE /v1/jobs/{id}              (cancel a queued job)

Env (EnvironmentFile=/data/latentsync-api/.env, chmod 600):
  RUNPOD_API_KEY, LATENTSYNC_RUNPOD_ENDPOINT_ID,
  R2_ACCOUNT_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
  optional: PORT (8208), LATENTSYNC_RUNPOD_CONCURRENCY (2),
            LATENTSYNC_RUNPOD_TIMEOUT_MIN (60), LATENTSYNC_R2_PREFIX

Internal LAN only — no auth. The caller owns all content gating (this model
lipsyncs any face video to any audio: likeness + consent checks are the
caller's job, same compliance stance as longcat-api).
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Annotated, Any, Literal

import boto3
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("latentsync-api")

ROOT = Path(os.environ.get("LATENTSYNC_API_ROOT", "/data/latentsync-api"))
OUT_JOBS = ROOT / "jobs"
OUT_UPLOADS = ROOT / "uploads"

JOB_HISTORY = 200

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.environ.get("LATENTSYNC_RUNPOD_ENDPOINT_ID", "")
RUNPOD_CONCURRENCY = int(os.environ.get("LATENTSYNC_RUNPOD_CONCURRENCY", "2"))
RUNPOD_TIMEOUT_MIN = int(os.environ.get("LATENTSYNC_RUNPOD_TIMEOUT_MIN", "60"))
R2_PREFIX = os.environ.get("LATENTSYNC_R2_PREFIX", "renderjobs/latentsync")


class LipsyncRequest(BaseModel):
    video_path: str
    audio_path: str
    inference_steps: int = 20     # 20-50: quality/speed tradeoff
    guidance_scale: float = 1.5   # 1.0-3.0
    seed: int = 1247
    enable_deepcache: bool = True


class JobRecord(BaseModel):
    id: str
    status: Literal["queued", "running", "done", "failed", "cancelled"]
    params: dict[str, Any]
    queued_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    output_path: str | None = None


class State:
    jobs: dict[str, JobRecord] = {}
    order: deque[str] = deque(maxlen=JOB_HISTORY)
    work: queue.Queue[str] = queue.Queue()
    lock = threading.Lock()
    workers: list[threading.Thread] = []


def _r2():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ACCOUNT_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _run_job(job_id: str) -> None:
    rec = State.jobs[job_id]
    if rec.status == "cancelled":
        return
    rec.status = "running"
    rec.started_at = time.time()
    log.info("job %s submitted to runpod", job_id)

    try:
        bucket = os.environ["R2_BUCKET"]
        s3 = _r2()
        prefix = f"{R2_PREFIX}/{job_id}"
        p = rec.params
        video_key = f"{prefix}/in/video{Path(p['video_path']).suffix or '.mp4'}"
        audio_key = f"{prefix}/in/audio{Path(p['audio_path']).suffix or '.wav'}"
        output_key = f"{prefix}/out.mp4"
        s3.upload_file(p["video_path"], bucket, video_key)
        s3.upload_file(p["audio_path"], bucket, audio_key)

        headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
        resp = requests.post(
            f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run",
            json={"input": {
                "video_key": video_key,
                "audio_key": audio_key,
                "output_key": output_key,
                "params": {
                    "inference_steps": p["inference_steps"],
                    "guidance_scale": p["guidance_scale"],
                    "seed": p["seed"],
                    "enable_deepcache": p["enable_deepcache"],
                },
            }},
            headers=headers, timeout=30,
        )
        resp.raise_for_status()
        rid = resp.json()["id"]

        deadline = time.time() + RUNPOD_TIMEOUT_MIN * 60
        while True:
            if time.time() > deadline:
                raise RuntimeError(f"runpod job {rid} timed out after {RUNPOD_TIMEOUT_MIN} min")
            st = requests.get(
                f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{rid}",
                headers=headers, timeout=30,
            ).json()
            status = st.get("status")
            if status == "COMPLETED":
                out = st.get("output") or {}
                if out.get("error"):
                    raise RuntimeError(f"worker error: {out['error']} | {out.get('log_tail', '')[-400:]}")
                break
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise RuntimeError(f"runpod job {rid} {status}: {st.get('error') or st.get('output')}")
            time.sleep(10)

        output_path = str(OUT_JOBS / f"{job_id}.mp4")
        s3.download_file(bucket, output_key, output_path)
        rec.output_path = output_path
        rec.status = "done"
    except Exception as exc:  # noqa: BLE001
        log.exception("job %s failed", job_id)
        rec.error = repr(exc)
        rec.status = "failed"
    finally:
        rec.finished_at = time.time()


def _worker_loop() -> None:
    while True:
        job_id = State.work.get()
        try:
            _run_job(job_id)
        except Exception:  # noqa: BLE001
            log.exception("worker crashed on job %s", job_id)
        finally:
            State.work.task_done()


app = FastAPI(title="latentsync-api")


@app.on_event("startup")
def _startup() -> None:
    OUT_JOBS.mkdir(parents=True, exist_ok=True)
    OUT_UPLOADS.mkdir(parents=True, exist_ok=True)
    missing = [k for k in ("R2_ACCOUNT_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID",
                           "R2_SECRET_ACCESS_KEY") if not os.environ.get(k)]
    if not RUNPOD_API_KEY:
        missing.append("RUNPOD_API_KEY")
    if not RUNPOD_ENDPOINT_ID:
        missing.append("LATENTSYNC_RUNPOD_ENDPOINT_ID")
    if missing:
        raise RuntimeError(f"missing config: {', '.join(missing)}")
    for i in range(max(1, RUNPOD_CONCURRENCY)):
        t = threading.Thread(target=_worker_loop, name=f"latentsync-worker-{i}", daemon=True)
        t.start()
        State.workers.append(t)
    log.info("runpod dispatch started: endpoint=%s workers=%d",
             RUNPOD_ENDPOINT_ID, max(1, RUNPOD_CONCURRENCY))


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "dispatch": "runpod",
        "endpoint_id": RUNPOD_ENDPOINT_ID,
        "queue_depth": State.work.qsize(),
        "worker_alive": any(t.is_alive() for t in State.workers),
    }


@app.post("/v1/uploads")
async def upload(file: Annotated[UploadFile, File()]) -> dict[str, str]:
    suffix = Path(file.filename or "").suffix or ".bin"
    out = OUT_UPLOADS / f"{uuid.uuid4().hex}{suffix}"
    data = await file.read()
    out.write_bytes(data)
    return {"path": str(out)}


@app.post("/v1/jobs/lipsync")
def post_lipsync(req: LipsyncRequest) -> JobRecord:
    if not Path(req.video_path).is_file():
        raise HTTPException(400, f"video path not found: {req.video_path}")
    if not Path(req.audio_path).is_file():
        raise HTTPException(400, f"audio path not found: {req.audio_path}")
    job_id = uuid.uuid4().hex
    rec = JobRecord(id=job_id, status="queued", params=req.model_dump(), queued_at=time.time())
    with State.lock:
        State.jobs[job_id] = rec
        State.order.append(job_id)
        while len(State.jobs) > JOB_HISTORY:
            State.jobs.pop(next(iter(State.jobs)), None)
    State.work.put(job_id)
    return rec


@app.get("/v1/jobs")
def list_jobs(limit: int = 50) -> list[JobRecord]:
    with State.lock:
        ids = list(State.order)[-limit:]
        return [State.jobs[i] for i in ids if i in State.jobs][::-1]


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str) -> JobRecord:
    rec = State.jobs.get(job_id)
    if rec is None:
        raise HTTPException(404, f"job {job_id} not found")
    return rec


@app.get("/v1/jobs/{job_id}/video")
def get_video(job_id: str) -> FileResponse:
    rec = State.jobs.get(job_id)
    if rec is None:
        raise HTTPException(404, f"job {job_id} not found")
    if rec.status != "done" or not rec.output_path:
        raise HTTPException(409, f"job {job_id} status is {rec.status}, no video available")
    if not Path(rec.output_path).is_file():
        raise HTTPException(410, f"output file missing on disk: {rec.output_path}")
    return FileResponse(rec.output_path, media_type="video/mp4", filename=f"{job_id}.mp4")


@app.delete("/v1/jobs/{job_id}")
def cancel_job(job_id: str) -> JobRecord:
    rec = State.jobs.get(job_id)
    if rec is None:
        raise HTTPException(404, f"job {job_id} not found")
    if rec.status == "queued":
        rec.status = "cancelled"
        rec.finished_at = time.time()
    return rec
