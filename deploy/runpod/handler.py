"""RunPod serverless handler for LatentSync 1.6 (video + audio -> lipsynced video).

Same pattern as our other render workers: inputs from R2, subprocess the
upstream inference script, output to R2. The pod sees only R2 creds.

Checkpoints live on the network volume at
/runpod-volume/weights/latentsync/checkpoints (symlinked into the repo at
image build). Runtime-fetched aux models (insightface / face-alignment /
torch hub) also persist on the volume because HOME is pointed there — the
first job on a fresh VOLUME warms them; every later worker reuses them.

Seed mode (run once after endpoint creation — no throwaway pod needed, the
worker itself has the volume mounted):
  {"seed_weights": true}
  -> downloads whisper/tiny.pt + latentsync_unet.pt from
     ByteDance/LatentSync-1.6 into the volume checkpoints dir.

Job mode:
  {
    "video_key":  "renderjobs/latentsync/<job>/in/video.mp4",
    "audio_key":  "renderjobs/latentsync/<job>/in/audio.wav",
    "output_key": "renderjobs/latentsync/<job>/out.mp4",
    "params": {           # all optional
      "inference_steps": 20,        # 20-50
      "guidance_scale": 1.5,        # 1.0-3.0
      "seed": 1247,
      "enable_deepcache": true
    }
  }
"""

import os
import subprocess
import tempfile
import time
import traceback

import boto3
import runpod

REPO = os.environ.get("LATENTSYNC_DIR", "/opt/LatentSync")
CKPT_DIR = "/runpod-volume/weights/latentsync/checkpoints"
UNET_CONFIG = os.environ.get("LATENTSYNC_UNET_CONFIG", "configs/unet/stage2_512.yaml")
LOG_TAIL_LINES = 60
JOB_TIMEOUT_S = int(os.environ.get("JOB_TIMEOUT_S", str(2 * 3600)))


def _s3():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ACCOUNT_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _seed_weights():
    from huggingface_hub import hf_hub_download  # pinned 0.30.2 in the image

    os.makedirs(CKPT_DIR, exist_ok=True)
    got = {}
    for fname in ("whisper/tiny.pt", "latentsync_unet.pt"):
        path = hf_hub_download(
            repo_id="ByteDance/LatentSync-1.6", filename=fname,
            local_dir=CKPT_DIR,
        )
        got[fname] = os.path.getsize(path)
    return {"status": "ok", "seeded": got, "checkpoints_dir": CKPT_DIR}


def handler(job):
    inp = job["input"]
    if inp.get("seed_weights"):
        try:
            return _seed_weights()
        except Exception:
            return {"error": "seed failed", "log_tail": traceback.format_exc()[-2000:]}

    p = inp.get("params") or {}
    bucket = os.environ["R2_BUCKET"]
    s3 = _s3()
    work = tempfile.mkdtemp(prefix="latentsync_")

    try:
        # preflight the volume checkpoints — fail loudly, never re-download here
        unet = os.path.join(CKPT_DIR, "latentsync_unet.pt")
        if not os.path.isfile(unet):
            return {"error": f"checkpoint missing: {unet} — run a seed_weights job first"}

        video_ext = os.path.splitext(inp["video_key"])[1] or ".mp4"
        video_path = os.path.join(work, "in" + video_ext)
        audio_ext = os.path.splitext(inp["audio_key"])[1] or ".wav"
        audio_path = os.path.join(work, "audio" + audio_ext)
        s3.download_file(bucket, inp["video_key"], video_path)
        s3.download_file(bucket, inp["audio_key"], audio_path)

        out_path = os.path.join(work, "out.mp4")
        cmd = [
            "python3", "-m", "scripts.inference",
            "--unet_config_path", UNET_CONFIG,
            "--inference_ckpt_path", unet,
            "--inference_steps", str(int(p.get("inference_steps", 20))),
            "--guidance_scale", str(float(p.get("guidance_scale", 1.5))),
            "--video_path", video_path,
            "--audio_path", audio_path,
            "--video_out_path", out_path,
        ]
        if "seed" in p:
            cmd += ["--seed", str(int(p["seed"]))]
        if p.get("enable_deepcache", True):
            cmd += ["--enable_deepcache"]

        t0 = time.time()
        proc = subprocess.run(
            cmd, cwd=REPO, capture_output=True, text=True, timeout=JOB_TIMEOUT_S,
        )
        tail = "\n".join(
            (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-LOG_TAIL_LINES:])
        if proc.returncode != 0:
            return {"error": f"inference exited {proc.returncode}", "log_tail": tail}
        if not os.path.isfile(out_path):
            return {"error": "inference succeeded but output mp4 missing", "log_tail": tail}
        wall = round(time.time() - t0, 1)

        s3.upload_file(out_path, bucket, inp["output_key"],
                       ExtraArgs={"ContentType": "video/mp4"})
        return {
            "status": "ok",
            "output_key": inp["output_key"],
            "video_mb": round(os.path.getsize(out_path) / 1024 / 1024, 2),
            "wall_seconds": wall,
            "log_tail": tail[-1000:],
        }
    except Exception:
        return {"error": "worker exception", "log_tail": traceback.format_exc()[-2000:]}


runpod.serverless.start({"handler": handler})
