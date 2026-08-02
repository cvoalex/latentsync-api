# latentsync-api — LatentSync 1.6 lipsync (video + audio → lipsynced video)

ByteDance [LatentSync](https://github.com/bytedance/LatentSync) 1.6 as an
internal job-queue API, following our render-worker pattern — except this service was **born
remote**: the GPU work only ever runs on the RunPod serverless endpoint
(`combo-latentsync`), never on prod GPUs. The prod-side server is a thin
dispatcher on port **8208**.

## Flow

```
POST /v1/uploads (video)  -> {"path": ...}
POST /v1/uploads (audio)  -> {"path": ...}
POST /v1/jobs/lipsync {"video_path": ..., "audio_path": ...} -> JobRecord
GET  /v1/jobs/{id}        -> poll until "done"
GET  /v1/jobs/{id}/video  -> the lipsynced MP4
```

Job params (all optional): `inference_steps` 20–50 (default 20),
`guidance_scale` 1.0–3.0 (default 1.5), `seed` (default 1247 — generation is
seeded, keep it fixed for reproducibility), `enable_deepcache` (default true).

Input: any video with one visible face + any speech audio. The pipeline
resamples to 25 fps / 16 kHz internally; v1.6 runs the 512×512 face crop
(`configs/unet/stage2_512.yaml`). ~18 GB VRAM at inference.

## Layout

| Path | What |
|---|---|
| `api/server.py` | the dispatcher API (uploads, job queue, RunPod submit/poll, R2 in/out) |
| `deploy/latentsync-api.service` | systemd unit (prod, port 8208; secrets via `/data/latentsync-api/.env`, chmod 600) |
| `deploy/runpod/` | serverless worker: handler (+ one-time `seed_weights` mode), Dockerfile, build script |

## The RunPod worker

Image `ghcr.io/cvoalex/runpod-latentsync:vN` — built from upstream pins
(ubuntu:22.04 = system python 3.10; torch **2.5.1+cu121**). ⚠️ cu121 has no
sm_120 kernels → the endpoint pools are **Ada/Ampere only, NO Blackwell**.
Checkpoints (`latentsync_unet.pt` + `whisper/tiny.pt` from
`ByteDance/LatentSync-1.6`) live on the `render-weights-nc1` network volume
at `/runpod-volume/weights/latentsync/checkpoints` — seeded once via a
`{"seed_weights": true}` job (the worker mounts the volume; no throwaway pod).
`HOME` points at the volume too, so aux models that insightface /
face-alignment fetch on first run persist across workers.

Versioned tags only on redeploy (workers cache `:latest` silently).

## Compliance

Same stance as longcat-api: the model does **no** content gating and will
lipsync any face to any audio. The caller owns likeness/consent checks (AB
2602 / ArcFace gate), audio-transcript moderation, and output review. LAN
only, no auth — never expose publicly.
