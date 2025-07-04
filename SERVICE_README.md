# LatentSync Service

This service polls an API for lip-sync generation tasks, processes them using LatentSync, and uploads the results.

## Architecture

The service is split into two main components:

1. **`api_client.py`** - Handles all API communication
   - Polling for new tasks
   - Downloading input files
   - Uploading results
   - Status updates

2. **`latentsync_service.py`** - Main service logic
   - Initializes the LatentSync pipeline
   - Processes tasks
   - Manages the main polling loop

## Setup

1. Install LatentSync dependencies (see main README)

2. Set environment variables:
```bash
export WEBSERVER_URL=https://your-api-server.com/
export API_AUTH_TOKEN=your-auth-token
```

3. Ensure models are downloaded:
```bash
huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints
huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints
```

## Running the Service

### Interactive Mode
```bash
./run_service.sh
```

### As a Background Service (systemd)
1. Edit `latentsync.service` with your paths and credentials
2. Install the service:
```bash
sudo cp latentsync.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable latentsync
sudo systemctl start latentsync
```

3. Check status:
```bash
sudo systemctl status latentsync
sudo journalctl -u latentsync -f  # View logs
```

### Docker
```bash
# Build image
docker build -t latentsync-service .

# Run container
docker run -d \
  -e WEBSERVER_URL=https://your-api-server.com/ \
  -e API_AUTH_TOKEN=your-auth-token \
  --name latentsync \
  latentsync-service
```

## API Endpoints

The service expects the following API endpoints:

### 1. Get Next Task
- **Endpoint**: `GET /api/avatar_next?auth={token}`
- **Response**:
```json
{
  "userid": "user123",
  "id": "request123",
  "videourl": "https://example.com/input.mp4",
  "audiourl": "https://example.com/input.wav",
  "mouth": true,
  "steps": 20
}
```

### 2. Update Status
- **Endpoint**: `POST /api`
- **Body**:
```json
{
  "params": {
    "auth": "token",
    "sys": "avatar",
    "act": "update"
  },
  "values": {
    "id": "request123",
    "videoid": "",
    "outputid": "output_path",
    "status": "processing|completed|failed",
    "errormessage": ""
  }
}
```

### 3. Get Upload URL
- **Endpoint**: `POST /api`
- **Body**:
```json
{
  "params": {
    "auth": "token",
    "sys": "avatar",
    "act": "get_upload_url"
  },
  "values": {
    "id": "request123",
    "userid": "user123"
  }
}
```
- **Response**:
```json
{
  "upload_url": "https://example.com/upload/presigned-url"
}
```

## Configuration

Edit these constants in `latentsync_service.py`:

```python
UNET_CONFIG_PATH = "configs/unet/stage2.yaml"  # Model config
INFERENCE_CKPT_PATH = "checkpoints/latentsync_unet.pt"  # Model checkpoint
DEFAULT_INFERENCE_STEPS = 20  # Denoising steps (5-50)
DEFAULT_GUIDANCE_SCALE = 1.5  # Guidance scale (1.0-3.0)
```

## Monitoring

The service logs to stdout with the following format:
```
2024-01-01 12:00:00,000 - module - LEVEL - message
```

Log levels:
- **INFO**: Normal operations
- **WARNING**: Non-critical issues
- **ERROR**: Failures that don't stop the service
- **DEBUG**: Detailed information (set with `logging.DEBUG`)

## Error Handling

The service handles errors gracefully:
- Failed downloads/uploads are retried
- Processing errors mark the task as failed
- The service continues polling after errors
- Temporary files are automatically cleaned up

## Performance

- Initial pipeline loading: ~30 seconds
- Per-video processing time depends on:
  - Video length
  - Number of inference steps
  - Hardware (GPU/MPS/CPU)
  
Typical processing times:
- GPU (CUDA): 1-2 minutes per video
- MPS (Apple Silicon): 3-5 minutes per video
- CPU: 10-20 minutes per video

## Troubleshooting

1. **Service won't start**
   - Check environment variables are set
   - Verify model files exist in `checkpoints/`
   - Check API endpoint is accessible

2. **Out of memory errors**
   - Use `stage2.yaml` instead of `stage2_512.yaml`
   - Reduce inference steps
   - Ensure no other GPU processes are running

3. **Slow processing**
   - Check if GPU/MPS is being used
   - Reduce inference steps for faster (lower quality) results
   - Consider using DeepCache (enabled by default)

4. **API connection errors**
   - Verify WEBSERVER_URL includes trailing slash
   - Check API_AUTH_TOKEN is valid
   - Test endpoints manually with curl