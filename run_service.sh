#!/bin/bash

# LatentSync Service Runner Script

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check required environment variables
if [ -z "$WEBSERVER_URL" ]; then
    echo "Error: WEBSERVER_URL environment variable not set"
    echo "Example: export WEBSERVER_URL=https://your-api-server.com/"
    exit 1
fi

if [ -z "$API_AUTH_TOKEN" ]; then
    echo "Error: API_AUTH_TOKEN environment variable not set"
    echo "Example: export API_AUTH_TOKEN=your-auth-token"
    exit 1
fi

# Enable MPS fallback for macOS
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run the service
echo "Starting LatentSync service..."
echo "API Endpoint: $WEBSERVER_URL"
echo "Auth Token: [HIDDEN]"
echo "Press Ctrl+C to stop"
echo ""

python latentsync_service.py