#!/usr/bin/env python3
"""
LatentSync Service - Polls API for generation tasks, processes videos, and uploads results
"""

import logging
import os
import time
import tempfile
import torch
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from accelerate.utils import set_seed

# Import LatentSync components
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature
from DeepCache import DeepCacheSDHelper

# Import API client
from api_client import APIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# LatentSync configuration
UNET_CONFIG_PATH = "configs/unet/stage2.yaml"
INFERENCE_CKPT_PATH = "checkpoints/latentsync_unet.pt"
DEFAULT_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 1.5


class LatentSyncService:
    def __init__(self, api_client: APIClient = None):
        """
        Initialize the LatentSync service
        
        Args:
            api_client: Optional APIClient instance. If not provided, creates one from environment variables
        """
        self.api_client = api_client or APIClient()
        self.pipeline = None
        self.device = None
        self.dtype = None
        
        # Initialize the pipeline once
        self._init_pipeline()
        
    def _init_pipeline(self):
        """Initialize the LatentSync pipeline"""
        logger.info("Initializing LatentSync pipeline...")
        
        # Load config
        config = OmegaConf.load(UNET_CONFIG_PATH)
        
        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16 if torch.cuda.get_device_capability()[0] > 7 else torch.float32
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float32
            # Enable MPS fallback for unsupported operations
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            
        logger.info(f"Using device: {self.device}, dtype: {self.dtype}")
        
        # Initialize components
        scheduler = DDIMScheduler.from_pretrained("configs")
        
        # Determine whisper model based on config
        if config.model.cross_attention_dim == 768:
            whisper_model_path = "checkpoints/whisper/small.pt"
        elif config.model.cross_attention_dim == 384:
            whisper_model_path = "checkpoints/whisper/tiny.pt"
        else:
            raise NotImplementedError("cross_attention_dim must be 768 or 384")
            
        audio_encoder = Audio2Feature(
            model_path=whisper_model_path,
            device=self.device,
            num_frames=config.data.num_frames,
            audio_feat_length=config.data.audio_feat_length,
        )
        
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=self.dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            INFERENCE_CKPT_PATH,
            device="cpu",
        )
        unet = unet.to(dtype=self.dtype)
        
        self.pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        ).to(self.device)
        
        # Enable DeepCache if not on CPU
        if self.device != "cpu":
            helper = DeepCacheSDHelper(pipe=self.pipeline)
            helper.set_params(cache_interval=3, cache_branch_id=0)
            helper.enable()
            
        self.config = config
        logger.info("Pipeline initialized successfully")
        
    def process_task(self, task):
        """Process a single lip-sync generation task"""
        request_id = task['id']
        user_id = task['userid']
        
        # Create temporary directory for this task
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                logger.info(f"Processing task {request_id} for user {user_id}")
                
                # Download input files
                video_path = os.path.join(temp_dir, "input_video.mp4")
                audio_path = os.path.join(temp_dir, "input_audio.wav")
                output_path = os.path.join(temp_dir, "output_video.mp4")
                
                if not self.api_client.download_file(task['videourl'], video_path):
                    raise Exception("Failed to download video")
                    
                if not self.api_client.download_file(task['audiourl'], audio_path):
                    raise Exception("Failed to download audio")
                    
                # Update status to processing
                self.api_client.generate_update(
                    request_id=request_id,
                    video_id="",
                    output_id="",
                    status="processing",
                    err_msg=""
                )
                
                # Run LatentSync inference
                logger.info(f"Running inference with {task['steps']} steps")
                
                # Set random seed for reproducibility
                seed = torch.randint(0, 2**32, (1,)).item()
                set_seed(seed)
                logger.info(f"Using seed: {seed}")
                
                # Run the pipeline
                end_position_info = self.pipeline(
                    video_path=video_path,
                    audio_path=audio_path,
                    video_out_path=output_path,
                    num_frames=self.config.data.num_frames,
                    num_inference_steps=task['steps'],
                    guidance_scale=DEFAULT_GUIDANCE_SCALE,
                    weight_dtype=self.dtype,
                    width=self.config.data.resolution,
                    height=self.config.data.resolution,
                    mask_image_path=self.config.data.mask_image_path,
                    start_time=task.get('start_time', None),
                )
                
                logger.info(f"Inference completed for task {request_id}")
                
                # Upload result
                upload_url = self.api_client.get_upload_url(request_id, user_id)
                if upload_url and self.api_client.upload_file(output_path, upload_url):
                    # Update status to completed
                    self.api_client.generate_update(
                        request_id=request_id,
                        video_id="",
                        output_id=output_path,
                        status="completed",
                        err_msg="",
                        end_position_info=end_position_info
                    )
                    logger.info(f"Task {request_id} completed successfully")
                else:
                    raise Exception("Failed to upload output video")
                    
            except Exception as e:
                logger.error(f"Error processing task {request_id}: {e}")
                # Update status to failed
                self.api_client.generate_update(
                    request_id=request_id,
                    video_id="",
                    output_id="",
                    status="failed",
                    err_msg=str(e)
                )
                
    def run(self, poll_interval=10):
        """
        Main service loop
        
        Args:
            poll_interval: Seconds to wait between polls when no tasks are available
        """
        logger.info("Starting LatentSync service...")
        logger.info(f"Polling interval: {poll_interval} seconds")
        logger.info(f"API endpoint: {self.api_client.endpoint}")
        
        while True:
            try:
                # Poll for next task
                task = self.api_client.avatar_next()
                
                if task:
                    # Process the task
                    self.process_task(task)
                else:
                    # No tasks available, wait before polling again
                    logger.debug("No tasks available, waiting...")
                    time.sleep(poll_interval)
                    
            except KeyboardInterrupt:
                logger.info("Service stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(poll_interval)
                
        logger.info("LatentSync service stopped")


def main():
    """Main entry point"""
    # Check environment variables
    if not os.environ.get('WEBSERVER_URL'):
        logger.error("WEBSERVER_URL environment variable not set")
        logger.error("Example: export WEBSERVER_URL=https://your-api-server.com/")
        return
        
    if not os.environ.get('API_AUTH_TOKEN'):
        logger.error("API_AUTH_TOKEN environment variable not set")
        logger.error("Example: export API_AUTH_TOKEN=your-auth-token")
        return
        
    # Create and run service
    try:
        service = LatentSyncService()
        service.run(poll_interval=10)  # Poll every 10 seconds
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        return


if __name__ == "__main__":
    main()