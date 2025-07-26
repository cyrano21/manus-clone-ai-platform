from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List
import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import io
import base64
import os
import uuid
import boto3
from botocore.exceptions import ClientError
import logging
import numpy as np
import cv2
import imageio
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Generation Service", version="1.0.0")
security = HTTPBearer()

# Configuration
API_TOKEN = os.getenv("API_TOKEN", "shared-secret-videos")
MODEL_NAME = os.getenv("MODEL_NAME", "stabilityai/stable-video-diffusion-img2vid-xt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# S3/MinIO configuration
S3_ENDPOINT = os.getenv("OBJECT_STORE_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("OBJECT_STORE_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("OBJECT_STORE_SECRET_KEY", "minioadmin")
S3_BUCKET = os.getenv("OBJECT_STORE_BUCKET", "manus-artifacts")

# Initialize S3 client
s3_client = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name='us-east-1'
)

# Global pipeline variable
pipeline = None

class VideoGenerationRequest(BaseModel):
    prompt: Optional[str] = Field(None, description="Text prompt for video generation")
    image_url: Optional[str] = Field(None, description="URL of input image for img2vid")
    duration: float = Field(3.0, ge=1.0, le=10.0, description="Video duration in seconds")
    fps: int = Field(8, ge=4, le=30, description="Frames per second")
    num_inference_steps: int = Field(25, ge=10, le=50, description="Number of denoising steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    width: int = Field(512, ge=256, le=1024, description="Video width")
    height: int = Field(512, ge=256, le=1024, description="Video height")

class VideoGenerationResponse(BaseModel):
    success: bool
    video_url: Optional[str] = None
    artifact_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[dict] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token")
    return credentials.credentials

async def load_model():
    """Load the video diffusion model"""
    global pipeline
    if pipeline is None:
        logger.info(f"Loading model: {MODEL_NAME}")
        try:
            pipeline = StableVideoDiffusionPipeline.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                variant="fp16" if DEVICE == "cuda" else None
            )
            pipeline = pipeline.to(DEVICE)
            
            # Enable memory efficient attention if available
            if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    logger.warning(f"Could not enable xformers: {e}")
            
            # Enable CPU offload for CUDA to save memory
            if DEVICE == "cuda":
                pipeline.enable_model_cpu_offload()
            
            logger.info("Video model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

def load_image_from_url(image_url: str) -> Image.Image:
    """Load image from URL"""
    import requests
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image.convert("RGB")
    except Exception as e:
        logger.error(f"Error loading image from URL: {e}")
        raise HTTPException(status_code=400, detail="Failed to load image from URL")

def frames_to_video(frames: List[np.ndarray], fps: int, output_path: str):
    """Convert frames to video file"""
    try:
        # Use imageio to create video
        with imageio.get_writer(output_path, fps=fps, codec='libx264') as writer:
            for frame in frames:
                writer.append_data(frame)
        logger.info(f"Video saved to {output_path}")
    except Exception as e:
        logger.error(f"Error creating video: {e}")
        raise

def upload_to_s3(video_data: bytes, filename: str) -> str:
    """Upload video to S3/MinIO and return URL"""
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=f"videos/{filename}",
            Body=video_data,
            ContentType="video/mp4"
        )
        
        # Generate URL
        url = f"{S3_ENDPOINT}/{S3_BUCKET}/videos/{filename}"
        return url
    except ClientError as e:
        logger.error(f"Error uploading to S3: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload video")

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    await load_model()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=pipeline is not None,
        device=DEVICE
    )

@app.post("/generate", response_model=VideoGenerationResponse)
async def generate_video(
    request: VideoGenerationRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Generate a video from image or text prompt"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Generating video with prompt: {request.prompt}")
        
        # Load input image if provided
        input_image = None
        if request.image_url:
            input_image = load_image_from_url(request.image_url)
            # Resize image to match video dimensions
            input_image = input_image.resize((request.width, request.height))
        elif request.prompt:
            # For text-to-video, we would need a different model
            # For now, create a simple colored frame as placeholder
            input_image = Image.new('RGB', (request.width, request.height), color='blue')
        else:
            raise HTTPException(status_code=400, detail="Either image_url or prompt must be provided")
        
        # Calculate number of frames
        num_frames = int(request.duration * request.fps)
        
        # Set random seed if provided
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(request.seed)
        
        # Generate video frames
        with torch.inference_mode():
            frames = pipeline(
                input_image,
                num_frames=num_frames,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                generator=generator
            ).frames[0]  # Get first batch
        
        # Convert PIL images to numpy arrays
        frame_arrays = []
        for frame in frames:
            frame_array = np.array(frame)
            frame_arrays.append(frame_array)
        
        # Create temporary video file
        artifact_id = str(uuid.uuid4())
        temp_video_path = f"/tmp/{artifact_id}.mp4"
        
        # Convert frames to video
        frames_to_video(frame_arrays, request.fps, temp_video_path)
        
        # Read video file and upload to S3
        with open(temp_video_path, 'rb') as f:
            video_data = f.read()
        
        filename = f"{artifact_id}.mp4"
        video_url = upload_to_s3(video_data, filename)
        
        # Clean up temporary file
        os.remove(temp_video_path)
        
        logger.info(f"Video generated successfully: {video_url}")
        
        return VideoGenerationResponse(
            success=True,
            video_url=video_url,
            artifact_id=artifact_id,
            metadata={
                "prompt": request.prompt,
                "image_url": request.image_url,
                "duration": request.duration,
                "fps": request.fps,
                "width": request.width,
                "height": request.height,
                "num_frames": num_frames,
                "steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed,
                "model": MODEL_NAME
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        return VideoGenerationResponse(
            success=False,
            error=str(e)
        )

@app.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    """Upload an image to use as input for video generation"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image = image.convert("RGB")
        
        # Generate unique filename
        artifact_id = str(uuid.uuid4())
        filename = f"{artifact_id}.png"
        
        # Convert back to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_data = img_buffer.getvalue()
        
        # Upload to S3
        image_url = upload_to_s3(img_data, f"input-images/{filename}")
        
        return {
            "success": True,
            "image_url": image_url,
            "artifact_id": artifact_id
        }
        
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models(token: str = Depends(verify_token)):
    """List available models"""
    return {
        "current_model": MODEL_NAME,
        "available_models": [
            "stabilityai/stable-video-diffusion-img2vid-xt",
            "stabilityai/stable-video-diffusion-img2vid"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
