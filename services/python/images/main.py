from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List
import torch
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from PIL import Image
import io
import base64
import os
import uuid
import boto3
from botocore.exceptions import ClientError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Generation Service", version="1.0.0")
security = HTTPBearer()

# Configuration
API_TOKEN = os.getenv("API_TOKEN", "shared-secret-images")
MODEL_NAME = os.getenv("MODEL_NAME", "stabilityai/stable-diffusion-xl-base-1.0")
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

class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt to avoid certain elements")
    width: int = Field(1024, ge=512, le=2048, description="Image width")
    height: int = Field(1024, ge=512, le=2048, description="Image height")
    num_inference_steps: int = Field(20, ge=10, le=100, description="Number of denoising steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale for prompt adherence")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

class ImageGenerationResponse(BaseModel):
    success: bool
    image_url: Optional[str] = None
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
    """Load the diffusion model"""
    global pipeline
    if pipeline is None:
        logger.info(f"Loading model: {MODEL_NAME}")
        try:
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                use_safetensors=True,
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
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

def upload_to_s3(image_data: bytes, filename: str) -> str:
    """Upload image to S3/MinIO and return URL"""
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=f"images/{filename}",
            Body=image_data,
            ContentType="image/png"
        )
        
        # Generate URL
        url = f"{S3_ENDPOINT}/{S3_BUCKET}/images/{filename}"
        return url
    except ClientError as e:
        logger.error(f"Error uploading to S3: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload image")

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

@app.post("/generate", response_model=ImageGenerationResponse)
async def generate_image(
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Generate an image from text prompt"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Generating image with prompt: {request.prompt[:100]}...")
        
        # Set random seed if provided
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(request.seed)
        
        # Generate image
        with torch.inference_mode():
            result = pipeline(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                generator=generator
            )
        
        image = result.images[0]
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_data = img_buffer.getvalue()
        
        # Generate unique filename
        artifact_id = str(uuid.uuid4())
        filename = f"{artifact_id}.png"
        
        # Upload to S3
        image_url = upload_to_s3(img_data, filename)
        
        logger.info(f"Image generated successfully: {image_url}")
        
        return ImageGenerationResponse(
            success=True,
            image_url=image_url,
            artifact_id=artifact_id,
            metadata={
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "width": request.width,
                "height": request.height,
                "steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed,
                "model": MODEL_NAME
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return ImageGenerationResponse(
            success=False,
            error=str(e)
        )

@app.post("/upscale")
async def upscale_image(
    token: str = Depends(verify_token)
):
    """Upscale an existing image (placeholder for future implementation)"""
    raise HTTPException(status_code=501, detail="Upscaling not implemented yet")

@app.get("/models")
async def list_models(token: str = Depends(verify_token)):
    """List available models"""
    return {
        "current_model": MODEL_NAME,
        "available_models": [
            "stabilityai/stable-diffusion-xl-base-1.0",
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
