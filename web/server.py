import os
import io
import logging
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError

from services.image_generation_service import SDXLImageGenerationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000, 
                        description="Text prompt for image generation")
    model_id: str = "cagliostrolab/animagine-xl-3.1",
    negative_prompt: str = Field(default="lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]", max_length=1000)
    num_inference_steps: int = Field(default=28, ge=1, le=200)
    guidance_scale: float = Field(default=7, ge=1.0, le=20.0)
    height: int = Field(default=1216, ge=256, le=2048)
    width: int = Field(default=832, ge=256, le=2048)

class ImageGenerationServer:
    def __init__(self, 
                 auth_key: Optional[str] = None,
                 model_id: str = ""):
        """
        Initialize the image generation server.
        
        :param auth_key: Static authentication key for endpoint security
        :param model_id: Hugging Face model ID for SDXL
        """
        self.auth_key = auth_key or os.getenv("GENERATION_AUTH_KEY", "default_secret_key")
        
        # Create image generation service
        self.image_service = SDXLImageGenerationService(model_id)
        
        # Create FastAPI app
        self.app = FastAPI(title="SDXL Image Generation Server")
        
        # Add global exception handler
        @self.app.exception_handler(ValidationError)
        async def validation_exception_handler(request: Request, exc: ValidationError):
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Validation Error",
                    "details": exc.errors()
                }
            )
        
        self.setup_routes()
    
    def setup_routes(self):
        """Set up API routes for image generation."""
        @self.app.post("/generate")
        async def generate_image(
            request: GenerationRequest, 
            authorization: str = Header(None)
        ):
            # Log incoming request details
            logger.info(f"Received generation request: {request}")
            
            # Check authorization
            if authorization != self.auth_key:
                logger.warning(f"Unauthorized access attempt with key: {authorization}")
                raise HTTPException(status_code=403, detail="Unauthorized")
            
            try:
                # Generate image using the service
                image = self.image_service.generate_image(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    height=request.height,
                    width=request.width
                )
                
                # Convert image to bytes for streaming
                image_bytes = self.image_service.image_to_bytes(image)
                
                logger.info("Image generation successful")
                return StreamingResponse(
                    io.BytesIO(image_bytes), 
                    media_type="image/png"
                )
            
            except Exception as e:
                logger.error(f"Image generation failed: {str(e)}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Image generation failed: {str(e)}"
                )
    
    def run(self, host='0.0.0.0', port=8000):
        """
        Run the server using Uvicorn.
        
        :param host: Host to bind the server
        :param port: Port to run the server
        """
        uvicorn.run(self.app, host=host, port=port)