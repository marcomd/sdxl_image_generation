import os
import io
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.image_generation_service import SDXLImageGenerationService

class GenerationRequest(BaseModel):
    prompt: str = "A nice girl with purple eyes and blue long hair, V, smile"
    negative_prompt: str = "lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"
    num_inference_steps: int = 28
    guidance_scale: float = 7
    height: int = 1216
    width: int = 832

class ImageGenerationServer:
    def __init__(self, 
                 auth_key: Optional[str] = None,
                 model_id: str = "cagliostrolab/animagine-xl-3.1"):
        """
        Initialize the image generation server.
        
        :param auth_key: Static authentication key for endpoint security
        :param model_id: Hugging Face model ID for SDXL
        """
        self.auth_key = auth_key or os.getenv("GENERATION_AUTH_KEY", "default_secret_key")
        
        # Create image generation service
        self.image_service = SDXLImageGenerationService(model_id)
        
        # Create FastAPI app
        self.app = FastAPI(title="Tales.ninja Image Generation Server")
        self.setup_routes()
    
    def setup_routes(self):
        """Set up API routes for image generation."""
        @self.app.post("/generate")
        async def generate_image(
            request: GenerationRequest, 
            authorization: str = Header(None)
        ):
            # Check authorization
            if authorization != self.auth_key:
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
                
                return StreamingResponse(
                    io.BytesIO(image_bytes), 
                    media_type="image/png"
                )
            
            except Exception as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Image generation failed: {str(e)}"
                )
    
    def run(self, host='127.0.0.1', port=8000):
        """
        Run the server using Uvicorn.
        
        :param host: Host to bind the server
        :param port: Port to run the server
        """
        uvicorn.run(self.app, host=host, port=port)