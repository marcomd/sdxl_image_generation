import os
import io
import logging
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError

from services.image_generation_service import SDXLImageGenerationService
from services.model_settings import get_model_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000, 
                        description="Text prompt for image generation")
    model_type: str = Field(
        default="anime",
        description="Type of model to use (e.g. 'anime', 'real', 'turbo')"
    )
    height: int = Field(default=1216, ge=256, le=2048)
    width: int = Field(default=832, ge=256, le=2048)

class ImageGenerationServer:
    def __init__(self, 
                 auth_key: Optional[str] = None):
        """
        Initialize the image generation server.
        
        :param auth_key: Static authentication key for endpoint security
        """
        self.auth_key = auth_key
        
        # Initialize single model cache
        self.current_model_id = None
        self.current_generation_service = None
        
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
    
    def get_model_service(self, 
                          model_id: str, 
                          logger: Optional[logging.Logger] = None
                        ) -> SDXLImageGenerationService:
        """
        Get or create an image generation service for the specified model.
        
        :param model_id: Hugging Face model ID
        :return: Image generation service instance
        """
        
        # Clear the current model from memory if it is different
        if self.current_model_id != model_id:
            if self.current_generation_service is not None:
                self.current_generation_service.clear_memory()
            
            self.current_generation_service = SDXLImageGenerationService(model_id=model_id, logger=logger)
            self.current_model_id = model_id
            
        
        return self.current_generation_service

    def setup_routes(self):
        """Set up API routes for image generation."""
        @self.app.post("/generate")
        async def generate_image(
            request: GenerationRequest, 
            authorization: str = Header(None)
        ):
            # Log incoming request details
            logger.info(f" Received generation request: {request}")
            
            # Check authorization
            if authorization != self.auth_key:
                logger.warning(f" Unauthorized access attempt with key: {authorization}")
                raise HTTPException(status_code=403, detail="Unauthorized")
            
            try:
                # Get model settings
                try:
                    settings = get_model_settings(request.model_type)
                except KeyError as e:
                    raise HTTPException(status_code=400, detail=str(e))

                # Debug
                logger.info(f" Generating image with settings: {settings}")

                # Get the appropriate model service
                model_service = self.get_model_service(model_id=settings["model_id"], 
                                                       logger=logger)

                # Generate image using the service
                image = model_service.generate_image(
                    prompt=request.prompt,
                    negative_prompt=settings["negative_prompt"],
                    num_inference_steps=settings["num_inference_steps"],
                    guidance_scale=settings["guidance_scale"],
                    height=request.height,
                    width=request.width
                )
                
                # Convert image to bytes for streaming
                image_bytes = model_service.image_to_bytes(image)
                
                logger.info(" Image generation successful")
                return StreamingResponse(
                    io.BytesIO(image_bytes), 
                    media_type="image/png"
                )
            
            except Exception as e:
                logger.error(f" Image generation failed: {str(e)}")
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