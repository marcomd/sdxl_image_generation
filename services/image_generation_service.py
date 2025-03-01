import os
import io
import gc
import torch
from typing import Optional
from PIL import Image
from diffusers import StableDiffusionXLPipeline
import logging

class SDXLImageGenerationService:
    """
    Dedicated service for generating images using Stable Diffusion XL
    """
    def __init__(self, 
                 model_id: str,
                 logger: Optional[logging.Logger] = None
                ):
        """
        Initialize the SDXL image generation service.
        
        :param model_id: Hugging Face model ID for SDXL
        """
        # Set MPS memory limits
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.8"
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.5"
        
        self.setup_device(model_id)
        self.logger = logger
        
    
    def setup_device(self, model_id):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            # Metal Performance Shaders (MPS) is available
            try:
                self.device = "mps"
            except:
                self.device = "cpu"
        else:
            self.device = "cpu"

        # Load the SDXL pipeline
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.bfloat16,
            use_safetensors=True
        )
        
        # Move pipeline to specified device
        self.pipeline = self.pipeline.to(self.device)
        
    
    def clear_memory(self):
        if self.device == "mps":
            torch.mps.empty_cache()
            gc.collect()
    
    def debug_memory(self):
        if self.logger is None: return
        
        self.logger.info(f" Memory usage: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        height: int = 512,
        width: int = 512
    ) -> Image.Image:
        """
        Generate an image from a text prompt.
        
        :param prompt: Text description of the image to generate
        :param negative_prompt: Describes what to avoid in the image
        :param num_inference_steps: Number of denoising steps
        :param guidance_scale: How closely to follow the prompt
        :param height: Image height
        :param width: Image width
        :return: Generated PIL Image
        """
        try:
            # Clear memory before generation
            self.clear_memory()
            
            # Debug memory usage
            self.debug_memory()

            # Generate image with specified parameters
            generated_images = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            ).images

            self.debug_memory()
            
            # Clear memory after generation
            self.clear_memory()

            # Return the first generated image
            return generated_images[0]
        
        except Exception as e:
            raise RuntimeError(f"Image generation failed with prompt '{prompt}', num_inference_steps {num_inference_steps}, guidance_scale {guidance_scale}, height {height}, width {width}: {str(e)}")
    

    def image_to_bytes(self, image: Image.Image) -> bytes:
        """
        Convert image to bytes for streaming.
        
        :param image: PIL Image to convert
        :return: Image as bytes
        """
        byte_stream = io.BytesIO()
        image.save(byte_stream, format='JPEG')
        return byte_stream.getvalue()