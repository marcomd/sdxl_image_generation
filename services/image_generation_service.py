import os
import io
import gc
import torch
from typing import Optional
from PIL import Image
from diffusers import StableDiffusionXLPipeline

class SDXLImageGenerationService:
    """
    Dedicated service for generating images using Stable Diffusion XL
    """
    def __init__(self, 
                 model_id: str):
        """
        Initialize the SDXL image generation service.
        
        :param model_id: Hugging Face model ID for SDXL
        """
        # Set MPS memory limits
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.8"
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.5"
        
        self.setup_device(model_id)
        
    
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
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True
        )
        
        # Move pipeline to specified device
        self.pipeline = self.pipeline.to(self.device)
        
    
    def clear_memory(self):
        if self.device == "mps":
            torch.mps.empty_cache()
            gc.collect()
    
    
    def generate_image(
        self, 
        prompt: str, 
        negative_prompt: str = "lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
        num_inference_steps: int = 0,
        guidance_scale: float = 0,
        height: int = 0,
        width: int = 0
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
            
            # Generate image with specified parameters
            generated_images = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            ).images
            
            # Clear memory after generation
            self.clear_memory()

            # Return the first generated image
            return generated_images[0]
        
        except Exception as e:
            raise RuntimeError(f"Image generation failed: {str(e)}")

    
    def save_image(
        self, 
        image: Image.Image, 
        output_path: Optional[str] = None
    ) -> str:
        """
        Save the generated image.
        
        :param image: PIL Image to save
        :param output_path: Custom output path (optional)
        :return: Path where image was saved
        """
        # If no path provided, generate a unique filename
        if output_path is None:
            os.makedirs('generated_images', exist_ok=True)
            output_path = os.path.join(
                'generated_images', 
                f'generated_image_{len(os.listdir("generated_images")) + 1}.jpg'
            )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save image
        image.save(output_path)
        return output_path
    

    def image_to_bytes(self, image: Image.Image) -> bytes:
        """
        Convert image to bytes for streaming.
        
        :param image: PIL Image to convert
        :return: Image as bytes
        """
        byte_stream = io.BytesIO()
        image.save(byte_stream, format='JPEG')
        return byte_stream.getvalue()