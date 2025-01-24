import os
import io
import torch
from typing import Optional
from PIL import Image
from diffusers import StableDiffusionXLPipeline

class SDXLImageGenerationService:
    """
    Dedicated service for generating images using Stable Diffusion XL
    """
    def __init__(self, 
                 model_id: str,
                 device: Optional[str] = None):
        """
        Initialize the SDXL image generation service.
        
        :param model_id: Hugging Face model ID for SDXL
        :param device: Specific device to run the model (cuda/cpu)
        """
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load the SDXL pipeline
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True
        )
        
        # Move pipeline to specified device
        self.pipeline = self.pipeline.to(self.device)
    
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
            # Generate image with specified parameters
            generated_images = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            ).images
            
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