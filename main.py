import os
from web.server import ImageGenerationServer

def main():
    """
    Main entry point for the SDXL Image Generation Server
    """
    # Read authentication key from environment or use a default
    auth_key = os.getenv("GENERATION_AUTH_KEY", "your_secure_generation_key")
    
    # Optional: Configure model path or use default
    # - stabilityai/stable-diffusion-xl-base-1.0
    # - cagliostrolab/animagine-xl-3.1
    model_id = os.getenv(
        "SDXL_MODEL_ID", 
        "cagliostrolab/animagine-xl-3.1"
    )
    
    # Create and run the server
    server = ImageGenerationServer(
        auth_key=auth_key,
        model_id=model_id
    )
    
    # Start the server
    print(f"Starting SDXL Image Generation Server...")
    server.run()

if __name__ == "__main__":
    main()