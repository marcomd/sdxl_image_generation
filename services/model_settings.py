from typing import Dict, Any

DEFAULT_NEGATIVE_PROMPT: str = "lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"

MODEL_SETTINGS: Dict[str, Dict[str, Any]] = {
    "sdxl-anime": {
        "model_id": "cagliostrolab/animagine-xl-4.0",
        "num_inference_steps": 28,
        "guidance_scale": 5.0,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT
    },
    "sdxl-turbo": {
        "model_id": "stabilityai/sdxl-turbo",
        "num_inference_steps": 1,
        "guidance_scale": 0.0,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT
    }
}

def get_model_settings(model_type: str) -> Dict[str, Any]:
    """
    Get settings for a specific model type.
    Raises KeyError if model type is not found.
    """
    if model_type not in MODEL_SETTINGS:
        raise KeyError(f"Unknown model type: {model_type}. Available types: {list(MODEL_SETTINGS.keys())}")
    return MODEL_SETTINGS[model_type]