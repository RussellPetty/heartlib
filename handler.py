import runpod
import torch
import base64
import tempfile
import os
from heartlib import HeartMuLaGenPipeline

# Global model variable - loaded once on cold start
pipe = None

def load_model():
    global pipe
    if pipe is None:
        model_path = os.environ.get("MODEL_PATH", "/app/ckpt")
        version = os.environ.get("MODEL_VERSION", "3B")
        print(f"Loading HeartMuLa model from {model_path}, version {version}...")
        pipe = HeartMuLaGenPipeline.from_pretrained(
            model_path,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            version=version,
        )
        print("Model loaded successfully!")
    return pipe


def handler(job):
    """
    RunPod handler for music generation.

    Input:
        prompt: str - Text describing the music style/tags (required)
        lyrics: str - Song lyrics (optional, defaults to instrumental markers)
        max_audio_length_ms: int - Max audio length in milliseconds (default: 240000)
        temperature: float - Sampling temperature (default: 1.0)
        topk: int - Top-k sampling (default: 50)
        cfg_scale: float - Classifier-free guidance scale (default: 1.5)

    Returns:
        audio_base64: str - Base64 encoded MP3 audio
    """
    job_input = job["input"]

    # Get prompt/tags (required)
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "prompt is required"}

    # Get lyrics (optional - default to instrumental)
    lyrics = job_input.get("lyrics", "[inst]\n[inst]\n[inst]")

    # Get generation parameters
    max_audio_length_ms = job_input.get("max_audio_length_ms", 240_000)
    temperature = job_input.get("temperature", 1.0)
    topk = job_input.get("topk", 50)
    cfg_scale = job_input.get("cfg_scale", 1.5)

    # Load model
    model = load_model()

    # Generate music
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        output_path = f.name

    try:
        with torch.no_grad():
            model(
                {
                    "lyrics": lyrics,
                    "tags": prompt,
                },
                max_audio_length_ms=max_audio_length_ms,
                save_path=output_path,
                topk=topk,
                temperature=temperature,
                cfg_scale=cfg_scale,
            )

        # Read and encode the audio file
        with open(output_path, "rb") as f:
            audio_data = f.read()

        audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        return {
            "audio_base64": audio_base64,
            "format": "mp3",
            "sample_rate": 48000
        }

    finally:
        # Clean up temp file
        if os.path.exists(output_path):
            os.remove(output_path)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
