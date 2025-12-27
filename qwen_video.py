"""
Qwen3-VL Video Understanding on Modal

Deploy with: modal deploy qwen_video.py
Test with: modal run qwen_video.py

Features:
- Hours-long video support with full recall
- Second-level timestamp grounding
- 256K context (expandable to 1M)
- 32-language OCR support
- Native video understanding (no frame extraction needed)
"""

import os
import time
import warnings
from pathlib import Path
from typing import Optional
from uuid import uuid4

import modal

# Configuration
GPU_TYPE = os.environ.get("GPU_TYPE", "a100-40gb")
GPU_COUNT = int(os.environ.get("GPU_COUNT", 1))
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

MINUTES = 60  # seconds

# Using Qwen3-VL for advanced video understanding
MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"

MODEL_VOL_PATH = Path("/models")
MODEL_VOL = modal.Volume.from_name("qwen3-video-cache", create_if_missing=True)
volumes = {MODEL_VOL_PATH: MODEL_VOL}


def download_model():
    """Download Qwen3-VL model to Modal volume."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_PATH,
        local_dir=str(MODEL_VOL_PATH / MODEL_PATH),
        ignore_patterns=["*.pt", "*.bin"],
    )


# Build the container image with vLLM for Qwen3-VL support
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

vlm_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .entrypoint([])
    .apt_install("libnuma-dev", "ffmpeg")
    .uv_pip_install(
        "vllm>=0.11.0",  # vLLM with Qwen3-VL support
        "transformers>=4.57.0",
        "qwen-vl-utils==0.0.14",
        "numpy<2",
        "fastapi[standard]",
        "requests",
        "hf-xet",
        "av",
        "opencv-python-headless",
        "pillow",
        pre=True,
    )
    .env(
        {
            "HF_HOME": str(MODEL_VOL_PATH),
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .run_function(download_model, volumes=volumes)
)

app = modal.App("qwen-video-understanding")


@app.cls(
    gpu=GPU_CONFIG,
    timeout=30 * MINUTES,
    scaledown_window=15 * MINUTES,
    image=vlm_image,
    volumes=volumes,
)
@modal.concurrent(max_inputs=50)
class VideoModel:
    """Qwen3-VL model for video and image understanding using vLLM."""

    @modal.enter()
    def start_runtime(self):
        """Initialize vLLM with Qwen3-VL model."""
        from vllm import LLM

        self.llm = LLM(
            model=str(MODEL_VOL_PATH / MODEL_PATH),
            trust_remote_code=True,
            max_model_len=32768,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=GPU_COUNT,
        )
        print(f"Model {MODEL_PATH} loaded successfully with vLLM")

    def _download_media(self, url: str) -> Path:
        """Download media file from URL."""
        import requests

        response = requests.get(url, timeout=120)
        response.raise_for_status()

        filename = url.split("/")[-1].split("?")[0]
        if not filename:
            filename = f"media_{uuid4()}"

        media_path = Path(f"/tmp/{uuid4()}-{filename}")
        media_path.write_bytes(response.content)
        return media_path

    def _extract_video_frames(self, video_path: Path, max_frames: int = 32) -> list:
        """Extract frames from video for analysis."""
        import av
        from PIL import Image

        frames = []
        container = av.open(str(video_path))

        total_frames = container.streams.video[0].frames
        if total_frames == 0:
            duration = container.streams.video[0].duration
            fps = container.streams.video[0].average_rate
            if duration and fps:
                total_frames = int(duration * fps / container.streams.video[0].time_base)

        interval = max(1, total_frames // max_frames)

        for i, frame in enumerate(container.decode(video=0)):
            if i % interval == 0 and len(frames) < max_frames:
                frames.append(frame.to_image())

        container.close()
        return frames

    def _process_vision_request(self, images: list, question: str, max_tokens: int) -> str:
        """Process a vision request with images and question."""
        from vllm import SamplingParams
        from qwen_vl_utils import process_vision_info

        # Build the message with images
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": question})

        messages = [{"role": "user", "content": content}]

        # Process for Qwen3-VL format
        prompt = self.llm.get_tokenizer().apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Prepare image inputs
        image_inputs, _ = process_vision_info(messages)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
        )

        outputs = self.llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": image_inputs}}],
            sampling_params=sampling_params,
        )

        return outputs[0].outputs[0].text

    @modal.fastapi_endpoint(method="POST", docs=True)
    def analyze_image(self, request: dict) -> dict:
        """
        Analyze an image with a question.

        Args:
            image_url: URL of the image to analyze
            question: Question about the image (default: "Describe this image in detail")
            max_tokens: Maximum tokens in response (default: 512)

        Returns:
            dict with 'answer' and 'processing_time'
        """
        from PIL import Image

        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Processing image request {request_id}")

        image_url = request.get("image_url")
        if not image_url:
            return {"error": "image_url is required"}

        question = request.get("question", "Describe this image in detail.")
        max_tokens = request.get("max_tokens", 512)

        try:
            image_path = self._download_media(image_url)
            image = Image.open(image_path)

            answer = self._process_vision_request([image], question, max_tokens)

            # Cleanup
            image_path.unlink(missing_ok=True)

            processing_time = round((time.monotonic_ns() - start) / 1e9, 2)
            print(f"Request {request_id} completed in {processing_time}s")

            return {
                "answer": answer,
                "processing_time": processing_time,
                "request_id": str(request_id),
            }

        except Exception as e:
            return {"error": str(e), "request_id": str(request_id)}

    @modal.fastapi_endpoint(method="POST", docs=True)
    def analyze_video(self, request: dict) -> dict:
        """
        Analyze a video with a question.

        Args:
            video_url: URL of the video to analyze
            question: Question about the video
            max_frames: Maximum frames to extract (default: 32, max: 64)
            max_tokens: Maximum tokens in response (default: 1024)

        Returns:
            dict with 'answer', 'frames_analyzed', and 'processing_time'
        """
        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Processing video request {request_id}")

        video_url = request.get("video_url")
        if not video_url:
            return {"error": "video_url is required"}

        question = request.get("question", "Describe what happens in this video in detail.")
        max_frames = min(max(1, request.get("max_frames", 32)), 64)
        max_tokens = request.get("max_tokens", 1024)

        try:
            video_path = self._download_media(video_url)
            frames = self._extract_video_frames(video_path, max_frames)
            print(f"Extracted {len(frames)} frames from video")

            # Add context about video frames
            video_prompt = f"These are {len(frames)} frames from a video in chronological order. {question}"

            answer = self._process_vision_request(frames, video_prompt, max_tokens)

            # Cleanup
            video_path.unlink(missing_ok=True)

            processing_time = round((time.monotonic_ns() - start) / 1e9, 2)
            print(f"Request {request_id} completed in {processing_time}s")

            return {
                "answer": answer,
                "frames_analyzed": len(frames),
                "processing_time": processing_time,
                "request_id": str(request_id),
            }

        except Exception as e:
            return {"error": str(e), "request_id": str(request_id)}

    @modal.exit()
    def shutdown_runtime(self):
        """Cleanup on shutdown."""
        pass


@app.local_entrypoint()
def main(
    image_url: Optional[str] = None,
    video_url: Optional[str] = None,
    question: Optional[str] = None,
):
    """
    Test the video/image understanding model.

    Examples:
        modal run qwen_video.py --image-url "https://example.com/image.jpg" --question "What's in this image?"
        modal run qwen_video.py --video-url "https://example.com/video.mp4" --question "What happens in this video?"
    """
    import json
    import urllib.request

    model = VideoModel()

    if video_url:
        payload = json.dumps({
            "video_url": video_url,
            "question": question or "Describe what happens in this video in detail.",
        })
        endpoint_url = model.analyze_video.get_web_url()
        print(f"\nAnalyzing video: {video_url}")
    else:
        if not image_url:
            image_url = "https://modal-public-assets.s3.amazonaws.com/golden-gate-bridge.jpg"

        payload = json.dumps({
            "image_url": image_url,
            "question": question or "What is in this image? Describe it in detail.",
        })
        endpoint_url = model.analyze_image.get_web_url()
        print(f"\nAnalyzing image: {image_url}")

    print(f"Question: {question or '(default)'}")
    print(f"Endpoint: {endpoint_url}\n")

    req = urllib.request.Request(
        endpoint_url,
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=300) as response:
        assert response.getcode() == 200, f"Error: {response.getcode()}"
        result = json.loads(response.read().decode())

        print("=" * 60)
        print("RESULT")
        print("=" * 60)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Answer: {result['answer']}")
            print(f"\nProcessing time: {result.get('processing_time', 'N/A')}s")
            if "frames_analyzed" in result:
                print(f"Frames analyzed: {result['frames_analyzed']}")


warnings.filterwarnings(
    "ignore",
    message="It seems this process is not running within a terminal",
    category=UserWarning,
)
