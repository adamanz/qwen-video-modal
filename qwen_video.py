"""
Qwen2.5-VL Video Understanding on Modal

Deploy with: modal deploy qwen_video.py
Test with: modal run qwen_video.py
"""

import os
import time
import warnings
from pathlib import Path
from typing import Optional
from uuid import uuid4

import modal

# Configuration
GPU_TYPE = os.environ.get("GPU_TYPE", "a100-40gb")  # a100 for video processing
GPU_COUNT = int(os.environ.get("GPU_COUNT", 1))
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

SGL_LOG_LEVEL = "error"  # try "debug" or "info" if you have issues

MINUTES = 60  # seconds

# Using Qwen2.5-VL for better video understanding
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
TOKENIZER_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_CHAT_TEMPLATE = "qwen2-vl"

MODEL_VOL_PATH = Path("/models")
MODEL_VOL = modal.Volume.from_name("qwen-video-cache", create_if_missing=True)
volumes = {MODEL_VOL_PATH: MODEL_VOL}


def download_model():
    """Download Qwen2.5-VL model to Modal volume."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_PATH,
        local_dir=str(MODEL_VOL_PATH / MODEL_PATH),
        ignore_patterns=["*.pt", "*.bin"],
    )


# Build the container image
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

vlm_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .entrypoint([])
    .apt_install("libnuma-dev", "ffmpeg")  # ffmpeg for video processing
    .uv_pip_install(
        "transformers>=4.45.0",
        "numpy<2",
        "fastapi[standard]==0.115.4",
        "pydantic==2.9.2",
        "requests==2.32.3",
        "starlette==0.41.2",
        "torch==2.7.1",
        "sglang[all]==0.4.10.post2",
        "sgl-kernel==0.2.8",
        "hf-xet==1.1.5",
        "qwen-vl-utils",  # Qwen VL utilities
        "av",  # Video processing
        "opencv-python-headless",  # Frame extraction
        pre=True,
    )
    .env(
        {
            "HF_HOME": str(MODEL_VOL_PATH),
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .run_function(download_model, volumes=volumes)
    .uv_pip_install("term-image==0.7.1")
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
    """Qwen2.5-VL model for video and image understanding."""

    @modal.enter()
    def start_runtime(self):
        """Starts an SGL runtime to execute inference."""
        import sglang as sgl

        self.runtime = sgl.Runtime(
            model_path=MODEL_PATH,
            tokenizer_path=TOKENIZER_PATH,
            tp_size=GPU_COUNT,
            log_level=SGL_LOG_LEVEL,
        )
        self.runtime.endpoint.chat_template = sgl.lang.chat_template.get_chat_template(
            MODEL_CHAT_TEMPLATE
        )
        sgl.set_default_backend(self.runtime)
        print(f"Model {MODEL_PATH} loaded successfully")

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

    def _extract_video_frames(self, video_path: Path, max_frames: int = 16) -> list:
        """Extract frames from video for analysis."""
        import av

        frames = []
        container = av.open(str(video_path))

        total_frames = container.streams.video[0].frames
        if total_frames == 0:
            # Estimate from duration
            duration = container.streams.video[0].duration
            fps = container.streams.video[0].average_rate
            if duration and fps:
                total_frames = int(duration * fps / container.streams.video[0].time_base)

        # Calculate frame interval
        interval = max(1, total_frames // max_frames)

        for i, frame in enumerate(container.decode(video=0)):
            if i % interval == 0 and len(frames) < max_frames:
                frames.append(frame.to_image())

        container.close()
        return frames

    @modal.fastapi_endpoint(method="POST", docs=True)
    def analyze_image(self, request: dict) -> dict:
        """
        Analyze an image with a question.

        Args:
            image_url: URL of the image to analyze
            question: Question about the image (default: "Describe this image in detail")

        Returns:
            dict with 'answer' and 'processing_time'
        """
        import sglang as sgl

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

            @sgl.function
            def image_qa(s, image_path, question):
                s += sgl.user(sgl.image(str(image_path)) + question)
                s += sgl.assistant(sgl.gen("answer"))

            state = image_qa.run(
                image_path=image_path, question=question, max_new_tokens=max_tokens
            )

            processing_time = round((time.monotonic_ns() - start) / 1e9, 2)
            print(f"Request {request_id} completed in {processing_time}s")

            return {
                "answer": state["answer"],
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
            question: Question about the video (default: "Describe what happens in this video")
            max_frames: Maximum frames to extract (default: 16)

        Returns:
            dict with 'answer', 'frames_analyzed', and 'processing_time'
        """
        import sglang as sgl

        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Processing video request {request_id}")

        video_url = request.get("video_url")
        if not video_url:
            return {"error": "video_url is required"}

        question = request.get("question", "Describe what happens in this video in detail.")
        max_frames = request.get("max_frames", 16)
        max_tokens = request.get("max_tokens", 1024)

        try:
            # Download video
            video_path = self._download_media(video_url)

            # Extract frames
            frames = self._extract_video_frames(video_path, max_frames)
            print(f"Extracted {len(frames)} frames from video")

            # Save frames temporarily
            frame_paths = []
            for i, frame in enumerate(frames):
                frame_path = Path(f"/tmp/{request_id}_frame_{i}.jpg")
                frame.save(frame_path, "JPEG")
                frame_paths.append(frame_path)

            # Build multi-frame prompt
            @sgl.function
            def video_qa(s, frame_paths, question):
                # Add all frames as a sequence
                content = "These are frames from a video in chronological order:\n"
                for i, fp in enumerate(frame_paths):
                    content += sgl.image(str(fp))
                    if i < len(frame_paths) - 1:
                        content += " "
                content += f"\n\n{question}"
                s += sgl.user(content)
                s += sgl.assistant(sgl.gen("answer"))

            state = video_qa.run(
                frame_paths=frame_paths, question=question, max_new_tokens=max_tokens
            )

            # Cleanup
            for fp in frame_paths:
                fp.unlink(missing_ok=True)
            video_path.unlink(missing_ok=True)

            processing_time = round((time.monotonic_ns() - start) / 1e9, 2)
            print(f"Request {request_id} completed in {processing_time}s")

            return {
                "answer": state["answer"],
                "frames_analyzed": len(frames),
                "processing_time": processing_time,
                "request_id": str(request_id),
            }

        except Exception as e:
            return {"error": str(e), "request_id": str(request_id)}

    @modal.exit()
    def shutdown_runtime(self):
        self.runtime.shutdown()


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
        # Test video analysis
        payload = json.dumps({
            "video_url": video_url,
            "question": question or "Describe what happens in this video in detail.",
        })
        endpoint_url = model.analyze_video.get_web_url()
        print(f"\nAnalyzing video: {video_url}")
    else:
        # Test image analysis (default)
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


class Colors:
    """ANSI color codes"""
    GREEN = "\033[0;32m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    END = "\033[0m"
