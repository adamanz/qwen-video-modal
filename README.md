# Qwen3-VL Video Understanding on Modal

Deploy **Qwen3-VL-8B-Instruct** on Modal for serverless video and image analysis.

## Features

- **Hours-long video** support with full recall
- **Timestamp grounding** - second-level precision
- **256K context** (expandable to 1M)
- **32-language OCR** support
- **vLLM backend** - high-performance inference
- **Serverless**: Scales to zero, pay only for usage

## Quick Start

### 1. Install Modal CLI

```bash
pip install modal
modal setup
```

### 2. Deploy

```bash
modal deploy qwen_video.py
```

### 3. Use the API

**Analyze an image:**
```bash
curl -X POST https://YOUR-WORKSPACE--qwen-video-understanding-videomodel-analyze-image.modal.run \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/image.jpg", "question": "What is in this image?"}'
```

**Analyze a video:**
```bash
curl -X POST https://YOUR-WORKSPACE--qwen-video-understanding-videomodel-analyze-video.modal.run \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/video.mp4", "question": "What happens in this video?"}'
```

### 4. Test Locally

```bash
# Test with default image
modal run qwen_video.py

# Test with custom video
modal run qwen_video.py --video-url "https://example.com/video.mp4" --question "Describe this"
```

## API Reference

### POST /analyze_image

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_url` | string | required | URL of image to analyze |
| `question` | string | "Describe this image in detail." | Prompt/question |
| `max_tokens` | int | 512 | Max response tokens |

### POST /analyze_video

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_url` | string | required | URL of video to analyze |
| `question` | string | "Describe what happens..." | Prompt/question |
| `max_frames` | int | 32 | Frames to extract (1-64) |
| `max_tokens` | int | 1024 | Max response tokens |

## Configuration

Edit `qwen_video.py` to customize:

- `GPU_TYPE`: Default `a100-40gb` (options: `a10g`, `l40s`, `a100-80gb`)
- `MODEL_PATH`: Default `Qwen/Qwen3-VL-8B-Instruct`

## Cost

- **A100-40GB**: ~$3.30/hour
- **Serverless**: Only pay while processing requests
- **Cold start**: ~30-60 seconds on first request

## MCP Server

See [qwen-video-mcp-server](https://github.com/adamanz/qwen-video-mcp-server) to use this with Claude Desktop.

## License

MIT
