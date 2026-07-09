# Gemini Media MCP Server

MCP server for generating images and videos using Google Gemini and VEO models.

[What is an MCP Server?](https://www.anthropic.com/news/model-context-protocol)

## MCP Info

| Attribute | Details |
|-----------|---------|
| **Docker Image** | [cxoagi/gemini-media-mcp](https://hub.docker.com/repository/docker/cxoagi/gemini-media-mcp) |
| **Author** | [CxOAGI](https://github.com/CxOAGI) |
| **Repository** | [https://github.com/CxOAGI/gemini-media-mcp](https://github.com/CxOAGI/gemini-media-mcp) |

## Available Tools (2)

| Tools provided by this Server | Short Description |
|-------------------------------|-------------------|
| `generate_image` | Generate images using Gemini or Imagen models |
| `generate_video` | Generate videos using VEO models |

---

## Tools Details

### Tool: **`generate_image`**

Generate images using Gemini or Imagen models

| Parameters | Type | Description |
|-----------|------|-------------|
| `prompt` | string | Text description of the image to generate |
| `model` | string *optional* | Model to use (see list below) |
| `image_uri` | string *optional* | Input image URI for image-to-image generation |
| `image_base64` | string *optional* | Base64 encoded input image for image-to-image generation |
| `aspect_ratio` | string *optional* | Output aspect ratio (e.g. `1:1`, `16:9`, `9:16`) |
| `person_generation` | string *optional* | Policy for generating people: `allow_adult` or `allow_all` |

**Available Models (GA):**
- `gemini-3.1-flash-image` - Nano Banana 2; fast, 4K output, up to 14 reference images
- `gemini-3-pro-image` - Nano Banana Pro; 4K, reasoning, multi-turn editing
- `gemini-3.1-flash-lite-image` - cheapest; recommended migration target
- `gemini-2.5-flash-image` - Nano Banana; now considered legacy (migrate to `gemini-3.1-flash-lite-image`)

> **Note:** `imagen-3.0-generate-002` was shut down on 2025-11-10. The Imagen 4.x models are deprecated with a scheduled shutdown of 2026-08-17; prefer the Gemini image models above.

*This tool may perform destructive updates.*

*This tool interacts with external entities.*

---

### Tool: **`generate_video`**

Generate videos using VEO models. Works on both the Gemini API (Veo 3.1 on the paid tier; Veo 3.1 Lite is Gemini-API-only) and Vertex AI (adds GCS output and Vertex-only features).

| Parameters | Type | Description |
|-----------|------|-------------|
| `prompt` | string | Text description of the video to generate |
| `model` | string *optional* | VEO model to use: `veo-3.1-generate-001` (default), `veo-3.1-fast-generate-001`, `veo-3.1-lite-generate-preview` |
| `aspect_ratio` | string *optional* | Video aspect ratio: `16:9` (default) or `9:16` |
| `resolution` | string *optional* | Output resolution: `720p`, `1080p`, or `4K` (4K not on Lite) |
| `duration_seconds` | integer *optional* | Video duration in seconds (4/6/8s) |
| `include_audio` | boolean *optional* | Enable audio generation |
| `person_generation` | string *optional* | Policy for generating people: `allow_adult` or `allow_all` |
| `audio_prompt` | string *optional* | Audio description |
| `negative_prompt` | string *optional* | Things to avoid in the video |
| `seed` | integer *optional* | Random seed for reproducibility |
| `image_uri` | string *optional* | Input image URI for image-to-video generation |

**Available Models:**
- `veo-3.1-generate-001` - VEO 3.1 (4/6/8 seconds with audio support)
- `veo-3.1-fast-generate-001` - VEO 3.1 Fast (faster generation)
- `veo-3.1-lite-generate-preview` - VEO 3.1 Lite (cheapest; Gemini-API-only; text-to-video and image-to-video only)

**Supported Aspect Ratios:**
- `16:9` - Widescreen (default)
- `9:16` - Portrait/vertical

**Duration Options:**
- 4, 6, or 8 seconds

*This tool may perform destructive updates.*

*This tool interacts with external entities.*

---

## Use this MCP Server

### Using Vertex AI (Images + Videos)

```json
{
  "mcpServers": {
    "gemini-media": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-e", "GOOGLE_GENAI_USE_VERTEXAI=true",
        "-e", "GOOGLE_CLOUD_PROJECT=your-project-id",
        "-e", "GOOGLE_CLOUD_LOCATION=us-central1",
        "-e", "GOOGLE_APPLICATION_CREDENTIALS=/credentials.json",
        "-e", "DATA_FOLDER=/Users/yourusername/gemini-output",
        "-v", "/path/to/service-account.json:/credentials.json:ro",
        "-v", "/Users/yourusername/gemini-output:/Users/yourusername/gemini-output",
        "cxoagi/gemini-media-mcp"
      ]
    }
  }
}
```

### Using Gemini API (Images + Videos)

```json
{
  "mcpServers": {
    "gemini-media": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-e", "GEMINI_API_KEY=your-api-key",
        "-e", "DATA_FOLDER=/Users/yourusername/gemini-output",
        "-v", "/Users/yourusername/gemini-output:/Users/yourusername/gemini-output",
        "cxoagi/gemini-media-mcp"
      ]
    }
  }
}
```

**Important Notes:**
- Replace `/Users/yourusername/gemini-output` with your desired output directory
- The `DATA_FOLDER` environment variable must match the host path in the volume mount
- Generated files are saved to `images/` and `videos/` subdirectories within `DATA_FOLDER`
- Restart Claude Desktop after updating the configuration

[Why is it safer to run MCP Servers with Docker?](https://www.docker.com/blog/the-model-context-protocol-simplifying-building-ai-apps-with-anthropic-claude-desktop-and-docker/)

---

## Environment Variables

### Vertex AI Configuration (Images + Videos)

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_GENAI_USE_VERTEXAI` | ✅ | Set to `true` to enable Vertex AI |
| `GOOGLE_CLOUD_PROJECT` | ✅ | Your Google Cloud project ID |
| `GOOGLE_CLOUD_LOCATION` | ✅ | Region (e.g., `us-central1`) |
| `GOOGLE_APPLICATION_CREDENTIALS` | ✅ | Path to service account JSON key (inside container) |
| `DATA_FOLDER` | ✅ | Output directory path (must match host path in volume) |

### Gemini API Configuration (Images + Videos)

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ | Your Gemini API key |
| `DATA_FOLDER` | ✅ | Output directory path (must match host path in volume) |

---

## Google Cloud Setup

Video generation with VEO models works on the paid Gemini API tier (Veo 3.1 Lite is Gemini-API-only). Vertex AI is optional and adds GCS output plus Vertex-only features. To use Vertex AI:

### Quick Setup Steps:

1. **Create a Google Cloud Project** at [console.cloud.google.com](https://console.cloud.google.com)
2. **Enable Vertex AI API** in the API Library
3. **Create a Service Account** with "Vertex AI User" role
4. **Download JSON key file** for the service account
5. **Configure Docker** to mount the key file and set environment variables

For detailed setup instructions, see the [full documentation](https://github.com/CxOAGI/gemini-media-mcp#vertex-ai-setup).

**Security Note:** Never commit service account keys to version control!

---

## Supported Platforms

This image supports multiple architectures:
- `linux/amd64` - Intel/AMD 64-bit
- `linux/arm64` - ARM 64-bit (Apple Silicon, AWS Graviton, etc.)

---

## Source Code & Issues

- **Source Repository:** https://github.com/CxOAGI/gemini-media-mcp
- **Report Issues:** https://github.com/CxOAGI/gemini-media-mcp/issues
- **Documentation:** https://github.com/CxOAGI/gemini-media-mcp

---

## License

MIT License - See [LICENSE](https://github.com/CxOAGI/gemini-media-mcp/blob/main/LICENSE) for details.
