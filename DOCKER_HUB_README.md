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
| `model` | string *optional* | Model to use: `GEMINI` (default), `GEMINI3_PRO`, `IMAGEN3`, `IMAGEN4`, `IMAGEN4_ULTRA`, `IMAGEN4_FAST` |
| `image_uri` | string *optional* | Input image URI for image-to-image generation |
| `image_base64` | string *optional* | Base64 encoded input image for image-to-image generation |

**Available Models:**
- `GEMINI` - Gemini's built-in image generation (default)
- `GEMINI3_PRO` - Gemini 3 Pro model
- `IMAGEN3` - Google Imagen 3
- `IMAGEN4` - Google Imagen 4
- `IMAGEN4_ULTRA` - Imagen 4 Ultra (highest quality)
- `IMAGEN4_FAST` - Imagen 4 Fast (faster generation)

*This tool may perform destructive updates.*

*This tool interacts with external entities.*

---

### Tool: **`generate_video`**

Generate videos using VEO models (requires Vertex AI authentication)

| Parameters | Type | Description |
|-----------|------|-------------|
| `prompt` | string | Text description of the video to generate |
| `model` | string *optional* | VEO model to use: `VEO2` (default), `VEO3`, `VEO3_FAST` |
| `aspect_ratio` | string *optional* | Video aspect ratio: `16:9` (default) or `9:16` |
| `duration_seconds` | integer *optional* | Video duration in seconds (VEO2: 5-8s, VEO3: 4/6/8s) |
| `include_audio` | boolean *optional* | Enable audio generation (VEO3 only) |
| `audio_prompt` | string *optional* | Audio description (VEO3 only) |
| `negative_prompt` | string *optional* | Things to avoid in the video |
| `seed` | integer *optional* | Random seed for reproducibility |
| `image_uri` | string *optional* | Input image URI for image-to-video generation |

**Available Models:**
- `VEO2` - VEO 2 model (5-8 seconds)
- `VEO3` - VEO 3 model (4/6/8 seconds with optional audio)
- `VEO3_FAST` - VEO 3 Fast (faster generation)

**Supported Aspect Ratios:**
- `16:9` - Widescreen (default)
- `9:16` - Portrait/vertical

**Duration Options:**
- VEO2: 5-8 seconds
- VEO3: 4, 6, or 8 seconds

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

### Using Gemini API (Images Only)

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

### Gemini API Configuration (Images Only)

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ | Your Gemini API key |
| `DATA_FOLDER` | ✅ | Output directory path (must match host path in volume) |

---

## Google Cloud Setup

For video generation with VEO models, you need Google Cloud Vertex AI access. 

### Quick Setup Steps:

1. **Create a Google Cloud Project** at [console.cloud.google.com](https://console.cloud.google.com)
2. **Enable Vertex AI API** in the API Library
3. **Create a Service Account** with "Vertex AI User" role
4. **Download JSON key file** for the service account
5. **Configure Docker** to mount the key file and set environment variables

For detailed setup instructions, see the [full documentation](https://github.com/CxOAGI/gemini-media-mcp#vertex-ai-setup-required-for-veo-video-generation).

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
