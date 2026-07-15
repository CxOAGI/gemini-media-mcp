# Gemini Media MCP Server

MCP server for generating images and videos using Google Gemini and VEO models.

[What is an MCP Server?](https://www.anthropic.com/news/model-context-protocol)

## MCP Info

| Attribute | Details |
|-----------|---------|
| **Docker Image** | [cxoagi/gemini-media-mcp](https://hub.docker.com/repository/docker/cxoagi/gemini-media-mcp) |
| **Author** | [CxOAGI](https://github.com/CxOAGI) |
| **Repository** | [https://github.com/CxOAGI/gemini-media-mcp](https://github.com/CxOAGI/gemini-media-mcp) |

## Available Tools (5)

| Tools provided by this Server | Short Description |
|-------------------------------|-------------------|
| `generate_image` | Generate images using Gemini or Imagen models |
| `generate_video` | Generate videos using VEO models (with optional fast `draft` mode) |
| `generate_video_omni` | Fast conversational video via `gemini-omni-flash-preview` (Interactions API) |
| `edit_video` | Conversationally edit a previously omni-generated video |
| `loop_extend` | Extend a Veo video multiple times in one call |

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
| `draft` | boolean *optional* | When `true`, routes to `gemini-omni-flash-preview` for a fast 720p draft instead of Veo (default `false`) |

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

### Tool: **`generate_video_omni`**

Fast conversational video generation via Google's `gemini-omni-flash-preview` (Interactions API). This is the fast/cheap path for drafts and iteration; the Veo tools remain the high-fidelity path (1080p/4K, seeds, first/last frame). See [Fast drafts vs. high-fidelity](#fast-drafts-vs-high-fidelity).

| Parameters | Type | Description |
|-----------|------|-------------|
| `prompt` | string | Text description of the video to generate |
| `image_uris` | array *optional* | List of image URIs to condition on |
| `input_video_uri` | string *optional* | A video to edit |
| `aspect_ratio` | string *optional* | `16:9` (default) or `9:16` |
| `duration_seconds` | integer *optional* | Video duration, 3–10 (default 6) |
| `previous_interaction_id` | string *optional* | Continue editing a prior omni result |

**Notes:**
- 720p only, 24fps
- No `seed` or `negative_prompt` support
- Response includes an `interaction_id` for multi-turn editing

*This tool may perform destructive updates.*

*This tool interacts with external entities.*

---

### Tool: **`edit_video`**

Conversational edit of a previously omni-generated video. Omni holds the video context server-side (retained ~14 days on Vertex AI; longer on the paid Gemini API), so you describe only the change.

| Parameters | Type | Description |
|-----------|------|-------------|
| `previous_interaction_id` | string | The `interaction_id` from a prior `generate_video_omni` response |
| `prompt` | string | The edit instruction (e.g. `make the sky stormy`) |
| `aspect_ratio` | string *optional* | `16:9` (default) or `9:16` |
| `duration_seconds` | integer *optional* | Video duration, 3–10 (default 6) |

*This tool may perform destructive updates.*

*This tool interacts with external entities.*

---

### Tool: **`loop_extend`**

Convenience wrapper that extends a Veo-generated video multiple times in one call. Each Veo extension adds ~7s, up to 20 times.

| Parameters | Type | Description |
|-----------|------|-------------|
| `video_uri` | string | The Veo-generated video to extend |
| `prompt` | string *optional* | What the video continues with (default: "continue the action") |
| `times` | integer *optional* | Number of ~7s extensions to apply, 1-20 (default: 1) |
| `model` | string *optional* | Veo model (default `veo-3.1-generate-001`; Lite not supported) |
| `aspect_ratio` | string *optional* | `16:9` (default) or `9:16` — must match the source video |
| `include_audio` | boolean *optional* | Audio on extended sections (default `true`; Vertex only) |
| `output_gcs_uri` | string *optional* | GCS URI for output (required on Vertex) |

**Notes:**
- Veo 3.1 / Veo 3.1 Fast only (not Lite)
- 720p

*This tool may perform destructive updates.*

*This tool interacts with external entities.*

---

## Fast drafts vs. high-fidelity

Two video paths, chosen by where you are in the workflow:

- **`gemini-omni-flash-preview` (fast/cheap)** — 720p, 24fps, conversational multi-turn editing. Great for drafts, storyboards, and iteration. No seeds, negative prompts, or first/last-frame control. Reached via `generate_video_omni`, `edit_video`, `generate_video` with `draft=true`, and `generate_clip` with `animatic=true`.
- **Veo 3.1 / Fast / Lite (high-fidelity)** — up to 1080p/4K, seeds, first/last-frame control, reference images, extension. The path for final renders. Reached via `generate_video` (default) and `loop_extend`.

Typical workflows:
- **draft → finalize**: run `generate_video` with `draft=true` to preview quickly on omni, then re-run the same prompt with `draft=false` to render the final on Veo.
- **animatic → final**: run `generate_clip` with `animatic=true` (default `false`) to render each beat through `gemini-omni-flash-preview` as a fast storyboard preview of the whole reel, then re-run with `animatic=false` to commit to full Veo renders.

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
