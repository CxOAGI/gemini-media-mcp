# Gemini Media MCP Server

MCP server for generating images and videos using Google Gemini and VEO models.

[What is an MCP Server?](https://www.anthropic.com/news/model-context-protocol)

## MCP Info

| Attribute | Details |
|-----------|---------|
| **Docker Image** | [cxoagi/gemini-media-mcp](https://hub.docker.com/repository/docker/cxoagi/gemini-media-mcp) |
| **Author** | [CxOAGI](https://github.com/CxOAGI) |
| **Repository** | [https://github.com/CxOAGI/gemini-media-mcp](https://github.com/CxOAGI/gemini-media-mcp) |

## Available Tools (10)

| Tools provided by this Server | Short Description |
|-------------------------------|-------------------|
| `plan_generation` | Pick the right tool + model for an intent, with costs and ruled-out options. Generates nothing |
| `generate_image` | Generate images using Gemini image models |
| `generate_storyboard` | Render a keyframe per shot and return a real storyboard (inline contact sheet + HTML) |
| `generate_video` | Generate videos using VEO models (with optional fast `draft` mode) |
| `generate_clip` | Generate a whole multi-beat reel in one call, with optional bridges and a cheap animatic preview |
| `generate_transition` | Veo first+last-frame transition between two stills |
| `generate_bridge` | Transition between two existing clips (frames extracted for you) |
| `generate_video_omni` | Fast conversational video via `gemini-omni-flash-preview` (Interactions API) |
| `edit_video` | Conversationally edit a previously omni-generated video |
| `loop_extend` | Extend a Veo video multiple times in one call |

---

## Tools Details

> **Pricing built in:** every generation tool accepts `dry_run: true` and returns the cost of the exact call that would run, generating nothing. Real runs report metered cost in the response and sidecar manifest.


### Tool: **`plan_generation`**

Decide how to generate something before spending anything. Returns ranked, ready-to-call plans with costs, plus the models it ruled out and why. Pure rule-based routing — no model call, no cost, instant.

| Parameters | Type | Description |
|-----------|------|-------------|
| `intent` | string | Plain-language description of what you want to make |
| `budget` | string *optional* | `cheap`, `balanced`, or `best` |
| `media_kind` / `aspect_ratio` / `image_size` / `duration_seconds` / `num_beats` | *optional* | Force a value instead of inferring it |
| `needs_text_rendering` / `needs_4k` / `needs_audio` / `needs_extension` / `wants_gcs_output` / `is_draft` | boolean *optional* | Hard requirements |
| `num_reference_images` | integer *optional* | How many references you'll supply |
| `pinned_model` | string *optional* | A model you must use; reported as a conflict if it can't satisfy the request |

**Notes:**
- Catches impossible combinations before you pay for the failure (4K on a 1K-only model, extension on Veo Lite, GCS on the Gemini API)
- Every rejected model comes with a reason — nothing is dropped silently

*This tool is read-only.*

---

### Tool: **`generate_storyboard`**

Render one keyframe per shot and compose a real, readable storyboard.

| Parameters | Type | Description |
|-----------|------|-------------|
| `shots` | array | Ordered specs: `{prompt, caption?, duration_seconds?, notes?}` |
| `title` / `subtitle` | string *optional* | Drawn on the board |
| `model` / `aspect_ratio` / `image_size` | string *optional* | Keyframe settings (`9:16` → vertical panels) |
| `theme` | string *optional* | `dark` (default) or `light` |
| `dry_run` | boolean *optional* | Price the whole board without generating |

**Notes:**
- Returns a contact-sheet PNG **inline** plus a self-contained HTML page on disk
- A failed shot renders as a marked error panel and isn't billed — partial boards stay reviewable
- `shots` feeds straight into `generate_clip` as `beats`

*This tool may perform destructive updates.*

*This tool interacts with external entities.*

---

### Tool: **`generate_image`**

Generate images using Gemini image models

| Parameters | Type | Description |
|-----------|------|-------------|
| `prompt` | string | Text description of the image to generate |
| `model` | string *optional* | Model to use (see list below) |
| `image_uri` | string *optional* | Input image URI for image-to-image generation |
| `image_base64` | string *optional* | Base64 encoded input image for image-to-image generation |
| `aspect_ratio` | string *optional* | Output aspect ratio (e.g. `1:1`, `16:9`, `9:16`) |
| `person_generation` | string *optional* | Policy for generating people: `allow_adult` or `allow_all` |
| `dry_run` | boolean *optional* | Return only the cost estimate; generates nothing |

**Available Models (GA):**
- `gemini-3.1-flash-image` - Nano Banana 2; **default**; fast, up to 4K output, up to 14 reference images
- `gemini-3-pro-image` - Nano Banana Pro; 4K, reasoning, multi-turn editing
- `gemini-3.1-flash-lite-image` - cheapest, but **1K output only** (2K/4K unsupported)

### HTTP transports

```bash
docker run --rm -p 8000:8000 -e GEMINI_API_KEY=... -e DATA_FOLDER=/data \
  -v /host/path:/data cxoagi/gemini-media-mcp streamable-http   # or: sse
```

The server binds `0.0.0.0` inside a container so the published port is reachable, and `127.0.0.1` when run directly so a local run is not exposed to the network. Override with `--host` / `--port` or `FASTMCP_HOST`.

> **Retired IDs are rerouted, not failed.** The `gemini-3-pro-image-preview` and `gemini-3.1-flash-image-preview` aliases were retired on 2026-06-25, every `imagen-*` image endpoint is discontinued on 2026-08-17, and `gemini-2.5-flash-image` is scheduled for shutdown on 2026-10-02. Requesting one still returns an image — the server substitutes the GA replacement Google published instead of letting the call 404 — and announces the swap three ways: a `warnings` entry in the response JSON, an MCP `warning` log notification, and a `WARNING` record in the server log. Request a GA model directly.

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
| `dry_run` | boolean *optional* | Return only the cost estimate; generates nothing |
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

### Tool: **`generate_clip`**

Generate a multi-beat short clip — the building block for a reel. One call renders the whole sequence and returns an ordered manifest for a downstream cutting MCP.

| Parameters | Type | Description |
|-----------|------|-------------|
| `beats` | array | Ordered beat specs: `{prompt, duration_seconds?, seed?, first_frame_uri?, negative_prompt?, audio_prompt?}` |
| `aspect_ratio` | string *optional* | `9:16` (default, vertical social) or `16:9` |
| `model` | string *optional* | Veo model for every beat (default `veo-3.1-fast-generate-001`) |
| `include_audio` | boolean *optional* | Audio per beat (Vertex only) |
| `add_bridges` | boolean *optional* | Generate a transition between consecutive beats (requires local beat outputs) |
| `animatic` | boolean *optional* | Render every beat with `gemini-omni-flash` for a fast, cheap 720p storyboard preview of the whole reel |
| `output_gcs_uri` | string *optional* | GCS URI for all outputs |

**Notes:**
- A failed beat is recorded in the manifest's `errors` list; the run continues
- Returns `{kind, aspect_ratio, segments[], total_duration_seconds, errors[]}`
- Typical flow: `animatic: true` to preview cheaply, then re-run for full Veo renders

*This tool may perform destructive updates.*

*This tool interacts with external entities.*

---

### Tool: **`generate_transition`**

Veo 3.1 first+last-frame transition between two **still frames**.

| Parameters | Type | Description |
|-----------|------|-------------|
| `first_frame_uri` | string | Starting still (`gs://`, `https://`, `file://`) |
| `last_frame_uri` | string | Ending still |
| `prompt` | string *optional* | Transition motion and style |
| `model` | string *optional* | Veo model (default fast; **Lite is not supported** — no first/last-frame mode) |
| `duration_seconds` | number *optional* | 4/6/8s, snapped to nearest |
| `aspect_ratio` | string *optional* | Must match the surrounding clips |
| `include_audio` / `audio_prompt` / `negative_prompt` / `seed` / `output_gcs_uri` | *optional* | As for `generate_video` |

*This tool may perform destructive updates.*

*This tool interacts with external entities.*

---

### Tool: **`generate_bridge`**

Same as `generate_transition`, but takes two **clips** — it decodes the last frame of the first and the first frame of the second for you.

| Parameters | Type | Description |
|-----------|------|-------------|
| `from_clip_uri` | string | Clip whose last frame starts the bridge |
| `to_clip_uri` | string | Clip whose first frame ends the bridge |
| `prompt` / `model` / `duration_seconds` / `aspect_ratio` / `include_audio` / `audio_prompt` / `negative_prompt` / `seed` / `output_gcs_uri` | *optional* | As above |

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
