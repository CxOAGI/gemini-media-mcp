# Gemini Media MCP

MCP server for generating images and videos using Google Gemini and VEO models.

## Quick start

```bash
uvx gemini-media-mcp setup
```

The `setup` wizard walks you through the whole onboarding flow end-to-end:

1. Pick a credential mode: **Gemini API** (images + video, easier to set up) or **Vertex AI** (images + video, adds GCS output and Vertex-only features).
2. Enter your API key, or your Google Cloud project plus a service account JSON (file path or inline paste).
3. Choose where generated media should be written (defaults to `~/gemini-media`).
4. Optionally set a `VIDEO_GCS_BUCKET` for large video output, and auto-populate `GCS_ALLOWED_BUCKETS`.
5. Validate your credentials with a live check (constructs a Google GenAI client and lists models to confirm the key/credentials actually authenticate). Validation failures are non-fatal — you can continue anyway.
6. Print a ready-to-paste Claude Desktop JSON block. On macOS, the wizard can also merge the block directly into `~/Library/Application Support/Claude/claude_desktop_config.json` (existing servers are preserved and the prior file is backed up to `.bak`).

For scripted use, all prompts can be supplied via flags:

```bash
uvx gemini-media-mcp setup --non-interactive --mode=gemini --api-key=AIzaSy...
```

If you prefer to configure everything by hand, the manual steps are below.

## Setup

### Prerequisites

- For images **and** video with the simplest setup: a Gemini API key ([setup instructions](#gemini-api-setup)). Veo 3.1 video works on the **paid** Gemini API tier, and Veo 3.1 Lite is served exclusively through the Gemini API.
- For images and video with GCS output and Vertex-only features (e.g. the `generate_audio` flag): a Google Cloud project with the Vertex AI API enabled and a service account with Vertex AI permissions ([setup instructions](#vertex-ai-setup))

### Environment Variables

For Vertex AI (images + video, GCS output, Vertex-only features):

```bash
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=us-central1
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

**→ See [Vertex AI Setup](#vertex-ai-setup) for detailed instructions**

Alternatively, for the Gemini API (images + video, including Veo 3.1 on the paid tier):

```bash
export GEMINI_API_KEY=your-api-key
```

**→ See [Gemini API Setup](#gemini-api-setup) for detailed instructions**

Optional security hardening:

```bash
# Restrict gs:// fetches and output_gcs_uri to specific buckets.
# If unset and VIDEO_GCS_BUCKET is not set, gs:// fetches log a warning.
export GCS_ALLOWED_BUCKETS=bucket-a,bucket-b
```

Local file:// and bare-path inputs are always restricted to `DATA_FOLDER`.
HTTP(S) fetches reject hosts that resolve to private, loopback, link-local,
or metadata IPs, and downloads are capped at 50 MB.

### Claude Desktop Configuration

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "gemini-media": {
      "command": "uvx",
      "args": ["gemini-media-mcp"],
      "env": {
        "GOOGLE_GENAI_USE_VERTEXAI": "true",
        "GOOGLE_CLOUD_PROJECT": "your-project-id",
        "GOOGLE_CLOUD_LOCATION": "us-central1",
        "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/service-account.json"
      }
    }
  }
}
```

Or using Docker (note: `DATA_FOLDER` must be set to the host path, with matching volume mount):

```json
{
  "mcpServers": {
    "gemini-media": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
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

This writes files to your host path and returns paths like `/Users/yourusername/gemini-output/images/abc.png` that Claude Desktop can open directly. The `DATA_FOLDER` directory will contain `images/` and `videos/` subdirectories.

## Available Tools

### generate_image

Generate images using Gemini or Imagen models.

**Parameters:**
- `prompt` (required): Text description of the image
- `model`: Pick by use case.
  **GA (stable) — preferred in production:**
  - `gemini-3.1-flash-image` (Nano Banana 2) — fast, 4K output, up to 14 reference images
  - `gemini-3-pro-image` (Nano Banana Pro) — 4K, reasoning, `thought_signature` for multi-turn editing
  - `gemini-3.1-flash-lite-image` — cheapest; the recommended migration target from the legacy model
  - `gemini-2.5-flash-image` (Nano Banana) — default; now considered **legacy** by Google (migrate to `gemini-3.1-flash-lite-image`)

  > **Note:** `imagen-3.0-generate-002` was shut down on 2025-11-10 and is no longer available. The Imagen 4.x models (`imagen-4.0-fast-generate-001`, `imagen-4.0-generate-001`, `imagen-4.0-ultra-generate-001`) are **deprecated** with a scheduled shutdown of 2026-08-17; prefer the Gemini image models above.
- `image_uri`: Input image URI for image-to-image generation
- `image_base64`: Base64 encoded input image
- `aspect_ratio`: Output aspect ratio (e.g. `1:1`, `16:9`, `9:16`)
- `person_generation`: Policy for generating people (e.g. `allow_adult`, `dont_allow`)

**Gemini 3.x Image Parameters** (for `gemini-3-pro-image` and `gemini-3.1-flash-image`):
- `reference_image_uris`: List of up to 14 reference image URIs for multi-image composition
  - Up to 6 object images for high-fidelity inclusion
  - Up to 5 human images for character consistency across scenes
- `image_size`: Output resolution (`1K`, `2K`, `4K`) - must use uppercase K
- `thinking_level`: Reasoning depth (`low` for fast, `high` for complex generation)
- `media_resolution`: Input image processing quality (`MEDIA_RESOLUTION_LOW`, `MEDIA_RESOLUTION_MEDIUM`, `MEDIA_RESOLUTION_HIGH`)
- `thought_signature`: For multi-turn editing workflows - pass back the signature from previous responses

### generate_video

Generate videos using VEO models. Video works on **both** credential modes: Veo 3.1 runs on the paid Gemini API tier as well as on Vertex AI. Vertex AI additionally provides GCS output and some Vertex-only features (e.g. the `generate_audio` flag). Veo 3.1 Lite is served **exclusively** through the Gemini API.

**Parameters:**
- `prompt` (required): Text description of the video
- `model`: Model to use:
  - `veo-3.1-generate-001` (default): Highest quality, 4/6/8s duration, audio support
  - `veo-3.1-fast-generate-001`: Faster generation with audio support
  - `veo-3.1-lite-generate-preview`: Most cost-effective, 4/6/8s, audio; text-to-video and image-to-video only (no extension, reference images, first/last-frame, or 4K). Served via the Gemini API only; Vertex AI projects may return 404 for this model.
- `aspect_ratio`: `16:9` (default) or `9:16`
- `resolution`: Output resolution, `720p`, `1080p`, or `4K` (4K not supported on Veo 3.1 Lite)
- `duration_seconds`: Video duration (4/6/8s)
- `include_audio`: Enable audio generation
- `person_generation`: Policy for generating people (e.g. `allow_adult`, `dont_allow`)
- `audio_prompt`: Audio description
- `negative_prompt`: Things to avoid in the video
- `seed`: Random seed for reproducibility
- `image_uri`: First frame image URI for image-to-video generation

**Additional Parameters:**
- `last_frame_uri`: Last frame image URI for first+last frame control
  - When combined with `image_uri`, generates smooth transitions between frames
- `reference_image_uris`: List of up to 3 reference image URIs for subject preservation
  - Preserves the appearance of a person, character, or product in the output video
  - **Note**: Only supports 8-second duration (automatically enforced)
  - Cannot be used together with first/last frame inputs
- `extend_video_uri`: URI of existing VEO-generated video to extend
  - Extends the final second of the video and continues the action
  - Can be chained multiple times for longer videos (up to ~148s total)
  - Note: Cannot be used together with other image inputs

**Generation Modes** (automatically selected based on inputs):
- `text_to_video`: Text-only prompt
- `image_to_video`: First frame image input
- `first_last_frame`: First and last frame control
- `reference_to_video`: Reference images for subject preservation (8s only)
- `extend_video`: Extend existing video

## Google Vertex AI and Gemini Access

### Vertex AI Setup

Vertex AI gives you images and video plus GCS output and Vertex-only features. If you only need images and video without those extras, the [Gemini API](#gemini-api-setup) is simpler to set up.


#### Step 1: Create a Google Cloud Project
1. Go to the [Google Cloud Console](https://console.cloud.google.com)
2. Click the project dropdown at the top of the page
3. Click **"New Project"**
4. Enter a project name and click **"Create"**
5. Note your **Project ID** (you'll need this later)

#### Step 2: Enable Vertex AI API
1. In the Cloud Console, go to **"APIs & Services" > "Library"** (or visit [API Library](https://console.cloud.google.com/apis/library))
2. Search for **"Vertex AI API"**
3. Click on **"Vertex AI API"** in the results
4. Click the **"Enable"** button
5. Wait for the API to be enabled (this may take a minute)

#### Step 3: Create a Service Account
1. Go to **"IAM & Admin" > "Service Accounts"** (or visit [Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts))
2. Click **"Create Service Account"** at the top
3. Enter a name (e.g., "gemini-media-mcp") and description
4. Click **"Create and Continue"**
5. In the "Grant this service account access to project" section:
   - Click the **"Select a role"** dropdown
   - Search for **"Vertex AI User"**
   - Select **"Vertex AI User"** role
   - Click **"Continue"**
6. Click **"Done"** (you can skip the optional "Grant users access" section)

#### Step 4: Download Service Account Key
1. In the Service Accounts list, find the account you just created
2. Click the three dots (⋮) in the **"Actions"** column
3. Select **"Manage keys"**
4. Click **"Add Key" > "Create new key"**
5. Select **"JSON"** as the key type
6. Click **"Create"**
7. The JSON key file will automatically download to your computer
8. **Important**: Move this file to a secure location and note the path (e.g., `~/credentials/gemini-media-service-account.json`)
9. **Security Note**: Never commit this file to version control or share it publicly

#### Step 5: Update Configuration
Use the following values in your configuration:
- `GOOGLE_CLOUD_PROJECT`: Your Project ID from Step 1
- `GOOGLE_CLOUD_LOCATION`: `us-central1` (or your preferred region)
- `GOOGLE_APPLICATION_CREDENTIALS`: Full path to the JSON key file from Step 4

### Gemini API Setup

The simplest way to generate both images and video:

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy your key (starts with `AIzaSy...`)
5. Set the environment variable: `export GEMINI_API_KEY=your-api-key`

**Note**: The Gemini API supports Veo 3.1 video generation on the **paid** tier, and Veo 3.1 Lite is available only through the Gemini API. Vertex AI adds GCS output and some Vertex-only features (e.g. the `generate_audio` flag), but is not required for video.


## Contributing

### Development Setup

```bash
uv sync
```

### Running Tests

```bash
uv run pytest
```

### Code Quality

```bash
# Type checking
uv run basedpyright src/ tests/

# Linting and formatting
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Pre-commit hooks
uv run prek
```

### Building Docker Image

```bash
docker build -t gemini-media-mcp .

# With specific version
docker build --build-arg VERSION=1.0.0 -t gemini-media-mcp:1.0.0 .
```

## License

MIT

