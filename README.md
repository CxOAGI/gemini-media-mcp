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

Generate images using Gemini image models.

**Parameters:**
- `prompt` (required): Text description of the image
- `model`: Pick by use case.
  **GA (stable) — preferred in production:**
  - `gemini-3.1-flash-image` (Nano Banana 2) — **default**; fast, up to 4K output, up to 14 reference images
  - `gemini-3-pro-image` (Nano Banana Pro) — 4K, reasoning, `thought_signature` for multi-turn editing
  - `gemini-3.1-flash-lite-image` — cheapest, but **1K output only** (2K/4K are unsupported)

  > **Retired IDs are rerouted, not failed.** The models below no longer exist (or are about to). Requesting one still returns an image: the server substitutes the replacement Google published rather than letting the call 404. They are accepted only as compatibility aliases — request a GA model directly.
  >
  > | Retired ID | Gone since | Served by |
  > |---|---|---|
  > | `gemini-3-pro-image-preview` | 2026-06-25 | `gemini-3-pro-image` |
  > | `gemini-3.1-flash-image-preview` | 2026-06-25 | `gemini-3.1-flash-image` |
  > | all 8 `imagen-3.0-*` / `imagen-4.0-*` image IDs | 2026-08-17 | `gemini-3.1-flash-image` |
  > | `gemini-2.5-flash-image` | 2026-10-02 (scheduled) | `gemini-3.1-flash-image` |
  >
  > Every substitution is announced on three channels so it cannot go unnoticed: a `warnings` entry in the response JSON, an MCP `warning`-level log notification to the client, and a `WARNING` record in the server log.
  >
  > If you hold Provisioned Throughput on a discontinued Imagen model, move that order yourself — Google does not stop it automatically at retirement.
- `image_uri`: Input image URI for image-to-image generation
- `image_base64`: Base64 encoded input image
- `aspect_ratio`: Output aspect ratio (e.g. `1:1`, `16:9`, `9:16`)
- `person_generation`: Policy for generating people (e.g. `allow_adult`, `dont_allow`)

**Gemini 3.x Image Parameters** (for `gemini-3-pro-image`, `gemini-3.1-flash-image`, `gemini-3.1-flash-lite-image`):
- `reference_image_uris`: List of up to 14 reference image URIs for multi-image composition
  - Up to 6 object images for high-fidelity inclusion
  - Up to 5 human images for character consistency across scenes
- `image_size`: Output resolution (`1K`, `2K`, `4K`) - must use uppercase K.
  `gemini-3.1-flash-lite-image` supports `1K` only; asking it for `2K`/`4K` drops the
  parameter and returns a warning rather than failing the request.
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
- `draft` (default `false`): When `true`, routes the request to `gemini-omni-flash-preview` for a fast 720p draft instead of Veo. Iterate cheaply, then re-run with `draft=false` to finalize on Veo. See [Fast drafts vs. high-fidelity](#fast-drafts-vs-high-fidelity).

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

### generate_video_omni

Fast conversational video generation via Google's `gemini-omni-flash-preview` (Interactions API). This is the **fast/cheap path** — ideal for drafts and rapid iteration. The high-fidelity Veo tools remain the path for final renders (1080p/4K, seeds, first/last frame). See [Fast drafts vs. high-fidelity](#fast-drafts-vs-high-fidelity).

**Parameters:**
- `prompt` (required): Text description of the video
- `image_uris`: List of image URIs to condition on (optional)
- `input_video_uri`: A video to edit (optional)
- `aspect_ratio`: `16:9` (default) or `9:16`
- `duration_seconds`: Video duration, 3–10 (default 6)
- `previous_interaction_id`: Continue editing a prior omni result (optional)

**Notes:**
- 720p only, 24fps
- No `seed` or `negative_prompt` support
- The response includes an `interaction_id` for multi-turn editing (pass it to `edit_video` or back into `previous_interaction_id`)

### edit_video

Conversational edit of a previously omni-generated video. Because omni holds the video context server-side (background interactions are retained ~14 days on Vertex AI; longer on the paid Gemini API), you describe **only the change** — no need to re-supply the source video.

**Parameters:**
- `previous_interaction_id` (required): The `interaction_id` from a prior `generate_video_omni` response
- `prompt` (required): The edit instruction (e.g. `"make the sky stormy"`)
- `aspect_ratio`: `16:9` (default) or `9:16`
- `duration_seconds`: Video duration, 3–10 (default 6)

### loop_extend

Convenience wrapper that extends a Veo-generated video multiple times in one call. Each Veo extension adds ~7s, and can be chained up to 20 times.

**Parameters:**
- `video_uri` (required): The Veo-generated video to extend
- `prompt`: What the video continues with (default: "continue the action")
- `times`: Number of ~7s extensions to apply, 1-20 (default: 1)
- `model`: Veo model to use (default `veo-3.1-generate-001`; Lite is not supported)
- `aspect_ratio`: `16:9` (default) or `9:16` — must match the source video
- `include_audio`: Generate audio on the extended sections (default `true`; Vertex only)
- `output_gcs_uri`: GCS URI for output (optional; required on Vertex)

**Notes:**
- Veo 3.1 / Veo 3.1 Fast only (not Lite)
- 720p

### generate_clip

Generate a **multi-beat short clip** — the building block for a reel or short. Each beat is rendered in order, and the tool returns an ordered manifest a cutting MCP (e.g. vfx-mcp) can splice into a finished clip.

This is the highest-leverage tool in the server: one call produces a whole sequence instead of N round-trips.

**Parameters:**
- `beats` (required): Ordered list of beat specs. Each accepts `{prompt, duration_seconds?, seed?, first_frame_uri?, negative_prompt?, audio_prompt?}`
- `aspect_ratio`: Default `9:16` for vertical social clips
- `model`: VEO model applied to every beat (default `veo-3.1-fast-generate-001`)
- `include_audio`: Audio per beat (Vertex only)
- `add_bridges`: Generate a transition between consecutive beats using the last frame of beat N and the first frame of beat N+1. Requires local (`file://`) beat outputs
- `animatic`: Render every beat with `gemini-omni-flash` (fast, cheap 720p) for a **storyboard preview of the whole reel** before committing to full Veo renders. Bridges and Veo-only controls (`seed`, `negative_prompt`) are ignored in this mode
- `output_gcs_uri`: GCS URI for all outputs

**Partial failure is non-fatal.** A failed beat is recorded in the manifest's `errors` list and the run continues; bridges that would have used the failed beat are skipped.

**Returns:** a clip manifest — `{kind, aspect_ratio, segments[], total_duration_seconds, errors[]}`.

**Suggested flow:** run once with `animatic: true` to preview the whole sequence cheaply, then re-run with `animatic: false` once the beats read well.

### generate_transition

Generate a transition video **between two still frames** using Veo 3.1's first+last-frame mode. Pair with a cutting MCP that extracts the last frame of clip A and the first frame of clip B.

**Parameters:**
- `first_frame_uri` (required): Starting still (`gs://`, `https://`, `file://`)
- `last_frame_uri` (required): Ending still
- `prompt`: Transition motion and style (default: `smooth cinematic transition between the two frames`)
- `model`: Veo model — default `veo-3.1-fast-generate-001`. **Lite does not support first/last-frame mode** and cannot be used
- `duration_seconds`: 4/6/8s, snapped to nearest
- `aspect_ratio`, `include_audio`, `audio_prompt`, `negative_prompt`, `seed`, `output_gcs_uri`

### generate_bridge

Same primitive as `generate_transition`, but takes **two clips instead of two stills**: it decodes the last frame of `from_clip_uri` and the first frame of `to_clip_uri` for you, so no frame extraction step is needed.

**Parameters:**
- `from_clip_uri` (required): Clip whose last frame starts the bridge
- `to_clip_uri` (required): Clip whose first frame ends the bridge
- `prompt`, `model`, `duration_seconds`, `aspect_ratio`, `include_audio`, `audio_prompt`, `negative_prompt`, `seed`, `output_gcs_uri` — as above

**Returns:** JSON with `video_url`, `sidecar_url`, and the source clip URIs.

### Fast drafts vs. high-fidelity

There are two video paths, and you choose based on where you are in the workflow:

- **`gemini-omni-flash-preview` (fast/cheap)** — 720p, 24fps, conversational multi-turn editing. Great for drafts, storyboards, and iteration. No seeds, no negative prompts, no first/last-frame control. Reached via `generate_video_omni`, `edit_video`, `generate_video(draft=true)`, and `generate_clip(animatic=true)`.
- **Veo 3.1 / Fast / Lite (high-fidelity)** — up to 1080p/4K, seeds for reproducibility, first/last-frame control, reference images, and extension. The path for final renders. Reached via `generate_video` (default) and `loop_extend`.

Typical workflows:
- **draft → finalize**: run `generate_video(draft=true)` to preview quickly on omni, then re-run the same prompt with `draft=false` to render the final on Veo.
- **animatic → final**: run `generate_clip(animatic=true)` to render each beat via `gemini-omni-flash-preview` as a fast storyboard preview of the whole reel, then re-run with `animatic=false` (the default) to commit to full Veo renders.

> **Note:** `generate_clip`'s new `animatic` parameter (default `false`) renders each beat through `gemini-omni-flash-preview` instead of Veo, so you can preview an entire reel cheaply before committing to full Veo renders.

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

