# Gemini Media MCP

MCP server for generating images and videos using Google Gemini and VEO models.

## Setup

### Prerequisites

- For video generation (VEO): Google Cloud project with Vertex AI API enabled and a service account with Vertex AI permissions ([setup instructions](#vertex-ai-setup-required-for-veo-video-generation))
- For image generation only: Gemini API key ([setup instructions](#gemini-api-setup-image-generation-only))

### Environment Variables

For Vertex AI (required for VEO video generation):

```bash
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=us-central1
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

**→ See [Vertex AI Setup](#vertex-ai-setup-required-for-veo-video-generation) for detailed instructions**

Alternatively, for Gemini API (image generation only):

```bash
export GEMINI_API_KEY=your-api-key
```

**→ See [Gemini API Setup](#gemini-api-setup-image-generation-only) for detailed instructions**

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
- `model`: Model to use:
  - `gemini-2.5-flash-image` (default): Fast, creative editing
  - `gemini-3-pro-image-preview`: Highest quality, 4K resolution, multi-reference support
  - `imagen-3.0-generate-002`: High quality, text-only input
  - `imagen-4.0-generate-001`: Balanced quality/speed
  - `imagen-4.0-ultra-generate-001`: Highest quality
  - `imagen-4.0-fast-generate-001`: Fastest generation
- `image_uri`: Input image URI for image-to-image generation
- `image_base64`: Base64 encoded input image

**Gemini 3 Pro Image Parameters** (for `gemini-3-pro-image-preview` only):
- `reference_image_uris`: List of up to 14 reference image URIs for multi-image composition
  - Up to 6 object images for high-fidelity inclusion
  - Up to 5 human images for character consistency across scenes
- `image_size`: Output resolution (`1K`, `2K`, `4K`) - must use uppercase K
- `thinking_level`: Reasoning depth (`low` for fast, `high` for complex generation)
- `media_resolution`: Input image processing quality (`MEDIA_RESOLUTION_LOW`, `MEDIA_RESOLUTION_MEDIUM`, `MEDIA_RESOLUTION_HIGH`)
- `thought_signature`: For multi-turn editing workflows - pass back the signature from previous responses

### generate_video

Generate videos using VEO models (requires Vertex AI).

**Parameters:**
- `prompt` (required): Text description of the video
- `model`: Model to use:
  - `veo-2.0-generate-001` (default): Stable, 5-8s duration, no audio
  - `veo-3.1-generate-preview`: Highest quality, 4/6/8s duration, audio support
  - `veo-3.1-fast-generate-preview`: Faster generation with audio support
- `aspect_ratio`: `16:9` (default) or `9:16`
- `duration_seconds`: Video duration (VEO2: 5-8s, VEO3: 4/6/8s)
- `include_audio`: Enable audio generation (VEO3 only)
- `audio_prompt`: Audio description (VEO3 only)
- `negative_prompt`: Things to avoid in the video
- `seed`: Random seed for reproducibility
- `image_uri`: First frame image URI for image-to-video generation

**VEO 3.1 Parameters** (for `veo-3.1-*` models only):
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

### Vertex AI Setup (Required for VEO Video Generation)

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

### Gemini API Setup (Image Generation Only)

For simpler image generation without video capabilities:

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy your key (starts with `AIzaSy...`)
5. Set the environment variable: `export GEMINI_API_KEY=your-api-key`

**Note**: The Gemini API does not support VEO video generation. For video capabilities, you must use Vertex AI.


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

