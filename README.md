# Gemini Media MCP

Plan, generate, compose, and edit images and video on Google Gemini and Veo 3.1, with a cost estimate before every call.

## What it does

- **Plan:** `plan_generation` ranks tool and model options for an intent with costs, and generates nothing.
- **Generate:** images with Gemini, video with Veo 3.1, fast video with Omni.
- **Compose:** storyboards, multi-beat reels, transitions between stills, bridges between clips.
- **Edit and extend:** conversational video edits, seamless scene extension, loop and extend.
- **Cost-aware:** any call runs `dry_run` for a quote; real runs report metered cost and its pricing source.

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
- For images and video with GCS output and Vertex-only features (e.g. controllable audio: `include_audio` is only honoured on Vertex, where it maps to Veo's `generate_audio` config; on the Gemini API Veo 3.1 always produces audio): a Google Cloud project with the Vertex AI API enabled and a service account with Vertex AI permissions ([setup instructions](#vertex-ai-setup))

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

Other variables the server reads:

| Variable | Description |
|---|---|
| `DATA_FOLDER` | Where generated media is written (default `data`). **Required** when running in a container |
| `VIDEO_GCS_BUCKET` | Default bucket for large video output; also seeds the `GCS_ALLOWED_BUCKETS` allowlist |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Service account key as inline JSON instead of a file path (Vertex mode only). Written by `setup`; the server materialises it into a temp file. `GOOGLE_APPLICATION_CREDENTIALS` also accepts inline JSON |
| `RUNNING_IN_CONTAINER` | Set to `true` by the Docker image. Makes `DATA_FOLDER` mandatory and switches the sse/streamable-http bind address to `0.0.0.0` (`127.0.0.1` otherwise). Presence of `/.dockerenv` has the same effect |
| `FASTMCP_HOST` | Bind address for the sse/streamable-http transports; `--host` wins over it |

### CLI flags

```bash
gemini-media-mcp [--log-level LEVEL] [--host HOST] [--port PORT] [--mount-path PATH] [stdio|sse|streamable-http]
```

- `--log-level`: `DEBUG`, `INFO` (default), `WARNING`, `ERROR`, `CRITICAL`
- `--host` / `--port`: bind address and port for `sse` / `streamable-http` (default port 8000)
- `--mount-path`: mount path for the `sse` transport (e.g. `/custom`). `streamable-http` ignores it and always serves `/mcp`.

`--host`, `--port` and `--mount-path` are also accepted *after* the transport subcommand, which is the only form a `docker run` entrypoint can produce. `--log-level` must come before it.

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

**Every generation tool supports `dry_run: true`** — it returns the cost estimate for the exact call that would run (rerouted model, snapped duration, bridges counted) and generates nothing. Real runs report the metered cost in the response and the sidecar manifest. `generate_clip` is the one to always price first: 3 beats at 8s is $2.40 on the fast tier and $9.60 on standard.


### plan_generation

**Start here when you are not sure which tool or model to use.** Describe what you want in plain language and get back ranked, ready-to-call plans — which tool, which model, which parameters, why that model won, what each option costs, and which models were ruled out and for what reason.

It generates nothing, costs nothing, and is instant: pure rule-based routing over this server's capability tables, not a model call. It never replaces the explicit `generate_*` tools — it tells you how to drive them.

**Parameters:**
- `intent` (required): plain language, e.g. `a 3-beat vertical reel about coffee`, `a poster with the words GRAND OPENING`
- Optional overrides (these always beat what's inferred from the text): `budget` (`cheap`/`balanced`/`best`), `media_kind`, `aspect_ratio`, `image_size`, `duration_seconds`, `num_beats`, `needs_text_rendering`, `needs_4k`, `needs_audio`, `needs_extension`, `num_reference_images`, `wants_gcs_output`, `is_draft`, `pinned_model`

**Returns:** ranked `routes` (tool, model, ready-to-use `params`, score, rationale, caveats, cost), `rejected` models **with reasons**, `conflicts`, a suggested multi-step `workflow`, and `notes`.

It catches requests that cannot work *before* you pay for the failure — 4K on a 1K-only model, extension or first/last-frame on Veo Lite, GCS output on the Gemini API — and reports each as a conflict with a fix.

> Capability beats budget by design. Ask for legible text on a `cheap` budget and it still recommends `gemini-3-pro-image`: a cheap image that fails the brief isn't cheap.

### generate_storyboard

The missing step between an idea and `generate_clip`. Renders one keyframe per shot, then composes them into a **real, readable storyboard** — numbered panels with slug lines, prompts, camera notes and duration badges — instead of a bare list of image URLs.

Two artifacts come back, because MCP clients render inline images but do not execute HTML:
1. **A composited contact-sheet PNG** — written full-resolution to disk (`sheet_url`) and returned inline as a downscaled preview. The preview is the thing you look at in chat; open `sheet_url` when a panel needs a closer read. It is downscaled because the full board runs past a megabyte from about a dozen shots up, and an MCP client drops a result that large outright.
2. **A self-contained HTML page written to disk** (`file://` URL) with full-size frames, complete prompt text and cumulative timecode. Fully offline: images embedded as data URIs, no external requests.

**Parameters:**
- `shots` (required): ordered specs — `{prompt, caption?, duration_seconds?, notes?}`. `caption` is a slug line, `notes` are camera/lighting notes. Capped at **24 shots** per call, because every shot is a billed image
- `title`, `subtitle`: drawn on the board
- `model`, `aspect_ratio`, `image_size`: keyframe generation settings (`9:16` gives vertical panels)
- `theme`: `dark` (default) or `light`
- `dry_run`: price the whole board without generating

**A failed shot does not abort the board** — it renders as a clearly marked panel showing the actual error, so a partial storyboard stays reviewable, and it isn't billed. The `shots` list is designed to be fed straight into `generate_clip` as `beats` once the board reads well.

### generate_image

Generate images using Gemini image models.

**Parameters:**
- `prompt` (required): Text description of the image
- `model` (required — there is no default; name one explicitly): Pick by use case.
  **GA (stable) — preferred in production:**
  - `gemini-3.1-flash-image` (Nano Banana 2) — the general-purpose choice; fast, up to 4K output, up to 14 reference images
  - `gemini-3-pro-image` (Nano Banana Pro) — 4K, reasoning, `thought_signature` for multi-turn editing
  - `gemini-3.1-flash-lite-image` — cheapest, but **1K output only** (2K/4K are unsupported)

  > **Retired IDs are rerouted, not failed.** The models below no longer exist (or are about to). Requesting one still returns an image: the server substitutes the replacement Google published rather than letting the call 404. They are accepted only as compatibility aliases — request a GA model directly.
  >
  > | Retired ID | Gone since | Served by |
  > |---|---|---|
  > | `gemini-3-pro-image-preview` | 2026-06-25 | `gemini-3-pro-image` |
  > | `gemini-3.1-flash-image-preview` | 2026-06-25 | `gemini-3.1-flash-image` |
  > | `imagen-3.0-generate-002` | 2025-11-10 | `gemini-3.1-flash-image` |
  > | `imagen-3.0-capability-001`, `imagen-3.0-capability-002`, `imagen-3.0-fast-generate-001`, `imagen-3.0-generate-001`, `imagen-4.0-fast-generate-001`, `imagen-4.0-generate-001` | 2026-08-17 | `gemini-3.1-flash-image` |
  > | `imagen-4.0-ultra-generate-001` | 2026-08-17 | `gemini-3-pro-image` |
  > | `gemini-2.5-flash-image` | 2026-10-02 (scheduled) | `gemini-3.1-flash-image` |
  >
  > Imagen Ultra is the top Imagen tier, so it reroutes to the top Gemini image model rather than dropping to flash — **`gemini-3-pro-image` is billed at a materially higher rate than `gemini-3.1-flash-image`**. Price the call with `dry_run` before running it.
  >
  > Every substitution is announced on three channels so it cannot go unnoticed: a `warnings` entry in the response JSON, an MCP `warning`-level log notification to the client, and a `WARNING` record in the server log.
  >
  > If you hold Provisioned Throughput on a discontinued Imagen model, move that order yourself — Google does not stop it automatically at retirement.
- `image_uri`: Input image URI for image-to-image generation
- `image_base64`: Base64 encoded input image
- `aspect_ratio`: Output aspect ratio (e.g. `1:1`, `16:9`, `9:16`)
- `person_generation`: Policy for generating people — `dont_allow`, `allow_adult`, or `allow_all` (some regions restrict these values)
- `dry_run`: Return only the cost estimate and the resolved model/parameters — generates nothing, free and instant

Every real run reports `usage` (the token counts the API metered) and `cost` derived from them, and writes both into the sidecar manifest. A dry run prices **the call that would actually be issued**: ask for `imagen-4.0-generate-001` at 4K and it quotes `gemini-3.1-flash-image`; ask `gemini-3.1-flash-lite-image` for 4K and it quotes its 1K default and tells you why.

**Gemini 3.x Image Parameters** (for `gemini-3-pro-image`, `gemini-3.1-flash-image`, `gemini-3.1-flash-lite-image`):
- `reference_image_uris`: List of up to 14 reference image URIs for multi-image composition
  - Up to 6 object images for high-fidelity inclusion
  - Up to 5 human images for character consistency across scenes
- `image_size`: Output resolution (`1K`, `2K`, `4K`) - must use uppercase K.
  `gemini-3.1-flash-lite-image` supports `1K` only; asking it for `2K`/`4K` drops the
  parameter and returns a warning rather than failing the request.
- `media_resolution`: Input image processing quality (`MEDIA_RESOLUTION_LOW`, `MEDIA_RESOLUTION_MEDIUM`, `MEDIA_RESOLUTION_HIGH`)
- `thought_signature_url`: For multi-turn editing workflows — pass back the `thought_signature_url` from a previous response to continue editing the same image. (The parameter is the file URL; passing a `thought_signature` key is silently ignored by MCP.)

### generate_video

Generate videos using VEO models. Video works on **both** credential modes: Veo 3.1 runs on the paid Gemini API tier as well as on Vertex AI. Vertex AI additionally provides GCS output and some Vertex-only features (e.g. `include_audio`, which is only honoured on Vertex). Veo 3.1 Lite is served **exclusively** through the Gemini API.

**Parameters:**
- `prompt` (required): Text description of the video
- `model` (required — there is no default; name one explicitly):
  - `veo-3.1-generate-001`: Highest quality, 4/6/8s duration, audio support
  - `veo-3.1-fast-generate-001`: Faster generation with audio support
  - `veo-3.1-lite-generate-preview`: Most cost-effective, 4/6/8s, audio; text-to-video and image-to-video only (no extension, reference images, first/last-frame, or 4K). Served via the Gemini API only; Vertex AI projects may return 404 for this model.
- `aspect_ratio`: `16:9` (default) or `9:16`
- `resolution`: Output resolution, `720p`, `1080p`, or `4K` (4K not supported on Veo 3.1 Lite)
- `duration_seconds`: Video duration (4/6/8s), **default `8.0`** — the longest and most expensive option. Omitting it on `veo-3.1-generate-001` bills a full 8s render (from $3.20 at 720p/1080p); pass `4` explicitly when a short beat will do.
- `include_audio` (default `false`): Enable audio generation (Vertex only; on the Gemini API Veo 3.1 always produces audio)
- `person_generation`: Policy for generating people — `allow_adult` or `allow_all` (some regions restrict these values)
- `audio_prompt`: Audio description
- `negative_prompt`: Things to avoid in the video
- `seed`: Random seed for reproducibility
- `image_uri`: First frame image URI for image-to-video generation
- `draft` (default `false`): When `true`, routes the request to `gemini-omni-flash-preview` for a fast 720p draft instead of Veo. Iterate fast, then re-run with `draft=false` to finalize on Veo (note: omni is $0.10136/s — marginally above Veo Fast's $0.10/s, so `draft` buys speed, not savings). See [Fast drafts vs. high-fidelity](#fast-drafts-vs-high-fidelity).

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

Fast conversational video generation through Google's Omni models (Interactions API). This is the **fast path** — ideal for drafts and rapid iteration — and the only family with multi-turn conversational editing. See [Fast drafts vs. high-fidelity](#fast-drafts-vs-high-fidelity).

Two models, selected with `omni_model`:

| | `gemini-omni-flash-preview` (default) | `gemini-omni-1.1-flash` |
|---|---|---|
| Output | 720p / 24fps, fixed | 360p, 720p, 1080p, 4K |
| Conversational editing | yes | yes |
| Scene extension | no | yes (`extend_video_omni`) |
| First/last-frame interpolation | no | yes |
| Image references | inferred from position | `<IMAGE_REF_N>`, explicit |
| Video references | no | up to 3 clips × 3s |
| Native audio | not usable | yes (prompt-directed) |
| Seed / negative prompt | no | no |
| Price at 720p | $0.10136/s | $0.10136/s |

Neither is the *cheap* path at 720p: both bill $0.10136/s, a hair above Veo Fast's $0.10/s. What 1.1 adds is a genuinely cheap **draft** tier — `resolution="360p"` is about a third of that, and Google says it renders up to 60% faster.

The default stays on the preview model so existing calls render exactly what they always did. Pass `omni_model="gemini-omni-1.1-flash"` for anything in the right-hand column.

**Parameters:**
- `prompt` (required): Text description of the video. Timecodes work in plain language (`"[0-3s] a person walks"`), as does audio direction (`"include calm background music"`). If you write your own role tags (`<FIRST_FRAME>`, `<IMAGE_REF_0>`, `[# Sources ...]`) they are passed through untouched; otherwise the declarations are generated from the media arguments and echoed back as `effective_prompt`
- `omni_model`: `gemini-omni-flash-preview` (default) or `gemini-omni-1.1-flash`. Arguments the chosen model cannot honor are **refused, never dropped** — a 4K request on the preview model is an error, not a 720p render billed as 4K
- `image_uris`: Image URIs whose role the model infers — one is a starting frame, several are subject references (optional; **at most 8 images total** — more is rejected, since each is buffered in memory)
- `first_frame_uri` / `last_frame_uri` (1.1): Keyframe interpolation. `last_frame_uri` requires `first_frame_uri`; the same URI for both makes a seamless loop
- `reference_image_uris` (1.1): Subject/style references, bound to `<IMAGE_REF_0>`, `<IMAGE_REF_1>`, … in order
- `reference_video_uris` (1.1): Up to 3 clips of up to 3s each, bound to `<VIDEO_REF_0>`, … for character or object likeness. Audio in a reference clip is ignored
- `input_video_uri`: A video to edit (optional). On 1.1 a clip too large to inline is uploaded through the Files API automatically (which needs a Gemini API key — Vertex has no Files API)
- `resolution` (1.1): `360p`, `720p` (default), `1080p` or `4K`. 1080p and 4K are **upscaled** from the base render
- `aspect_ratio`: `16:9` (default) or `9:16` — **not sent when the request is an edit** (an `input_video_uri` or a `previous_interaction_id` makes it one); the API rejects it on an edit task
- `duration_seconds`: Video duration, 3–10 (default 6) — likewise **not sent on an edit**, and the rendered length is then chosen by the service: a measured 3s source edited with `duration_seconds=4` came back at 10.01s. On an edit the response reports `duration_seconds: null` and the quote uses omni's 10s maximum as an upper bound
- `previous_interaction_id`: Continue editing a prior omni result (optional)
- `timeout_seconds`: Overall deadline for create + polling (default 600). A render typically takes over a minute; raise it for long queues

**Notes:**
- No `seed` or `negative_prompt` on either model — put negatives in the prompt (`"no dialogue"`, `"no extra sound effects"`)
- **Large outputs.** A render above 720p, or any extension, exceeds the API's 4 MB inline response limit. On the Gemini API those are requested with `delivery="uri"` and downloaded here automatically (the response says so in a warning); on Vertex AI, pass `output_gcs_uri` to have the service write straight to a bucket you control
- The response includes an `interaction_id` for multi-turn editing and extension (pass it to `edit_video`, `extend_video_omni`, or back into `previous_interaction_id`)
- All output carries SynthID watermarking

**Pricing note for the new resolutions.** Google publishes exactly one Omni output-video rate — 5,792 tokens per second *of 720p video* at $17.50/1M tokens. So: 720p is quoted from the pricing page; 360p is quoted at the one-third ratio stated in the [launch post](https://blog.google/innovation-and-ai/technology/developers-tools/build-with-gemini-omni-1-1-flash/), which is not a pricing page; and 1080p/4K are quoted at the published 720p rate because no separate rate exists for an upscaled render. Every estimate says which of those it is in its `source_note`.

### edit_video

Conversational edit of a previously omni-generated video. Because omni holds the video context server-side (background interactions are retained ~14 days on Vertex AI; longer on the paid Gemini API), you describe **only the change** — no need to re-supply the source video.

Simple instructions edit best: `"Make this video anime"`, `"Make the phone invisible. Keep everything else the same."` Over-describing the scene changes parts you meant to keep.

**Parameters:**
- `previous_interaction_id` (required): The `interaction_id` from a prior `generate_video_omni` response
- `prompt` (required): The edit instruction (e.g. `"make the sky stormy"`)
- `omni_model`: Must be the model that produced `previous_interaction_id` — the interaction's video context lives with it, so an edit cannot switch models mid-conversation
- `resolution`: `gemini-omni-1.1-flash` only; the resolution to render the edit at
- `aspect_ratio`: accepted but **never sent** — the API rejects it on an edit task, so it does not change the output
- `duration_seconds`: accepted but **never sent**, and it does not set the output length. The service picks the rendered length, and it is predictable from neither this value nor the source video: a measured 3s source edited with `duration_seconds=4` rendered 10.01s. The response reports `duration_seconds: null`; a real run bills the length measured from the rendered file, and `dry_run` quotes omni's 10s maximum so a pre-flight never under-states
- `timeout_seconds`: Overall deadline for the edit render (default 600)

### extend_video_omni

Append a seamless continuation to an existing video (`gemini-omni-1.1-flash` only). Extension is neither editing nor a fresh render: the model reads the **last 10 seconds** of the source as context and generates what happens next, keeping motion, characters and audio coherent across the join. Some of the source's final frames are altered so the transition is invisible.

Two sources, with different rules:

- **`previous_interaction_id`** — a clip this server already rendered. The video context lives on the service, nothing is uploaded, and spoken dialogue may be added in the continuation.
- **`input_video_uri`** — a clip of your own. It must be **10 seconds or shorter** (checked locally before anything is uploaded), it cannot gain new dialogue if someone is talking in it, and uploading a video to be extended is unavailable in the EEA, Switzerland and the UK.

Each turn appends up to 10s, to a **cumulative 40s**. `times` chains that many turns, threading each result into the next.

**Parameters:**
- `prompt` (required): How the scene continues — `"Extend this video"`, `"Continue the scene: the camera pans across the mountains"`. Describe the audio if it should change, and say so if you want a cut to a new scene. In a timecode, `0s` is the start of the **new** footage, not of the source
- `previous_interaction_id` **or** `input_video_uri` (exactly one required)
- `times`: How many extension turns to chain (default 1, maximum 4). Turns run on the backend that minted the interaction, so a chain never drifts between Vertex AI and the Gemini API mid-way
- `omni_model`: An interaction cannot change models mid-conversation, so continuing one created on another model is refused rather than sent as a plain edit and billed as an extension
- `resolution`, `reference_image_uris`, `reference_video_uris`, `output_gcs_uri`, `timeout_seconds`, `dry_run` — as for `generate_video_omni`. References ride along on the first turn, which is where a new character is introduced

**Returns:** JSON with the final `video_url`, the `interaction_id` to keep extending from, `final_duration_seconds` (the finished clip), `completed_turns`, one segment record per turn, and the summed cost. If a turn fails part-way through a chain, the turns that already rendered are still returned with their `interaction_id`, alongside an `error` telling you where to resume — they were billed.

**How this is priced.** A turn renders the *whole growing clip*, not just the tail it appends: the docs cap "a total length of 40s", place a cut 2s into the extension of a 10s source "after 12s", and say "some of the final frames in your input video will be edited". Since Omni bills per second of output, turn 2 costs more than turn 1. A `dry_run` measures the source when it can (a prior interaction's sidecar) and projects each turn's output length from it; when it cannot, it assumes the documented 10s maximum source. `turn_output_seconds` shows the projection and `billed_seconds` its total.

### loop_extend

Convenience wrapper that extends a Veo-generated video multiple times in one call. Each Veo extension adds ~7s, and can be chained up to 20 times. This is the **Veo** extension path; `extend_video_omni` is the Omni one, and they are not interchangeable.

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
- `include_audio` (default `true`): Audio per beat (Vertex only)
- `beats` are capped at 20 per call — each is a billed Veo render, and `add_bridges` nearly doubles that. Split longer sequences into several clips
- `add_bridges`: Generate a transition between consecutive beats using the last frame of beat N and the first frame of beat N+1. Requires local (`file://`) beat outputs
- `animatic`: Render every beat with `gemini-omni-flash-preview` (fast 720p) for a **storyboard preview of the whole reel** before committing to full Veo renders. Bridges and Veo-only controls (`seed`, `negative_prompt`) are ignored in this mode
- `output_gcs_uri`: GCS URI for all outputs

**Partial failure is non-fatal.** A failed beat is recorded in the manifest's `errors` list and the run continues; bridges that would have used the failed beat are skipped.

**Returns:** a clip manifest — `{kind, aspect_ratio, segments[], total_duration_seconds, errors[]}`.

**Suggested flow:** run once with `animatic: true` to preview the whole sequence fast, then re-run with `animatic: false` once the beats read well.

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
- `prompt`: Transition motion and style (default: `smooth cinematic cut between the two clips`)
- `model`, `duration_seconds`, `aspect_ratio`, `include_audio`, `audio_prompt`, `negative_prompt`, `seed`, `output_gcs_uri` — as above

**Returns:** JSON with `video_url`, `sidecar_url`, and the source clip URIs.

### Fast drafts vs. high-fidelity

There are two video paths, and you choose based on where you are in the workflow:

- **`gemini-omni-flash-preview` (fastest turnaround)** — 720p, 24fps, conversational multi-turn editing. Great for drafts, storyboards, and iteration. No seeds, no negative prompts, no first/last-frame control. Reached via `generate_video_omni`, `edit_video`, `generate_video(draft=true)`, and `generate_clip(animatic=true)`.
- **`gemini-omni-1.1-flash` (fast, and far more controllable)** — everything above plus 360p/1080p/4K output, first/last-frame interpolation, video references, native audio and scene extension. Still no seeds and no negative prompts. Reached by passing `omni_model="gemini-omni-1.1-flash"` to `generate_video_omni` or `edit_video`, and by `extend_video_omni`.
- **Veo 3.1 / Fast / Lite (high-fidelity)** — up to 1080p/4K, seeds for reproducibility, first/last-frame control, reference images, and extension. The path for final renders. Reached via `generate_video` (default) and `loop_extend`.

Typical workflows:
- **cheap draft → finalize**: run `generate_video_omni(omni_model="gemini-omni-1.1-flash", resolution="360p")` for a preview at roughly a third of the 720p price, then re-run at `720p`/`1080p`/`4K` — or on Veo — once the shot is right.
- **draft → finalize**: run `generate_video(draft=true)` to preview quickly on omni, then re-run the same prompt with `draft=false` to render the final on Veo.
- **animatic → final**: run `generate_clip(animatic=true)` to render each beat via `gemini-omni-flash-preview` as a fast storyboard preview of the whole reel, then re-run with `animatic=false` (the default) to commit to full Veo renders.

> **Note:** `generate_clip`'s new `animatic` parameter (default `false`) renders each beat through `gemini-omni-flash-preview` instead of Veo, so you can preview an entire reel quickly before committing to full Veo renders (price parity with the fast tier; the saving is real only against the standard tier).

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

**Note**: The Gemini API supports Veo 3.1 video generation on the **paid** tier, and Veo 3.1 Lite is available only through the Gemini API. Vertex AI adds GCS output and some Vertex-only features (e.g. controllable audio — `include_audio` only takes effect on Vertex), but is not required for video.


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

