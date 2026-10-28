# Gemini Media MCP Server

Plan, generate, compose, and edit images and video with Gemini image models, Veo 3.1, and Gemini Omni, with a cost estimate before every call and an optional Gemini 4 Argon director that checks each take against your notes.

[What is an MCP Server?](https://www.anthropic.com/news/model-context-protocol)

## MCP Info

| Attribute | Details |
|-----------|---------|
| **Docker Image** | [cxoagi/gemini-media-mcp](https://hub.docker.com/repository/docker/cxoagi/gemini-media-mcp) |
| **Author** | [CxOAGI](https://github.com/CxOAGI) |
| **Repository** | [https://github.com/CxOAGI/gemini-media-mcp](https://github.com/CxOAGI/gemini-media-mcp) |

## Available Tools (11)

| Tools provided by this Server | Short Description |
|-------------------------------|-------------------|
| `plan_generation` | Pick the right tool + model for an intent, with costs and ruled-out options. Generates nothing |
| `generate_image` | Generate images using Gemini image models |
| `generate_storyboard` | Render a keyframe per shot and return a real storyboard (inline contact-sheet preview + full-size PNG and HTML on disk) |
| `generate_video` | Generate videos using VEO models (with optional fast `draft` mode) |
| `generate_clip` | Generate a whole multi-beat reel in one call, with optional bridges and a fast animatic preview |
| `generate_transition` | Veo first+last-frame transition between two stills |
| `generate_bridge` | Transition between two existing clips (frames extracted for you) |
| `generate_video_omni` | Fast conversational video via `gemini-omni-1.1-flash` (Interactions API) |
| `edit_video` | Conversationally edit a previously omni-generated video |
| `extend_video_omni` | Append a seamless continuation to an existing video (`gemini-omni-1.1-flash`) |
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
- Catches impossible combinations before you pay for the failure (4K on a 1K-only model, Veo on the Gemini API, GCS on the Gemini API)
- Every rejected model comes with a reason — nothing is dropped silently

*This tool is read-only.*

---

### Tool: **`generate_storyboard`**

Render one keyframe per shot and compose a real, readable storyboard.

| Parameters | Type | Description |
|-----------|------|-------------|
| `shots` | array | Ordered specs: `{prompt, caption?, duration_seconds?, notes?}`. Capped at 24 shots per call — every shot is a billed image |
| `title` / `subtitle` | string *optional* | Drawn on the board |
| `model` / `aspect_ratio` / `image_size` | string *optional* | Keyframe settings (`9:16` → vertical panels) |
| `theme` | string *optional* | `dark` (default) or `light` |
| `dry_run` | boolean *optional* | Price the whole board without generating |

**Notes:**
- Writes a full-resolution contact-sheet PNG and a self-contained HTML page to disk, and returns a downscaled **inline** preview of the sheet — the full board runs past the size an MCP client will accept from about a dozen shots up, so open `sheet_url` for a closer read
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
| `model` | string **required** | Model to use — no default; name one explicitly (see list below) |
| `image_uri` | string *optional* | Input image URI for image-to-image generation |
| `image_base64` | string *optional* | Base64 encoded input image for image-to-image generation |
| `aspect_ratio` | string *optional* | Output aspect ratio (e.g. `1:1`, `16:9`, `9:16`) |
| `person_generation` | string *optional* | Policy for generating people: `dont_allow`, `allow_adult`, or `allow_all` |
| `dry_run` | boolean *optional* | Return only the cost estimate; generates nothing |
| `director_notes` | string *optional* | What the image must get right, in plain words. Gemini 4 Argon checks each take against them and retakes on a clear fail; see [Virtual director](#virtual-director) |
| `max_retakes` | integer *optional* | Retakes the director may spend when a check fails, 0 to 2 (default 1) |

**Available Models (GA):**
- `gemini-3.1-flash-image` - Nano Banana 2; the general-purpose choice; fast, up to 4K output, up to 14 reference images
- `gemini-3-pro-image` - Nano Banana Pro; 4K, reasoning, multi-turn editing
- `gemini-3.1-flash-lite-image` - cheapest, but **1K output only** (2K/4K unsupported)

### HTTP transports

```bash
docker run --rm -p 8000:8000 -e GEMINI_API_KEY=... -e DATA_FOLDER=/data \
  -v /host/path:/data cxoagi/gemini-media-mcp streamable-http   # or: sse
```

The server binds `0.0.0.0` inside a container so the published port is reachable, and `127.0.0.1` when run directly so a local run is not exposed to the network (the image sets `RUNNING_IN_CONTAINER=true` to signal which case applies). Override with `--host` / `--port` or `FASTMCP_HOST`. `--mount-path` sets the mount path for the `sse` transport only (e.g. `/custom`); `streamable-http` ignores it and always serves `/mcp`. `--log-level` takes `DEBUG`, `INFO` (default), `WARNING`, `ERROR` or `CRITICAL`. `--host`, `--port` and `--mount-path` are accepted after the transport subcommand, which is the form `docker run` produces; `--log-level` must come before it.

> **Retired IDs are rerouted, not failed.** Requesting one still returns an image — the server substitutes the GA replacement Google published instead of letting the call 404 — and announces the swap three ways: a `warnings` entry in the response JSON, an MCP `warning` log notification, and a `WARNING` record in the server log. Request a GA model directly.
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
> Imagen Ultra reroutes to the top Gemini image model rather than dropping to flash, and **`gemini-3-pro-image` is billed at a materially higher rate than `gemini-3.1-flash-image`** — price it with `dry_run` first.

*This tool may perform destructive updates.*

*This tool interacts with external entities.*

---

### Tool: **`generate_video`**

Generate videos using VEO models, which run on Vertex AI only (with GCS output and Vertex-only features). Google switched off Veo 3.1 on the Gemini Developer API on October 22, 2026; on a server using a Gemini API key this tool refuses and points to `generate_video_omni` (`gemini-omni-1.1-flash`) or Vertex AI.

| Parameters | Type | Description |
|-----------|------|-------------|
| `prompt` | string | Text description of the video to generate |
| `model` | string **required** | VEO model to use — no default; name one explicitly: `veo-3.1-generate-001` or `veo-3.1-fast-generate-001` |
| `aspect_ratio` | string *optional* | Video aspect ratio: `16:9` (default) or `9:16` |
| `resolution` | string *optional* | Output resolution: `720p`, `1080p`, or `4K` |
| `duration_seconds` | number *optional* | Video duration in seconds (4/6/8s). **Default `8.0`**, the longest and priciest option — an omitted duration on `veo-3.1-generate-001` bills a full 8s render (from $3.20 at 720p/1080p) |
| `include_audio` | boolean *optional* | Enable audio generation (default `false`) |
| `person_generation` | string *optional* | Policy for generating people: `allow_adult` or `allow_all` |
| `dry_run` | boolean *optional* | Return only the cost estimate; generates nothing |
| `director_notes` | string *optional* | What the clip must get right, in plain words. Gemini 4 Argon checks each take against them and retakes on a clear fail; see [Virtual director](#virtual-director) |
| `max_retakes` | integer *optional* | Retakes the director may spend when a check fails, 0 to 2 (default 1) |
| `audio_prompt` | string *optional* | Audio description |
| `negative_prompt` | string *optional* | Things to avoid in the video |
| `seed` | integer *optional* | Random seed for reproducibility |
| `image_uri` | string *optional* | Input image URI for image-to-video generation |
| `draft` | boolean *optional* | When `true`, renders a fast 720p draft on `gemini-omni-1.1-flash` instead of Veo (default `false`) |
| `draft_resolution` | string *optional* | Resolution for a `draft=true` pass. Naming one renders the draft on `gemini-omni-1.1-flash`; `360p` is a third of the 720p price |

**Available Models** (no default — `model` must be supplied):
- `veo-3.1-generate-001` - VEO 3.1 (4/6/8 seconds with audio support)
- `veo-3.1-fast-generate-001` - VEO 3.1 Fast (faster generation)

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
| `beats` | array | Ordered beat specs: `{prompt, duration_seconds?, seed?, first_frame_uri?, negative_prompt?, audio_prompt?}`. Capped at 20 beats per call |
| `aspect_ratio` | string *optional* | `9:16` (default, vertical social) or `16:9` |
| `model` | string *optional* | Veo model for every beat (default `veo-3.1-fast-generate-001`) |
| `include_audio` | boolean *optional* | Audio per beat (default `true`; Vertex only) |
| `add_bridges` | boolean *optional* | Generate a transition between consecutive beats (requires local beat outputs) |
| `animatic` | boolean *optional* | Render every beat with `gemini-omni-1.1-flash` for a fast 720p storyboard preview of the whole reel |
| `animatic_resolution` | string *optional* | Resolution for the animatic pass. Naming one renders it on `gemini-omni-1.1-flash`; `360p` is a third of the 720p price |
| `output_gcs_uri` | string *optional* | GCS URI for all outputs |

**Notes:**
- A failed beat is recorded in the manifest's `errors` list; the run continues
- Returns `{kind, aspect_ratio, segments[], total_duration_seconds, errors[]}`
- Typical flow: `animatic: true` to preview quickly, then re-run for full Veo renders

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
| `model` | string *optional* | Veo model (default fast) |
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
| `prompt` | string *optional* | Transition motion and style (default: `smooth cinematic cut between the two clips`) |
| `model` / `duration_seconds` / `aspect_ratio` / `include_audio` / `audio_prompt` / `negative_prompt` / `seed` / `output_gcs_uri` | *optional* | As above |

*This tool may perform destructive updates.*

*This tool interacts with external entities.*

---

### Tool: **`generate_video_omni`**

Fast conversational video generation through Google's Omni models (Interactions API). This is the fast path for drafts and iteration and the only family with multi-turn conversational editing. See [Fast drafts vs. high-fidelity](#fast-drafts-vs-high-fidelity).

`gemini-omni-1.1-flash` (the default) renders 360p/720p/1080p/4K with first/last-frame interpolation, image and video references, native audio, and scene extension (see `extend_video_omni`). It bills $0.10136/s at 720p — a hair above Veo Fast — and its `resolution="360p"` draft tier is about a third of that. `gemini-omni-flash-preview` was switched off on 2026-10-22 and is refused.

| Parameters | Type | Description |
|-----------|------|-------------|
| `prompt` | string | Text description of the video to generate. Role tags you write yourself (`<FIRST_FRAME>`, `<IMAGE_REF_0>`, `[# Sources ...]`) pass through untouched; otherwise the declarations are generated from the media arguments and echoed back as `effective_prompt` |
| `omni_model` | string *optional* | `gemini-omni-1.1-flash` (default). `gemini-omni-flash-preview` was switched off on 2026-10-22 and is refused. Arguments the model cannot honor are refused, never dropped |
| `image_uris` | array *optional* | Image URIs whose role the model infers — at most 8 images in total; a longer list is rejected |
| `first_frame_uri` / `last_frame_uri` | string *optional* | 1.1 only. Keyframe interpolation; `last_frame_uri` requires `first_frame_uri`, and the same URI for both loops the clip |
| `reference_image_uris` | array *optional* | 1.1 only. Bound to `<IMAGE_REF_0>`, `<IMAGE_REF_1>`, … in order |
| `reference_video_uris` | array *optional* | 1.1 only. Up to 3 clips of up to 3s each, bound to `<VIDEO_REF_0>`, … Audio in them is ignored |
| `input_video_uri` | string *optional* | A video to edit. On 1.1 a clip too large to inline is uploaded through the Files API (Gemini API key required — Vertex has no Files API) |
| `resolution` | string *optional* | 1.1 only. `360p`, `720p` (default), `1080p` or `4K`; the last two are upscaled from the base render |
| `aspect_ratio` | string *optional* | `16:9` (default) or `9:16`. Not sent when the request is an edit (an `input_video_uri` or `previous_interaction_id` makes it one) — the API rejects it on an edit task |
| `duration_seconds` | number *optional* | Video duration, 3–10 (default 6). Also not sent on an edit; the service then picks the rendered length |
| `previous_interaction_id` | string *optional* | Continue editing a prior omni result |
| `timeout_seconds` | integer *optional* | Deadline for create + polling (default 600); raise it for long queues |
| `director_notes` | string *optional* | What the clip must get right, in plain words. Gemini 4 Argon checks each take against them and retakes on a clear fail; see [Virtual director](#virtual-director) |
| `max_retakes` | integer *optional* | Retakes the director may spend when a check fails, 0 to 2 (default 1) |

**Notes:**
- No `seed` or `negative_prompt` on either model — put negatives in the prompt (`no dialogue`)
- A render above 720p, or any extension, exceeds the 4 MB inline response limit: on the Gemini API it is requested with `delivery="uri"` and downloaded here automatically; on Vertex AI, pass `output_gcs_uri`
- On an edit turn the response reports `duration_seconds: null`, and a `dry_run` quotes omni's 10s maximum as an upper bound
- Response includes an `interaction_id` for multi-turn editing and extension
- Google publishes one Omni output rate, pinned to 720p. 360p is quoted at the one-third ratio from the launch blog and 1080p/4K at the published 720p rate; every estimate says which in its `source_note`
- All output carries SynthID watermarking

*This tool may perform destructive updates.*

*This tool interacts with external entities.*

---

### Tool: **`edit_video`**

Conversational edit of a previously omni-generated video. Omni holds the video context server-side (retained ~14 days on Vertex AI; longer on the paid Gemini API), so you describe only the change.

| Parameters | Type | Description |
|-----------|------|-------------|
| `previous_interaction_id` | string | The `interaction_id` from a prior `generate_video_omni` response |
| `prompt` | string | The edit instruction (e.g. `make the sky stormy`) |
| `omni_model` | string *optional* | Must be the model that produced `previous_interaction_id` — the video context lives with it |
| `resolution` | string *optional* | `gemini-omni-1.1-flash` only; the resolution to render the edit at |
| `aspect_ratio` | string *optional* | Accepted but never sent — the API rejects it on an edit task, so it does not change the output |
| `duration_seconds` | number *optional* | Accepted but never sent, and it does not set the output length: the service picks it, predictable from neither this value nor the source (a measured 3s source edited with `duration_seconds=4` rendered 10.01s). The response reports `duration_seconds: null`; a real run bills the measured file and `dry_run` quotes omni's 10s maximum |
| `timeout_seconds` | integer *optional* | Deadline for the edit render (default 600) |

*This tool may perform destructive updates.*

*This tool interacts with external entities.*

---

### Tool: **`extend_video_omni`**

Append a seamless continuation to an existing video (`gemini-omni-1.1-flash` only). The model reads the last 10 seconds of the source as context and generates what happens next, keeping motion, characters and audio coherent across the join.

| Parameters | Type | Description |
|-----------|------|-------------|
| `prompt` | string | How the scene continues — `Extend this video`, `Continue the scene: the camera pans across the mountains`. In a timecode, `0s` is the start of the new footage |
| `previous_interaction_id` | string *optional* | A clip this server rendered. Nothing is uploaded and dialogue may be added |
| `input_video_uri` | string *optional* | A clip of your own — 10s or shorter on the Gemini API, 30s on Vertex AI (checked locally against the right one). Give exactly one of these two |
| `times` | integer *optional* | Extension turns to chain (default 1, maximum 4 — each appends up to `duration_seconds`, to a cumulative 40s counting the source) |
| `resolution` | string *optional* | `360p`, `720p` (default), `1080p` or `4K` |
| `reference_image_uris` / `reference_video_uris` | array *optional* | References for introducing a new character or object; they ride along on the first turn |
| `timeout_seconds` | integer *optional* | Deadline for EACH turn (default 600) |

**Notes:**
- Extension appends to the END of a clip only; prepending or extending the middle is not supported
- Extending an *uploaded* video is unavailable in the EEA, Switzerland and the UK; extending a model-generated one works everywhere
- **Cost grows quadratically.** A turn renders the whole assembled clip, not the 10s it appends (measured: a 3.01s source extended once returned 13.01s), so every turn re-bills the footage before it: `times=2` from a 3s source bills ~36s to make 20s of new footage. Each turn's `video_url` supersedes the previous one. Always `dry_run` a chain first
- A chain that fails part-way still returns the turns that rendered, with the `interaction_id` to resume from

*This tool may perform destructive updates.*

*This tool interacts with external entities.*

---

### Tool: **`loop_extend`**

Convenience wrapper that extends a Veo-generated video multiple times in one call. Each Veo extension adds ~7s, up to 20 times. This is the Veo extension path; `extend_video_omni` is the Omni one, and they are not interchangeable.

| Parameters | Type | Description |
|-----------|------|-------------|
| `video_uri` | string | The Veo-generated video to extend |
| `prompt` | string *optional* | What the video continues with (default: "continue the action") |
| `times` | integer *optional* | Number of ~7s extensions to apply, 1-20 (default: 1) |
| `model` | string *optional* | Veo model (default `veo-3.1-generate-001`) |
| `aspect_ratio` | string *optional* | `16:9` (default) or `9:16` — must match the source video |
| `include_audio` | boolean *optional* | Audio on extended sections (default `true`; Vertex only) |
| `output_gcs_uri` | string *optional* | GCS URI for output (required on Vertex) |

**Notes:**
- Veo 3.1 / Veo 3.1 Fast, on Vertex AI
- 720p

*This tool may perform destructive updates.*

*This tool interacts with external entities.*

---

## Fast drafts vs. high-fidelity

Two video paths, chosen by where you are in the workflow:

- **`gemini-omni-1.1-flash` (fast turnaround)** — 360p/720p/1080p/4K, conversational multi-turn editing, first/last-frame interpolation, video references, native audio and scene extension. No seeds or negative prompts. Great for drafts, storyboards and iteration. Reached via `generate_video_omni`, `edit_video`, `extend_video_omni`, `generate_video` with `draft=true`, and `generate_clip` with `animatic=true`.
- **Veo 3.1 / Fast (high-fidelity, Vertex AI only)** — up to 1080p/4K, seeds, first/last-frame control, reference images, extension. The path for final renders. Reached via `generate_video` (default) and `loop_extend`.

Typical workflows:
- **draft → finalize**: run `generate_video` with `draft=true` to preview quickly on omni, then re-run the same prompt with `draft=false` to render the final on Veo.
- **animatic → final**: run `generate_clip` with `animatic=true` (default `false`) to render each beat through `gemini-omni-1.1-flash` as a fast storyboard preview of the whole reel, then re-run with `animatic=false` to commit to full Veo renders.

---

## Virtual director

`generate_image`, `generate_video` and `generate_video_omni` take optional `director_notes`: what the shot must get right, in plain words. Without them nothing changes: no extra call and no extra cost. With them, Gemini 4 Argon turns the notes into checks, each quoting the words it came from; the take renders exactly as it would without notes and is reviewed against every check (`pass`, `fail`, `unsure` or `not_applicable`, with where in the image or when in the clip the problem is); a clear fail buys a retake, up to `max_retakes` (0 to 2, default 1), with your prompt unchanged plus fix notes built from the checks. The response's `director` block lists the checks, every take with its results and costs, `chosen_take` and `retakes_used`; `cost` covers every take and every director call, and each take's sidecar manifest carries its review.

- No confidence number is reported; the states have not been compared with human labels.
- A fail needs evidence, or it counts as `unsure`, and an `unsure` review stops instead of paying for a retake.
- Text inside a render is assessed, never followed: retake notes come only from the checks derived from your notes.
- No check assesses a person's identity, likeness, age, sex or gender, race or ethnicity, health, emotion, attractiveness or body.
- A `dry_run` with notes quotes the worst case. On the video tools, `timeout_seconds` bounds the whole directed call.

`DIRECTOR_MODEL` selects `gemini-4-argon` (default) or `gemini-3.8-flash` (about a third of the price). On Vertex AI, director calls use the global endpoint.

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

### Using Gemini API (Images + Omni Videos)

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
| `GOOGLE_APPLICATION_CREDENTIALS` | ✅ | Path to service account JSON key (inside container). Also accepts the key as inline JSON |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | — | The service account key as inline JSON, instead of mounting a file. Written by `gemini-media-mcp setup`; the server materialises it into a temp file |
| `DATA_FOLDER` | ✅ | Output directory path (must match host path in volume) |

### Gemini API Configuration (Images + Omni Videos)

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ | Your Gemini API key |
| `DATA_FOLDER` | ✅ | Output directory path (must match host path in volume) |

### Optional (both modes)

| Variable | Description |
|----------|-------------|
| `GCS_ALLOWED_BUCKETS` | Comma-separated allowlist for `gs://` inputs and `output_gcs_uri`. Unset (and no `VIDEO_GCS_BUCKET`) means `gs://` fetches only log a warning |
| `VIDEO_GCS_BUCKET` | Default bucket for large video output; also seeds the allowlist above |
| `FASTMCP_HOST` | Bind address for the sse/streamable-http transports; `--host` wins over it |
| `RUNNING_IN_CONTAINER` | Set to `true` by this image. Makes `DATA_FOLDER` mandatory and binds `0.0.0.0` instead of `127.0.0.1`. Leave it alone unless you are building your own image |
| `DIRECTOR_MODEL` | Model for `director_notes` reviews: `gemini-4-argon` (default) or `gemini-3.8-flash` |

---

## Google Cloud Setup

Video generation with VEO models needs Vertex AI: Google switched off Veo 3.1 on the Gemini Developer API on October 22, 2026, so a Gemini API key alone reaches video through Gemini Omni only. Vertex AI also adds GCS output plus Vertex-only features. To use Vertex AI:

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
