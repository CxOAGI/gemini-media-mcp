# Gemini Media MCP Server

Plan, generate, compose, and edit images and video with Gemini image models, Veo 3.1, and Gemini Omni, with a cost estimate before every call and an optional Gemini 4 Argon director that checks each take against your notes.

[What is an MCP Server?](https://www.anthropic.com/news/model-context-protocol)

## MCP Info

| Attribute | Details |
|-----------|---------|
| **Docker Image** | [cxoagi/gemini-media-mcp](https://hub.docker.com/repository/docker/cxoagi/gemini-media-mcp) |
| **Author** | [CxOAGI](https://github.com/CxOAGI) |
| **Repository** | [https://github.com/CxOAGI/gemini-media-mcp](https://github.com/CxOAGI/gemini-media-mcp) |

## Available Tools (13)

| Tools provided by this Server | Short Description |
|-------------------------------|-------------------|
| `plan_generation` | Pick the right tool + model for an intent, with costs and ruled-out options. Generates nothing |
| `generate_image` | Generate images using Gemini image models |
| `generate_storyboard` | Render a keyframe per shot and return a real storyboard (inline contact-sheet preview + full-size PNG and HTML on disk) |
| `generate_video` | Generate videos using VEO models (with optional fast `draft` mode) |
| `generate_clip` | Generate a whole multi-beat reel in one call, with optional bridges and a fast animatic preview |
| `generate_transition` | Veo first+last-frame transition between two stills |
| `generate_bridge` | Transition between two existing clips (frames extracted for you) |
| `generate_video_omni` | Fast conversational video via `gemini-omni-1.2-flash` (Interactions API) |
| `upscale_video_omni` | Upscale an approved Omni draft on Vertex AI |
| `edit_video` | Conversationally edit a previously omni-generated video |
| `get_omni_interaction` | Check on, and save, an Omni render that outlived its tool call |
| `extend_video_omni` | Append a seamless continuation to an existing video (`gemini-omni-1.2-flash`) |
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
| `resolution` | string *optional* | Exact video resolution (`360p`, `720p`, `1080p` or `4K`); for an upscale, the target |
| `source_video_uri` / `previous_interaction_id` | string *optional* | An existing clip or Omni interaction the plan edits, extends or upscales; carried into the routes' `params` |

**Notes:**
- Catches impossible combinations before you pay for the failure (4K on a 1K-only model, Veo on the Gemini API, GCS on the Gemini API)
- Every rejected model comes with a reason — nothing is dropped silently
- With 4 to 6 reference images only the Omni route is planned: Veo takes 3 and would send only the first 3. More than 6 is a conflict for every model
- A brief that asks for a new video and then an upscale (`..., then upscale it to 4K`, a bare `..., then upscale`, or `... and upscale the result`) is planned as two workflow steps: the render, then an `upscale_video_omni` call on its output, to 1080p when no target is named. A 4K named in the render clause (`a 4K video of a dog, then upscale it`) is the upscale's target, and the render is planned at 720p. The render is always planned at a resolution the target takes (720p for 4K, 360p for 720p, and 360p for 1080p on a cheap budget); a named render resolution the target cannot take is a conflict with `is_satisfiable: false`. On Vertex AI the upscale step says the output comes back inline unless you add `output_gcs_uri`; on the Gemini API, which has no standalone upscaling, the target resolution goes on the render instead
- A single shot of 11 to 15 seconds is planned first on `gemini-omni-1.2-flash`, which renders it in one call. Veo routes for it are an 8s `generate_video` render followed by one `loop_extend` (about 15s in all), and rank first only with a pinned Veo model, a seed or negative prompt, or `budget="best"`
- A transition between two frames is planned up to 10s, the longest any model interpolates between two endpoints; a longer one is the `transition_too_long` conflict, with no routes. With several beats, `duration_seconds` is the whole reel's runtime and each beat gets its share, snapped to Veo's 4/6/8s when it falls between them
- A `previous_interaction_id` is planned on the model this server recorded for it, set as `omni_model` in the route's `params`, or left out when there is no record so the tool continues on the interaction's own model. One recorded on the retired `gemini-omni-1.1-flash` plans no routes; the `interaction_model_retired` conflict names the tool to pass its saved clip to as `input_video_uri`
- On Vertex AI, an Omni route whose output is large (1080p or 4K, over 10s, an edit or an extension) carries the inline-delivery warning described under `generate_video_omni` as a caveat, unless `wants_gcs_output` is set

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

Generate videos using VEO models, which run on Vertex AI only (with GCS output and Vertex-only features). Google switched off Veo 3.1 on the Gemini Developer API on October 22, 2026; on a server using a Gemini API key this tool refuses and points to `generate_video_omni` (`gemini-omni-1.2-flash`) or Vertex AI.

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
| `draft` | boolean *optional* | When `true`, renders a fast 720p draft on `gemini-omni-1.2-flash` instead of Veo (default `false`). The draft runs as long as the Veo render will (4, 6 or 8 seconds), so a 3s request drafts at 4s, with a warning |
| `draft_resolution` | string *optional* | Resolution for a `draft=true` pass. Naming one renders the draft on `gemini-omni-1.2-flash`; `360p` is a third of the 720p price. A draft has no Cloud Storage destination, so on Vertex AI a 1080p or 4K draft comes back inline with a warning |

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
| `include_audio` | boolean *optional* | Audio per Veo beat (default `true`; Vertex only). Passing it with `animatic` is ignored with a warning |
| `add_bridges` | boolean *optional* | Generate a transition between consecutive beats (requires local beat outputs) |
| `animatic` | boolean *optional* | Render every beat with `gemini-omni-1.2-flash` for a fast 720p storyboard preview of the whole reel. Each beat runs as long as the Veo delivery will render it (4, 6 or 8 seconds), with a warning when that changes it. With `add_bridges`, each bridge renders 4s on Omni 1.2 between the beats' frames |
| `animatic_resolution` | string *optional* | Resolution for the animatic pass. Naming one renders it on `gemini-omni-1.2-flash`; `360p` is a third of the 720p price. An animatic has no Cloud Storage destination, so on Vertex AI a 1080p or 4K animatic, or a beat over 10s, comes back inline with a warning, given before any beat renders |
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

Generate video with **`gemini-omni-1.2-flash`**, the default for Omni, drafts and animatics. It is generally available on both the Gemini API and Vertex AI under the same model ID.

| Parameters | Type | Description |
|-----------|------|-------------|
| `prompt` | string | Text description of the video. Direct continuous camera movement, reference consistency, dialogue, speaker assignments, music and sound effects here |
| `omni_model` | string *optional* | `gemini-omni-1.2-flash` unless set; naming the retired `gemini-omni-1.1-flash` runs on 1.2 (see the notes below). Left unset with a `previous_interaction_id`, it is the model this server recorded for that interaction; naming a different one is refused, since an interaction cannot change models; the error gives the two ways out (leave `omni_model` unset, or pass the saved video as `input_video_uri` to start a new interaction). `edit_video` and `extend_video_omni` follow the same rule |
| `omni_backend` | string *optional* | `auto` (default), `vertex`, or `gemini_api`. Auto prefers a configured Gemini API key; `OMNI_BACKEND=vertex` keeps requests on Vertex |
| `duration_seconds` | number *optional* | 3 to 15 (default 6), rounded to a whole second. Under 3 (including 0) renders 3s and over 15 renders 15s, with a warning on the real run; a dry run reports `requested_duration_seconds` beside the length it priced. Negative or non-finite values are refused, as are the two combinations below that cannot render past 10s |
| `resolution` | string *optional* | `360p`, `720p` (default), `1080p` or `4K`; the last two include upsampling. Left unset on a continuation of a `previous_interaction_id`, it is the resolution recorded for that clip |
| `aspect_ratio` | string *optional* | `16:9` (default) or `9:16` |
| `first_frame_uri` / `last_frame_uri` | string *optional* | Start/end keyframes. A last frame requires a first frame; interpolation renders 3–10s, and a `last_frame_uri` with a longer `duration_seconds` is refused rather than shortened. The same image twice makes a loop, sent once with both roles naming it |
| `reference_image_uris` | array *optional* | Up to 6 subject/style references, with explicit role bindings |
| `image_uris` | array *optional* | Images whose role the model infers |
| `reference_video_uris` | array *optional* | Up to 3 clips of up to 3s each, for new generations only. Output over 10s takes at most 1: 2 or 3 with a longer `duration_seconds` is refused. Reference audio is ignored |
| `input_video_uri` | string *optional* | A clip of up to 15s to edit. On Vertex AI a `gs://` source is read by the service directly (a real run also downloads it once, up to 50 MB, to check its length), and any other source is sent inline, up to 20 MB. On the Gemini API a clip too large to send inline is uploaded through the Files API. Editing an uploaded clip is not available in the European Economic Area (Google's restriction): the service refuses, and this server cannot check it first. |
| `previous_interaction_id` | string *optional* | Continue an existing conversation on its original model and backend |
| `output_gcs_uri` | string *optional* | Deliver to an allowed Cloud Storage bucket on Vertex AI; ignored with a warning on the Gemini API. Vertex AI treats the URI as a prefix and writes the video to a new object under it (`gs://bucket/clips/fox.mp4` gives `gs://bucket/clips/fox.mp4/<number>.mp4`), so pass a prefix ending in `/`; a URI ending in a file name gets a warning. The response's `video_url` names the object written, and the server downloads it once (up to the 50 MB fetch cap) to measure its length and resolution. The safer delivery for a large output on Vertex AI, which comes back inline without it (see the notes below) |
| `timeout_seconds` | integer *optional* | Deadline for the whole interaction (default 210). A render still running at the deadline keeps rendering and bills: the error names its `interaction_id`, and `get_omni_interaction` collects it. A request refused with a 429 is sent again after the wait it asks for (else 10s, then 20s), at most twice, while the deadline leaves a render's worth of time. A create that fails with a server error is never sent again, since it may already have started a billed render |
| `dry_run` | boolean *optional* | Validate and quote without rendering |
| `director_notes` | string *optional* | What the clip must get right, in plain words. Gemini 4 Argon checks each take against them and retakes on a clear fail; see [Virtual director](#virtual-director) |
| `max_retakes` | integer *optional* | Retakes the director may spend when a check fails, 0 to 2 (default 1). Each retake renders from your inputs again, not as an edit of the take before it |

**Notes:**
- An edit (`input_video_uri` or `previous_interaction_id`) sends no `duration_seconds` or `aspect_ratio`, so `requested_duration_seconds` is null: the service picks the length, the real run reports the measured one, and a dry run quotes an upper bound. A real run that cannot measure its output reports `duration_seconds: null` and the bound it billed as `duration_upper_bound_seconds`
- A render above 720p, an extension turn, an edit (which can run 15s), or any other output over 10s can exceed the 4 MB inline response limit the Gemini API reference gives, so the Gemini API requests it as a Google-hosted URI rather than inline, and it is downloaded automatically
- **Large outputs on Vertex AI.** Vertex AI writes a render to Cloud Storage when the request carries `output_gcs_uri`, and otherwise returns the video bytes in the response, for which its [upscaling page](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/video/upscale-videos) states no size limit. One policy covers `generate_video_omni`, `edit_video`, `extend_video_omni` and `upscale_video_omni`: an output that is 1080p or 4K, over 10s, an edit (up to 15s), an extension turn, or an upscale of a source over 10s or of unknown length is rendered and returned inline when no `output_gcs_uri` is given. The dry run and the real response both carry a warning, given before anything renders, that `output_gcs_uri` is the safer delivery for an output that large. `generate_video` drafts and `generate_clip` animatics, which cannot deliver to Cloud Storage, get a warning suggesting a smaller output or `generate_video_omni` with `output_gcs_uri`
- If a render completes but its response carries no video, the error says so: the video was requested inline and no bytes came back. It gives the size as the likely reason when the output was large, and names `output_gcs_uri`. The interaction completed, so the render has been billed, and rendering it again bills again
- The model generates native synchronized audio. **Audio uploads, audio references and voice editing are unsupported by the current API**, even though launch demos show voice-conditioned examples
- Output-video pricing is $12 per million tokens: **$0.023172/s at 360p, $0.069504/s at 720p, $0.104256/s at 1080p, and $0.208512/s at 4K**. Quotes include one frame of encoder allowance; input and text tokens are additional when reported. See the [API guide](https://aistudio.google.com/docs/omni) and [pricing](https://aistudio.google.com/docs/pricing#gemini-omni-1.2-flash)
- **`gemini-omni-1.1-flash` is retired on this server.** A call that names it, or Vertex AI's spelling `gemini-omni-1.1-flash-preview`, runs on `gemini-omni-1.2-flash` with 1.2's capabilities and pricing. Its response, dry run included, reports `requested_model` and carries a warning saying the call ran on 1.2. Google applies the same redirect to that ID from **December 3, 2026**; this server applies it now, so a client that hard-codes the ID keeps working. `plan_generation` never offers 1.1, and plans a `pinned_model` of 1.1 as 1.2 with a note saying why
- An interaction created on 1.1 cannot be continued, because an interaction cannot change models. `generate_video_omni`, `edit_video`, `extend_video_omni` and `upscale_video_omni` refuse a `previous_interaction_id` recorded on 1.1 before anything is fetched, quoted or billed, dry run included. The error names the clip that render saved (from its sidecar manifest, when one is recorded) and the call to make instead: pass the clip as `input_video_uri` to `extend_video_omni` or `upscale_video_omni`, or to `generate_video_omni` to edit it on 1.2
- The old `gemini-omni-flash-preview` endpoint was switched off October 22; requests to it are refused

*This tool may perform destructive updates.*

*This tool interacts with external entities.*

---

### Tool: **`edit_video`**

Refine the same clip across multiple turns: pass `previous_interaction_id` and a `prompt` describing a background/product swap, lighting or weather change, or a new aspect ratio. Omni 1.2 accepts source clips up to **15s**. The edit runs on the model and backend that created the conversation.

| Parameters | Type | Description |
|-----------|------|-------------|
| `previous_interaction_id` | string | The `interaction_id` from a prior Omni response |
| `prompt` | string | The edit instruction; put framing changes here |
| `omni_model` | string *optional* | Left unset, the model this server recorded for the interaction (1.2 when there is no record). Naming a different model is refused, since an interaction cannot change models. An interaction created on the retired `gemini-omni-1.1-flash` cannot be edited: the error names its saved clip, to pass as `input_video_uri` to `generate_video_omni` |
| `omni_backend` | string *optional* | `auto` (default), `vertex` or `gemini_api`. One that contradicts the backend the interaction was created on is refused |
| `reference_image_uris` | array *optional* | Images to guide the edit: up to 6 on the Gemini API and 2 on Vertex. More is refused before anything is fetched or quoted |
| `resolution` | string *optional* | `360p`, `720p`, `1080p` or `4K`; left unset, the edit keeps the resolution recorded for the interaction (720p when none is recorded) |
| `output_gcs_uri` | string *optional* | Deliver to an allowed Cloud Storage bucket (Vertex AI only). An edit can run 15s, so on Vertex AI without it the edit comes back inline with a warning |
| `duration_seconds` / `aspect_ratio` | *optional* | Accepted for compatibility and omitted from the edit request |
| `timeout_seconds` | integer *optional* | Deadline for the whole interaction (default 210) |
| `dry_run` | boolean *optional* | Quote without rendering |

**Notes:**
- Real results report measured duration when available; dry runs use the model's maximum as a labelled bound
- Vertex interaction context is retained up to 7 days

*This tool may perform destructive updates.*

*This tool interacts with external entities.*

---

### Tool: **`extend_video_omni`**

Give exactly one `previous_interaction_id` or `input_video_uri`, plus a prompt that explicitly requests continuation. Omni uses the final **10s** of the source as context. Every turn renders the assembled clip, so each turn bills the footage from prior turns again.

| Parameters | Type | Description |
|-----------|------|-------------|
| `prompt` | string | How the scene continues; ask for the extension explicitly. A `previous_interaction_id` turn cannot declare its task, so the service infers extension from the prompt |
| `previous_interaction_id` / `input_video_uri` | string | Exactly one: a clip this server generated, or a clip to upload |
| `omni_model` | string *optional* | Left unset, continues a `previous_interaction_id` on the model this server recorded for it and extends an `input_video_uri` on 1.2; naming a model other than the recorded one is refused. An interaction created on the retired `gemini-omni-1.1-flash` cannot be extended: the error names its saved clip, to pass here as `input_video_uri` |
| `duration_seconds` | number *optional* | Each turn's **3–10s** increment, as a whole number of seconds (any other value is refused, not rounded). Left unset, it is not sent and the service appends its default 10s; set, it is sent on every turn |
| `times` | integer *optional* | Turns to chain (default 1), up to a **30s total on the Gemini API**, and at most 30 divided by the turn length per call (**3 turns** at the default 10s, 10 at 3s). Near the ceiling a turn appends the whole seconds that still fit (a 3.03s source's third 10s turn appends 6s, to 29.03s), and a turn with less than 3s left is not run. Vertex accepts inputs of any duration and retains up to their **last 30s** before appending each segment, so each output is at most **40s** |
| `reference_image_uris` | array *optional* | Images that introduce new elements; video references are unsupported for extension |
| `resolution` | string *optional* | As for `generate_video_omni` |
| `output_gcs_uri` | string *optional* | Cloud Storage destination for the final turn (Vertex AI only). A turn returns the whole clip so far, so on Vertex AI each turn sent without it (every turn but the last, when it is given) comes back inline with a warning; to send every turn to Cloud Storage, extend one turn per call (`times=1`) with `output_gcs_uri` |
| `omni_backend` | string *optional* | `auto` (default), `vertex` or `gemini_api`. The whole chain runs on one backend, and one that contradicts a `previous_interaction_id`'s backend is refused |
| `max_cost_usd` | number *optional* | Refuse a chain whose estimate exceeds this, before the first turn |
| `timeout_seconds` | integer *optional* | Deadline for each turn (default 210) |
| `dry_run` | boolean *optional* | Estimate every turn without rendering |

**Notes:**
- If the service reads a `previous_interaction_id` turn as an edit, a set duration makes it refuse the turn before rendering, and the error says nothing was billed for that turn and to ask for the extension explicitly in the prompt before continuing from the same id. With no duration set, such a turn is rendered and billed as an edit
- The source's length prices each turn and trims the chain at the ceiling. It comes from this server's record of a `previous_interaction_id`, or from measuring an uploaded source: a local file on dry and real runs, anything else on a real run. A `gs://` source on Vertex is read by the service directly and downloaded once (up to 50 MB) only to measure it. A dry run does not download a remote source, so it prices an unmeasured one at an upper bound
- Completed turns remain in the response if a later turn fails; the final `interaction_id` continues the chain

*This tool may perform destructive updates.*

*This tool interacts with external entities.*

---

### Tool: **`upscale_video_omni`**

Upscale an approved draft with Omni 1.2 on **Vertex AI** while preserving timing and audio. Provide exactly one `input_video_uri` or a Vertex `previous_interaction_id` from a 1.2 render, plus a target `resolution` (default `1080p`):

| Source | Supported targets |
| --- | --- |
| 360p | 720p, 1080p |
| 720p | 1080p, 4K |

| Parameters | Type | Description |
|-----------|------|-------------|
| `input_video_uri` / `previous_interaction_id` | string | Exactly one: the video to upscale, at most **15s** long |
| `resolution` | string *optional* | Target resolution (default `1080p`) |
| `source_resolution` | string *optional* | `360p` or `720p`. Needed on a dry run of a remote source with no record, and for a source this server cannot download (a record without a resolution is settled by measuring the clip it saved, on a real run); a value that disagrees with the record or the measured dimensions is refused |
| `output_gcs_uri` | string *optional* | Cloud Storage destination. Every 1080p or 4K target, the default included, and a 720p target from a source longer than 10s or of unknown length (counted as 15s), can exceed the 4 MB inline response limit. Without `output_gcs_uri` the upscale is rendered and returned inline, as Vertex AI's upscaling page documents, and the dry run and the real response carry a warning that `output_gcs_uri` is the safer delivery for an output that large |
| `max_cost_usd` | number *optional* | Refuse an upscale whose estimate exceeds this |
| `timeout_seconds` | integer *optional* | Deadline for the whole interaction (default 210) |
| `dry_run` | boolean *optional* | Quote without rendering |

**Notes:**
- The source's resolution and length come from this server's record of the interaction, or from measuring the file. A local or HTTP source is sent inline, which Vertex caps at **20 MB**; a `gs://` source is read by the service directly, and a real run downloads it once (up to 50 MB) only to measure it
- A `previous_interaction_id` recorded on the retired `gemini-omni-1.1-flash` is refused, and the error names its saved clip to pass as `input_video_uri`
- An upscale interaction cannot be continued by id: the service refuses an upscale of one ("Exactly one input video is required for the upscale task"). Given one, `upscale_video_omni` and `extend_video_omni` send the clip it saved as `input_video_uri`, and `edit_video` edits that clip through `generate_video_omni`, at its recorded resolution; the response says so. When the clip is gone, the call is refused before anything is sent, naming it.
- A 360p → 4K conversion needs two calls through 720p (the second can name the first's interaction); 1080p → 4K is unsupported
- Standalone upscaling bills zero input tokens. Output rates are 2,896 tokens/s for 360p → 720p or 720p → 1080p, 5,792 for 360p → 1080p, and 11,584 for 720p → 4K, at $12 per million tokens. See [standalone upscaling](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/video/upscale-videos)
- A quote prices the requested pair at the source's length (15s when unknown); a measured source is priced as measured, and a length that was not measured gets one frame of encoder allowance. `max_cost_usd` refuses a real run over the cap, and a dry run returns its quote with `would_be_refused`. A real run whose output comes back inline is metered at the length measured from the rendered file and at the resolution pair that actually rendered; if that pair has no published rate, the requested pair is priced and the cost detail says so. An output delivered to Cloud Storage is downloaded once (up to the 50 MB fetch cap) and measured the same way; one that cannot be read is priced as an estimate at the source's length, or 15s when that is unknown

*This tool may perform destructive updates.*

*This tool interacts with external entities.*

---

### Tool: **`get_omni_interaction`**

Check on an Omni render by its `interaction_id`, and save it once it has finished: one whose tool call timed out (the error names the id) or was cut off by the MCP host. The service keeps rendering and bills it either way; this tool only reads, so it never bills.

| Parameters | Type | Description |
|-----------|------|-------------|
| `interaction_id` | string | The id the timed-out call reported |
| `omni_backend` | string *optional* | `gemini_api` or `vertex`, where the interaction was created. Defaults to the backend this server recorded for it (a call that times out records it), else `OMNI_BACKEND` |
| `wait_seconds` | integer *optional* | How long to wait for a render still in progress, up to 1800 (default 0, one read). Many MCP hosts end a tool call near 240s |

**Notes:**
- The response carries the service's `status`; while the render is in progress nothing is saved
- Once it has completed, the video is saved (or, delivered to Cloud Storage, left there and downloaded once to measure), and the response carries `video_url`, the measured `duration_seconds` and `resolution`, and `cost` at those measurements. An upscale is priced at its recorded source resolution; an interaction this server has no record of is priced as a render
- The interaction is recorded like any render, so `edit_video`, `extend_video_omni` and `upscale_video_omni` can continue from its id. One already saved is returned from its record without reading the service

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

- **`gemini-omni-1.2-flash` (fast turnaround)** — 360p/720p/1080p/4K, conversational multi-turn editing, first/last-frame interpolation, video references, native audio, scene extension and, on Vertex AI, standalone upscaling. No seeds or negative prompts. Great for drafts, storyboards and iteration. The default for `generate_video_omni`, `extend_video_omni`, `generate_video` with `draft=true` and `generate_clip` with `animatic=true`, and the only model `upscale_video_omni` runs. A continuation (`edit_video`, or any `previous_interaction_id`) stays on the model that created the interaction.
- **Veo 3.1 / Fast (high-fidelity, Vertex AI only)** — up to 1080p/4K, seeds, first/last-frame control, reference images, extension. The path for final renders. Reached via `generate_video` (default) and `loop_extend`.

Typical workflows:
- **draft → finalize**: run `generate_video` with `draft=true` to preview quickly on omni, then re-run the same prompt with `draft=false` to render the final on Veo.
- **animatic → final**: run `generate_clip` with `animatic=true` (default `false`) to render each beat through `gemini-omni-1.2-flash` as a fast storyboard preview of the whole reel, then re-run with `animatic=false` to commit to full Veo renders.
- **draft → upscale** (Vertex AI): approve a 360p or 720p render, then pass it to `upscale_video_omni` for a 1080p copy (or 4K from 720p) that keeps its frames, timing and audio. The copy comes back inline unless you pass `output_gcs_uri`, the safer delivery for it.

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
