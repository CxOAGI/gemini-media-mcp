# Gemini Media MCP

Plan, generate, compose, and edit images and video with Gemini image models, Veo 3.1, and Gemini Omni, with a cost estimate before every call and an optional Gemini 4 Argon director that checks each take against your notes.

## What it does

- **Plan:** `plan_generation` ranks tool and model options for an intent with costs, and generates nothing.
- **Generate:** images with Gemini, video with Veo 3.1, fast video with Omni.
- **Compose:** storyboards, multi-beat reels, transitions between stills, bridges between clips.
- **Edit and extend:** conversational video edits, seamless scene extension, loop and extend, and Omni upscaling on Vertex AI.
- **Direct:** optional `director_notes` have Gemini 4 Argon turn your notes into checks, review each take, and retake on a clear fail.
- **Cost-aware:** any call runs `dry_run` for a quote; real runs report metered cost and its pricing source.

## Quick start

```bash
uvx gemini-media-mcp setup
```

The `setup` wizard walks you through the whole onboarding flow end-to-end:

1. Pick a credential mode: **Gemini API** (images + Omni video, easier to set up) or **Vertex AI** (images + Veo and Omni video, adds GCS output and Vertex-only features).
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

- For images **and** video with the simplest setup: a Gemini API key ([setup instructions](#gemini-api-setup)). Video on the Gemini API runs on Gemini Omni (`generate_video_omni`); Google switched off Veo 3.1 there on **October 22, 2026**, and the Veo tools refuse on that backend and name the alternatives.
- For images and video with Veo, GCS output and Vertex-only features (e.g. controllable audio: `include_audio` maps to Veo's `generate_audio` config): a Google Cloud project with the Vertex AI API enabled and a service account with Vertex AI permissions ([setup instructions](#vertex-ai-setup))

### Environment Variables

For Vertex AI (images + video, GCS output, Vertex-only features):

```bash
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=us-central1
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

**→ See [Vertex AI Setup](#vertex-ai-setup) for detailed instructions**

Alternatively, for the Gemini API (images, plus video through Gemini Omni; Veo needs Vertex AI):

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
| `OMNI_BACKEND` | Which backend serves the omni tools: `auto` (default), `vertex`, or `gemini_api`. `auto` uses the Gemini Developer API whenever `GEMINI_API_KEY` is set — even on a Vertex server, which is a different project, billing path and quota pool. Set `vertex` to keep omni traffic in the Vertex project. A per-call `omni_backend` argument overrides it; every omni response reports `backend` and `backend_reason`. |
| `DIRECTOR_MODEL` | Model the [virtual director](#virtual-director) uses: `gemini-4-argon` (default) or `gemini-3.8-flash`, about a third of the price |
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


### Virtual director

`generate_image`, `generate_video` and `generate_video_omni` take optional `director_notes`: what the shot must get right, in plain words. Without them nothing changes: no extra call and no extra cost.

With notes, the tool:

1. Turns the notes into checks with Gemini 4 Argon. Each check quotes the words of the notes it came from, and a check that quotes nothing in the notes is dropped as invented. Notes that cannot be checked by looking, such as file format or delivery, are listed under `skipped_notes`.
2. Renders the take exactly as it would without notes, then has Argon review it against every check. The reviewer is also shown the images the render was given: always the image being edited and the first and last frames, and the first three of `image_uris` and of `reference_image_uris`. Each is named for what it is, and lists are numbered in your order, so in the notes "reference 1" is `reference_image_uris[0]` (on omni, `<IMAGE_REF_0>` works too) and "input image 1" is `image_uris[0]`. Each take lists what the reviewer was shown under `references_shown`; anything it was not shown is named in a warning, and a check that needs it comes back `unsure`. Each check comes back `pass`, `fail`, `unsure` or `not_applicable`, with where in the image or when in the clip the problem is. Clips are sampled at 4 frames a second. A fail that does not say where counts as `unsure`.
3. When a check fails and `max_retakes` allows (0 to 2, default 1), renders again with your prompt, unchanged, plus a "Director's notes for this take" list made from the failed checks' fix instructions, and reviews that take. The notes say they win wherever they conflict with your prompt. A check that particular text appears (a headline, a label) is a `text` check carrying the `literal` the notes asked for and, where the notes say it, a `target` such as "the smaller line under the headline". Its fix note is always `<Target> must read exactly: "<literal>"`, written by the server, because a model-written fix puts how to compare the text ("with no trailing period") after it and the next render prints that too; a target that is long, quotes anything, names the text itself or uses a comparison word falls back to "The text". How the text is compared stays in the check, which only the reviewer reads. A text check is not judged by the reviewer but transcribed by it: the reviewer copies out the text it sees, and the server compares that with the literal character for character, with runs of whitespace collapsed and curly quotes straightened; no transcription is `unsure`. Image models tend to add a trailing period to a short line, so write the literal the way the copy is actually set rather than against the model's habit.

It returns the first take that passes, or else the take with the fewest fails (on a clip, then the least time showing them), and says which. A retake can fix one check and break another, so it is never assumed to be better: every take is reviewed, and the take returned can be an earlier one. The top-level `prompt` is always the one you sent; when the returned take is a retake, `effective_prompt` is the text that rendered it. The response's `director` block lists the checks, every take with its prompt, media, verdict, per-check results and costs, `chosen_take`, and `retakes_used`. A retake that never rendered is listed too, with verdict `not_rendered`, its error and its prompt, so the fix notes can be read either way. Each omni take carries its own `interaction_id`, so a retake that was not returned can still be continued or reconciled. `review_attempts` counts the review requests; when it is 2, `retried_after` says why: `quota` (a 429, which is not billed) or `transient` (a server error, which may be, and a warning says so). `cost` covers everything the call billed: every take and every director call. Each take's sidecar manifest records its review under `director_review`, together with your notes, the checks and the director model, so the manifest can be read without the response.

What holds throughout:

- **No confidence number.** The reviewer is never asked how sure it is, and none is reported. A model's own certainty is not calibrated; the states have to be compared with human labels before a number means anything.
- **A clip is judged second by second.** A check that must hold for the whole clip (steam throughout, no hands in frame) is a `throughout` check: the reviewer answers `y`, `n` or `?` for each second, and the server decides. Any `n` is a fail with the exact seconds as evidence; any `?`, or a list shorter than the clip, is `unsure`. A clip is judged in whole seconds, its length rounded to the nearest one (a 6.04 s clip is six seconds; an 8.6 s clip is nine, the last covering 8.0–8.6 s), so a gap shorter than a second fails that whole second.
- **The media cannot steer its own retake.** Text inside a render is assessed, never followed, and retake notes are built only from the fix instructions derived from your notes plus time ranges, never from what the reviewer wrote about the media.
- **The render always lands.** A review that hits a 429 or a server error is retried once after a short wait, if its time budget allows. The call that turns notes into checks is also retried once after a timeout (`derive_attempts`, `derive_retried_after`); on the video tools only while at least half of `timeout_seconds` stays free for the render. A render the service turns away with a 429 or a server error is tried again after 10 s and then 30 s, or after the wait the 429 asked for, each time only when the time left fits the wait, a render and a review (`render_attempts`, `render_retried_after`, `render_retry_waits`; a retry that cannot fit says `reason: "budget"`). A render that was already submitted is never sent again. Image quotas can allow only a few renders a minute and recover over tens of seconds, so a retake is often refused seconds after the take before it. One that still cannot run (quota, timeout, a safety block, a clip over 14 MB inline) is recorded as `abstained` with a reason, and the take is still returned. An `unsure` review stops instead of paying for a retake.
- **No judgments about people.** Checks never assess a person's identity, likeness, age, sex or gender, race or ethnicity, health, emotion, attractiveness or body.

The kinds of thing a review catches: a line set in capitals when the notes asked for title case, a footer the prompt never mentioned, a size annotation printed on a label, hands entering the frame at 00:01–00:04, a clip with no steam for its first second.

On the video tools, `timeout_seconds` bounds the whole directed call, because MCP hosts cancel long calls and a cancelled call has already billed its renders. Each take renders in the time left; a review that could not finish is skipped (`abstained`, reason `deadline`), and a retake is not started when the previous take shows it would not fit. Raise `timeout_seconds`, and your host's limit, to leave room for retakes.

A `dry_run` with notes quotes the worst case: every take rendered and reviewed, each review at its output cap, no cache hits. Three charges cannot be priced in advance and are left out: an image render's input tokens (`generate_image` quotes output only), a director request that fails without reporting its usage, and a render that fails with a server error or a timeout and is tried again, which may bill for the failed attempt; a warning names either of the last two if it happens. A retake that fails is listed with `submission` saying how far it got: `submitted` (it may still bill, and its quote is counted where the tool gives one), `unknown` (it may have reached Google, so `cost` is a lower bound) or `none` (nothing billed). Inputs are validated by the tool's own dry run before the first director call is paid for, and checks derived for the same notes, prompt and model earlier in the server's life are reused rather than derived again; `checks_source` says `compiled`, `cache`, or `failed` when the notes could not be turned into checks, and a cached set adds no derive cost. Director calls use `gemini-4-argon` ($2 / $10 per 1M input / output tokens through 2026-11-30, then $4 / $20); set `DIRECTOR_MODEL=gemini-3.8-flash` for about a third of that. On Vertex AI they go to the global endpoint, one of the three locations that serve Argon.

```json
{
  "prompt": "A ceramic mug on a walnut table, morning light",
  "model": "gemini-3.1-flash-image",
  "director_notes": "The label reads GOLDEN HOUR in serif capitals. No people or hands in the shot. The handle faces right.",
  "max_retakes": 1
}
```

### plan_generation

**Start here when you are not sure which tool or model to use.** Describe what you want in plain language and get back ranked, ready-to-call plans — which tool, which model, which parameters, why that model won, what each option costs, and which models were ruled out and for what reason.

It generates nothing, costs nothing, and is instant: pure rule-based routing over this server's capability tables, not a model call. It never replaces the explicit `generate_*` tools — it tells you how to drive them.

**Parameters:**
- `intent` (required): plain language, e.g. `a 3-beat vertical reel about coffee`, `a poster with the words GRAND OPENING`
- Optional overrides (these always beat what's inferred from the text): `budget` (`cheap`/`balanced`/`best`), `media_kind`, `aspect_ratio`, `image_size`, `duration_seconds`, `num_beats`, `needs_text_rendering`, `needs_4k`, `needs_audio`, `needs_extension`, `num_reference_images`, `wants_gcs_output`, `is_draft`, `pinned_model`, `resolution` (exact video resolution, or an upscale's target), `source_video_uri` and `previous_interaction_id` (an existing clip or Omni interaction the plan edits, extends or upscales; carried into the routes' `params`)

**Returns:** ranked `routes` (tool, model, ready-to-use `params`, score, rationale, caveats, cost), `rejected` models **with reasons**, `conflicts`, a suggested multi-step `workflow`, and `notes`.

It catches requests that cannot work *before* you pay for the failure — 4K on a 1K-only model, Veo on the Gemini API, GCS output on the Gemini API — and reports each as a conflict with a fix.

An intent that asks for a new video and then an upscale is planned as two workflow steps: the render, then an `upscale_video_omni` call that takes its output. That covers `a 6 second video of a fox in snow, then upscale it to 4K`, a bare `..., then upscale`, and `... and upscale the result`; with no target named, the upscale is to 1080p. A 4K named in the render clause (`a 4K video of a dog, then upscale it`) is the upscale's target: the render is planned at 720p and the upscale at 4K. The render is always planned at a resolution the target takes: 720p for 4K, 360p for 720p, and for 1080p 360p on a cheap budget, else 720p. A render resolution you name that the target cannot take (360p, then upscale to 4K) leaves the render routes in the plan beside the `upscale_resolution` conflict, with no workflow and `is_satisfiable: false`. On Vertex AI the upscale step's rationale says the output comes back inline unless you add `output_gcs_uri` ([Large outputs on Vertex AI](#large-outputs-on-vertex-ai)). On the Gemini API, which has no standalone upscaling, the target resolution goes on the render instead. An intent to upscale a video you already have (`upscale my video to 4K`) plans the upscale alone.

A single shot of 11 to 15 seconds (`a 12 second video of waves on a beach`) is planned on `gemini-omni-1.2-flash`, which renders it in one call, and that route ranks first. Veo routes for it are `loop_extend` chains: an 8s `generate_video` render, then one ~7s extension, about 15s in all, as the route's caveat says. When a Veo chain ranks first (a pinned Veo model, a seed or negative prompt, or `budget="best"`), the workflow lists the 8s render and the extension as two steps. 9 or 10 seconds still snaps to Veo's 8s, with a caveat.

A transition between two frames is planned up to 10s, the longest any model interpolates between two endpoints. A longer one is the `transition_too_long` conflict, with no routes: extending a transition would carry the shot past its last frame, so ask for 10s or less and extend the result afterwards if you need more. With several beats (`num_beats`, or a brief that names them), `duration_seconds` is the whole reel's runtime and each beat gets its share; a share Veo cannot render snaps to its 4/6/8s ladder, and the caveat gives the total that results.

With a `previous_interaction_id`, the plan continues the interaction on the model this server recorded for it, set as `omni_model` in the route's `params`. With no record, `omni_model` is left out, so the tool continues on whichever model made the interaction. An extension asks for the seconds requested: `extend this by 6 seconds` is one 6s turn (`duration_seconds=6`), not the 10s default. A runtime total (`make this 20 seconds long`) is measured against the clip's recorded length, and the route's quote is priced from that length, as `extend_video_omni`'s own quote is. An interaction recorded on the retired `gemini-omni-1.1-flash` plans no routes: the `interaction_model_retired` conflict names the tool to pass the clip it saved to as `input_video_uri` instead (`generate_video_omni` for an edit, `extend_video_omni` for an extension, `upscale_video_omni` for an upscale). On Vertex AI, any Omni route whose output is large carries the [inline-delivery warning](#large-outputs-on-vertex-ai) as a caveat, unless `wants_gcs_output` is set. A route the tool would refuse is not planned: a recorded clip over 15s to edit or upscale, a clip already at the Gemini API's 30s extension ceiling, more than 6 reference images, or more than 2 images on a Vertex AI edit each come back as a conflict that says what to change. With 4 to 6 reference images only the Omni route is planned: Veo takes 3 and would send only the first 3.

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
- `director_notes`, `max_retakes`: Have the [virtual director](#virtual-director) check the image against your notes and retake on a fail

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

Generate videos using VEO models, which run on **Vertex AI** only. Vertex AI also provides GCS output and Vertex-only features such as `include_audio`. Google switched off Veo 3.1 on the Gemini Developer API on **October 22, 2026**; on a server using a Gemini API key this tool refuses before quoting anything and points to `generate_video_omni` (`gemini-omni-1.2-flash`, Google's named replacement) or Vertex AI.

**Parameters:**
- `prompt` (required): Text description of the video
- `model` (required — there is no default; name one explicitly):
  - `veo-3.1-generate-001`: Highest quality, 4/6/8s duration, audio support
  - `veo-3.1-fast-generate-001`: Faster generation with audio support
- `aspect_ratio`: `16:9` (default) or `9:16`
- `resolution`: Output resolution, `720p`, `1080p`, or `4K`
- `duration_seconds`: Video duration (4/6/8s), **default `8.0`** — the longest and most expensive option. Omitting it on `veo-3.1-generate-001` bills a full 8s render (from $3.20 at 720p/1080p); pass `4` explicitly when a short beat will do.
- `include_audio` (default `false`): Enable audio generation
- `person_generation`: Policy for generating people — `allow_adult` or `allow_all` (some regions restrict these values)
- `audio_prompt`: Audio description
- `negative_prompt`: Things to avoid in the video
- `seed`: Random seed for reproducibility
- `image_uri`: First frame image URI for image-to-video generation
- `timeout_seconds` (default `210`): Overall deadline for the render. Common MCP hosts cap one tool call at roughly four minutes; a render that outlives the *host's* limit is cancelled by the host — the caller sees a bare "Tool execution failed" with no message, cost or operation name, and the Veo operation completes and bills anyway (a 4K render did exactly this). Under the host ceiling the server answers first: a timeout returns a structured error naming the `generation_mode`, the `attempted_cost`, the `operation_name` to reconcile in the console, and `submission`: `submitted` (the render may still complete and bill), `unknown` (the submit call itself ran out of time, so it may have reached Google) or `none` (nothing was sent). Raise it if your host allows a longer call; 4K is the slowest tier.
- `draft` (default `false`): When `true`, routes the request to `gemini-omni-1.2-flash` for a fast 720p draft instead of Veo. Iterate fast, then re-run with `draft=false` to finalize on Veo. The draft runs as long as that Veo render will (4, 6 or 8 seconds), so a 3s request drafts at 4s, with a warning. At 720p Omni 1.2 is $0.069504/s; `draft_resolution="360p"` lowers that to $0.023172/s. See [Fast drafts vs. high-fidelity](#fast-drafts-vs-high-fidelity).
- `draft_resolution`: Resolution for a `draft=true` pass. Naming one renders the draft on `gemini-omni-1.2-flash`, the model that has a resolution parameter; `360p` is about a third of the 720p price and the cheapest render this server can issue. Unset uses Omni 1.2 at 720p. A draft has no Cloud Storage destination, so on Vertex AI a 1080p or 4K draft comes back inline with a warning ([Large outputs on Vertex AI](#large-outputs-on-vertex-ai)).
- `director_notes`, `max_retakes`: Have the [virtual director](#virtual-director) check the clip against your notes and retake on a fail

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

Generate video with **`gemini-omni-1.2-flash`**, the default for Omni, drafts and animatics. It is generally available on both the Gemini API and Vertex AI under the same model ID.

- `duration_seconds`: output length from **3 to 15** seconds, default 6, rounded to a whole second. Under 3 (including 0) renders 3s and over 15 renders 15s, with a warning on the real run; a dry run reports `requested_duration_seconds` beside the length it priced. A negative or non-finite value is refused, and so are the two combinations below that cannot render past 10s.
- `resolution`: `360p`, `720p` (default), `1080p`, or `4K`; the last two include upsampling. Left unset on a continuation of a `previous_interaction_id`, it is the resolution this server recorded for that clip, so an edit or extension of a 360p draft is sent and priced at 360p.
- `aspect_ratio`: `16:9` (default) or `9:16`.
- `first_frame_uri` / `last_frame_uri`: start/end keyframes. A last frame requires a first frame; interpolation renders **3–10s**, and a `last_frame_uri` with a longer `duration_seconds` is refused rather than shortened. Use the same image twice for a loop; it is sent once, with both roles naming it.
- `reference_image_uris`: up to **6** subject/style references, with explicit role bindings. `image_uris` remains available when the model should infer each image's role.
- `reference_video_uris`: up to **3** references of up to **3s** each for new generations. **Output over 10s takes at most 1 video reference**: 2 or 3 with a longer `duration_seconds` is refused. Reference audio is ignored; video references cannot accompany editing or extension.
- `input_video_uri`: edit an uploaded clip up to **15s** long. On Vertex AI, a `gs://` source is read by the service directly (a real run also downloads it once, up to 50 MB, to check its length; a dry run does not download it), and any other source is sent inline, up to 20 MB. On the Gemini API a clip too large to send inline is uploaded through the Files API. Editing an uploaded clip is not available in the European Economic Area (Google's restriction): the service refuses, and this server cannot check it first.
- `previous_interaction_id`: continue an existing conversation on its original model and backend. An edit (this or `input_video_uri`) sends no `duration_seconds` or `aspect_ratio`, so `requested_duration_seconds` is null: the service picks the length, the real run reports the measured one, and a dry run quotes an upper bound. A real run that cannot measure its output reports `duration_seconds: null` and the bound it billed as `duration_upper_bound_seconds`, so a later upscale or extension never reads the bound as the clip's length.
- `omni_model`: `gemini-omni-1.2-flash` unless set; naming the retired `gemini-omni-1.1-flash` runs on 1.2 (see [Retired: gemini-omni-1.1-flash](#retired-gemini-omni-11-flash)). Left unset with a `previous_interaction_id`, it is the model this server recorded for that interaction. Naming a model other than the recorded one is refused, since an interaction cannot change models; the error gives the two ways out (leave `omni_model` unset, or pass the saved video as `input_video_uri` to start a new interaction). `edit_video` and `extend_video_omni` follow the same rule.
- `omni_backend`: `auto`, `vertex`, or `gemini_api`. Auto prefers a configured Gemini API key; `OMNI_BACKEND=vertex` keeps requests on Vertex.
- `output_gcs_uri`: deliver to an allowed Cloud Storage bucket on Vertex; on the Gemini API it is ignored with a warning. Vertex AI treats the URI as a prefix and writes the video to a new object under it (`gs://bucket/clips/fox.mp4` gives `gs://bucket/clips/fox.mp4/<number>.mp4`), so pass a prefix ending in `/`; a URI ending in a file name gets a warning. The response's `video_url` names the object written, and the server downloads it once (up to the 50 MB fetch cap) to measure its length and resolution. A render above 720p, an extension turn, an edit (which can run 15s), or any other output over 10s can exceed the 4 MB inline response limit the Gemini API reference gives, so the Gemini API requests it as a Google-hosted URI rather than inline, and it is downloaded here automatically. See [Large outputs on Vertex AI](#large-outputs-on-vertex-ai) for what Vertex does without `output_gcs_uri`.
- `dry_run=true` validates and quotes without rendering; `timeout_seconds` (default 210) covers the whole interaction. A render still running when it runs out keeps rendering and bills: the error names its `interaction_id`, and [get_omni_interaction](#get_omni_interaction) collects it. google-genai sends a request refused with 408 or 429 again within a few seconds. One still refused with 429 started nothing, so it is sent again after the wait the refusal asks for (else 10s, then 20s), at most twice, and only while the deadline leaves a render's worth of time; the response warns when it was, and a failure reports `quota_retries`. A create that fails with a server error is never sent again, since the service may already have started the render and a second request would bill a second one (google-genai would resend it 3 times by default). A status check that fails with a 429, a server error or a dropped connection is read again, up to 3 times in a row.
- `director_notes`, `max_retakes`: have the [virtual director](#virtual-director) check the clip against your notes and retake on a fail. Each retake renders from your inputs again; it is not a conversational edit of the take before it.

Direct continuous camera movement, reference consistency, dialogue, speaker assignments, music and sound effects in the prompt. The model generates native synchronized audio. **Audio uploads, audio references and voice editing are unsupported by the current API**, even though launch demos show voice-conditioned examples.

Output-video pricing is $12 per million tokens: **$0.023172/s at 360p, $0.069504/s at 720p, $0.104256/s at 1080p, and $0.208512/s at 4K**. Quotes include one frame of encoder allowance; input and text tokens are additional when reported. See the [API guide](https://aistudio.google.com/docs/omni) and [pricing](https://aistudio.google.com/docs/pricing#gemini-omni-1.2-flash).

#### Retired: gemini-omni-1.1-flash

`gemini-omni-1.1-flash` is retired on this server. A call that names it, or Vertex AI's spelling `gemini-omni-1.1-flash-preview`, runs on `gemini-omni-1.2-flash` with 1.2's capabilities and pricing. Its response, dry run included, reports `requested_model` and carries a warning saying the call ran on 1.2. Google applies the same redirect to that ID from **December 3, 2026**; this server applies it now, so a client that hard-codes the ID keeps working. `plan_generation` never offers 1.1, and plans a `pinned_model` of 1.1 as 1.2 with a note saying why.

An interaction created on 1.1 cannot be continued, because an interaction cannot change models. `generate_video_omni`, `edit_video`, `extend_video_omni` and `upscale_video_omni` refuse a `previous_interaction_id` recorded on 1.1 before anything is fetched, quoted or billed, dry run included. The error names the clip that render saved (from its sidecar manifest, when one is recorded) and the call to make instead: pass the clip as `input_video_uri` to `extend_video_omni` or `upscale_video_omni`, or to `generate_video_omni` to edit it on 1.2.

The old `gemini-omni-flash-preview` endpoint was switched off October 22; requests to it are refused.

#### Large outputs on Vertex AI

The 4 MB inline response limit comes from the Gemini API reference, where a large render is delivered as a Google-hosted URI. Vertex AI has no such delivery: it writes the render to Cloud Storage when the request carries `output_gcs_uri`, and otherwise returns the video bytes in the response, for which its [upscaling page](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/video/upscale-videos) states no size limit. One policy covers `generate_video_omni`, `edit_video`, `extend_video_omni` and `upscale_video_omni`, and their dry runs:

- An output is large when it is 1080p or 4K, runs over 10s, is an edit (the service picks its length, up to 15s), is an extension turn (which returns the whole assembled clip), or is an upscale of a source over 10s or of unknown length (taken as 15s).
- On Vertex AI with no `output_gcs_uri`, a large output is rendered and returned inline. The dry run and the real response both carry a warning, given before anything renders, that `output_gcs_uri` is the safer delivery for an output that large.
- `extend_video_omni` sends `output_gcs_uri` with the final turn only, so with `times` above 1 the earlier turns come back inline, and a warning says so. To send every turn to Cloud Storage, extend one turn per call (`times=1`) with `output_gcs_uri`, continuing from the `interaction_id` each call returns.
- `generate_video(draft=true)` and `generate_clip(animatic=true)` render on Omni but have no Cloud Storage destination. A large draft or animatic beat on Vertex AI gets a warning that suggests a smaller output (720p or below, or 10s or shorter) or `generate_video_omni` with `output_gcs_uri` instead.
- If a render completes but its response carries no video, the error says so: the video was requested inline and no bytes came back. It gives the size as the likely reason when the output was large, and names `output_gcs_uri` as the way to have Vertex AI write the render to Cloud Storage. The interaction completed, so the render has been billed, and rendering it again bills again.

### edit_video

Refine the same clip across multiple turns: pass `previous_interaction_id` and a `prompt` describing a background/product swap, lighting or weather change, or a new aspect ratio. Omni 1.2 accepts source clips up to **15s**. `reference_image_uris` can guide edits: up to 6 on the Gemini API and 2 on Vertex. More is refused before anything is fetched or quoted. `resolution` selects the output tier and `output_gcs_uri` supports Vertex delivery. An edit can run 15s, so on Vertex AI without `output_gcs_uri` it comes back inline with a warning ([Large outputs on Vertex AI](#large-outputs-on-vertex-ai)).

The edit runs on the model and backend that created the conversation. Leave `omni_model` unset and it uses the model this server recorded for the interaction (1.2 when there is no record). Naming a different model is refused, since an interaction cannot change models. An interaction created on the retired `gemini-omni-1.1-flash` cannot be edited: the error names its saved clip, to pass as `input_video_uri` to `generate_video_omni` ([Retired: gemini-omni-1.1-flash](#retired-gemini-omni-11-flash)). `duration_seconds` and `aspect_ratio` remain accepted for compatibility and are omitted on edit requests; put framing changes in the prompt. Real results report measured duration when available; dry runs use the model's maximum as a labelled bound. Vertex interaction context is retained up to 7 days.

### extend_video_omni

Give exactly one `previous_interaction_id` or `input_video_uri`, plus a prompt that explicitly requests continuation. Omni uses the final **10s** of the source as context. `duration_seconds` selects a **3–10s** increment as a whole number of seconds (any other value is refused, not rounded); `times` chains turns up to a **30s total on the Gemini API**, at most 30 divided by the turn length per call (**3 turns** at the default 10s, 10 at 3s); continue from the returned `interaction_id` for more. Near the ceiling a turn appends the whole seconds that still fit (a 3.03s source's third 10s turn appends 6s, to 29.03s), and a turn with less than 3s left is not run. Vertex accepts inputs of any duration and retains up to their **last 30s** before appending each segment, so each output is at most **40s**. Reference images can introduce new elements; video references are unsupported for extension. Extending an uploaded clip is not available in the European Economic Area (Google's restriction): the service refuses, and this server cannot check it first.

Left unset, `duration_seconds` is not sent and the service appends its default 10s; set, it is sent on every turn. A `previous_interaction_id` turn cannot declare its task, so the service infers extension from the prompt. If it reads the turn as an edit, a set duration makes it refuse the turn before rendering, and the error says nothing was billed for that turn and to ask for the extension explicitly in the prompt before continuing from the same id. With no duration set, such a turn is rendered and billed as an edit.

Every turn renders the assembled clip, so each turn bills the footage from prior turns again. Use `dry_run=true` to inspect the estimate and `max_cost_usd` to cap the chain. The source's length prices each turn and trims the chain at the ceiling. It comes from this server's record of a `previous_interaction_id`, or from measuring an uploaded source: a local file on dry and real runs, anything else on a real run. A `gs://` source on Vertex is read by the service directly and downloaded once (up to 50 MB) only to measure it. A dry run does not download a remote source, so it prices an unmeasured one at an upper bound. Completed turns remain in the response if a later turn fails. The final `interaction_id` continues the chain. `resolution` works as in `generate_video_omni`, `output_gcs_uri` (Vertex) applies to the final turn, and `timeout_seconds` applies to each turn. Since a turn returns the whole clip so far, on Vertex AI each turn sent without `output_gcs_uri` (every turn but the last, when one is given) comes back inline with a warning ([Large outputs on Vertex AI](#large-outputs-on-vertex-ai)).

Left unset, `omni_model` continues a `previous_interaction_id` on the model this server recorded for it and extends an `input_video_uri` on 1.2. An interaction created on the retired `gemini-omni-1.1-flash` cannot be extended: the error names its saved clip, to pass here as `input_video_uri` ([Retired: gemini-omni-1.1-flash](#retired-gemini-omni-11-flash)).

### upscale_video_omni

Upscale an approved draft with Omni 1.2 on **Vertex AI** while preserving timing and audio. Provide exactly one `input_video_uri` or a Vertex `previous_interaction_id` from a 1.2 render, plus a target `resolution` (default `1080p`):

| Source | Supported targets |
| --- | --- |
| 360p | 720p, 1080p |
| 720p | 1080p, 4K |

Sources must be at most **15s**. The source's resolution and length come from this server's record of the interaction, or from measuring the file; a record without a resolution (a Cloud Storage take from an earlier build) is settled by measuring the clip it saved, on a real run. A local or HTTP source is sent inline, which Vertex caps at **20 MB**; a `gs://` source is read by the service directly, and a real run downloads it once (up to 50 MB) only to measure it. A dry run does not download a remote source, so supply `source_resolution` for one with no record, and for a source this server cannot download. A `source_resolution` that disagrees with the record or the measured dimensions is refused. `dry_run` quotes without rendering and `max_cost_usd` caps spend. A 360p → 4K conversion needs two calls through 720p; 1080p → 4K is unsupported.

A `previous_interaction_id` recorded on the retired `gemini-omni-1.1-flash` is refused, and the error names its saved clip to pass as `input_video_uri` ([Retired: gemini-omni-1.1-flash](#retired-gemini-omni-11-flash)).

An upscale interaction cannot be continued by id: the service refuses an upscale of one ("Exactly one input video is required for the upscale task"). Given one, `upscale_video_omni` and `extend_video_omni` send the clip it saved as `input_video_uri`, and `edit_video` edits that clip through `generate_video_omni`, at its recorded resolution; the response says so. When the clip is gone, the call is refused before anything is sent, naming it. So 360p → 4K is two calls by id: upscale to 720p, then upscale that interaction to 4K.

**`output_gcs_uri` is optional.** Every 1080p or 4K target, the default included, and a 720p target from a source longer than 10s or of unknown length (counted as 15s), can exceed the 4 MB inline response limit. Without `output_gcs_uri` the upscale is rendered and returned inline, which is how Vertex AI's upscaling page says it delivers when no Cloud Storage destination is given. The dry run and the real response carry a warning that `output_gcs_uri` is the safer delivery for an output that large; pass one to have Vertex AI write the upscale to Cloud Storage ([Large outputs on Vertex AI](#large-outputs-on-vertex-ai)).

Standalone upscaling bills zero input tokens. Output rates are 2,896 tokens/s for 360p → 720p or 720p → 1080p, 5,792 for 360p → 1080p, and 11,584 for 720p → 4K, at $12 per million tokens. See [standalone upscaling](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/video/upscale-videos). A quote prices the requested pair at the source's length (15s when unknown); a measured source is priced as measured, and a length that was not measured gets one frame of encoder allowance. `max_cost_usd` refuses a real run over the cap, and a dry run returns its quote with `would_be_refused`. A real run whose output comes back inline is metered at the length measured from the rendered file and at the resolution pair that actually rendered; if that pair has no published rate, the requested pair is priced and the cost detail says so. An output delivered to Cloud Storage is downloaded once (up to the 50 MB fetch cap) and measured the same way; one that cannot be read is priced as an estimate at the source's length, or 15s when that is unknown.

### get_omni_interaction

Check on an Omni render by its `interaction_id`, and save it once it has finished. For a render whose tool call ended first: a timeout, whose error names the `interaction_id`, or an MCP host that cut the call off. The service keeps rendering and bills it either way; this tool only reads, so it never bills.

- `interaction_id` (required): the id the timed-out call reported
- `omni_backend`: `gemini_api` or `vertex`, where the interaction was created, since an id resolves only there. Defaults to the backend this server recorded for it (a call that times out records it), else `OMNI_BACKEND`
- `wait_seconds`: how long to wait for a render still in progress, up to 1800 (default 0, one read). Many MCP hosts end a tool call near 240s

The response carries the service's `status`. While the render is in progress nothing is saved. Once it has completed, the video is saved (or, delivered to Cloud Storage, left there and downloaded once to measure), and the response carries `video_url`, the measured `duration_seconds` and `resolution`, and `cost` at those measurements. An upscale is priced at its recorded source resolution; an interaction this server has no record of is priced as a render. The interaction is recorded like any render, so `edit_video`, `extend_video_omni` and `upscale_video_omni` can continue from its id. One already saved is returned from its record without reading the service.

### loop_extend

Convenience wrapper that extends a Veo-generated video multiple times in one call. Each Veo extension adds ~7s, and can be chained up to 20 times. This is the **Veo** extension path; `extend_video_omni` is the Omni one, and they are not interchangeable.

**Parameters:**
- `video_uri` (required): The Veo-generated video to extend
- `prompt`: What the video continues with (default: "continue the action")
- `times`: Number of ~7s extensions to apply, 1-20 (default: 1)
- `model`: Veo model to use (default `veo-3.1-generate-001`)
- `aspect_ratio`: `16:9` (default) or `9:16` — must match the source video
- `include_audio`: Generate audio on the extended sections (default `true`)
- `output_gcs_uri`: GCS URI for output (required, or set `VIDEO_GCS_BUCKET`)

**Notes:**
- Veo 3.1 / Veo 3.1 Fast, on Vertex AI
- 720p

### generate_clip

Generate a **multi-beat short clip** — the building block for a reel or short. Each beat is rendered in order, and the tool returns an ordered manifest a cutting MCP (e.g. vfx-mcp) can splice into a finished clip.

This is the highest-leverage tool in the server: one call produces a whole sequence instead of N round-trips.

**Parameters:**
- `beats` (required): Ordered list of beat specs. Each accepts `{prompt, duration_seconds?, seed?, first_frame_uri?, negative_prompt?, audio_prompt?}`
- `aspect_ratio`: Default `9:16` for vertical social clips
- `model`: VEO model applied to every beat (default `veo-3.1-fast-generate-001`)
- `include_audio` (default `true`): Audio per Veo beat (Vertex only). An animatic's Omni audio has no switch, so passing it with `animatic` is ignored with a warning
- `beats` are capped at 20 per call — each is a billed Veo render, and `add_bridges` nearly doubles that. Split longer sequences into several clips
- `add_bridges`: Generate a transition between consecutive beats using the last frame of beat N and the first frame of beat N+1. Requires local (`file://`) beat outputs
- `animatic`: Render every beat with `gemini-omni-1.2-flash` (fast 720p) for a **storyboard preview of the whole reel** before committing to full Veo renders. Each beat runs as long as the Veo delivery will render it (4, 6 or 8 seconds), so a 3s beat previews at 4s, with a warning. With `add_bridges`, each bridge renders 4s on Omni 1.2 between the beats' frames, at the animatic's resolution, and is quoted and measured like a beat. Veo-only controls (`seed`, `negative_prompt`) are ignored in this mode
- `animatic_resolution`: Resolution for the animatic pass. Naming one renders it on `gemini-omni-1.2-flash`; `360p` is a third of the 720p price, which is what turns the preview from roughly the cost of the delivery render into a real saving on a long reel. An animatic has no Cloud Storage destination, so on Vertex AI a 1080p or 4K animatic, or a beat over 10s, comes back inline with a warning, given before any beat renders ([Large outputs on Vertex AI](#large-outputs-on-vertex-ai))
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
- `model`: Veo model — default `veo-3.1-fast-generate-001`
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

- **`gemini-omni-flash-preview` (switched off)** — 720p, 24fps, conversational multi-turn editing. No seeds, no negative prompts, no first/last-frame control. **Switched off on 2026-10-22**; pinning `omni_model="gemini-omni-flash-preview"` is now refused before any quote or call. The draft and animatic paths no longer route here.
- **`gemini-omni-1.2-flash` (fast, and far more controllable)** — everything above plus 360p/1080p/4K output, first/last-frame interpolation, video references, native audio, scene extension and, on Vertex AI, standalone upscaling. Still no seeds and no negative prompts. The default for `generate_video_omni`, `extend_video_omni`, `draft` and `animatic`, and the only model `upscale_video_omni` runs. A continuation (`edit_video`, or any `previous_interaction_id`) stays on the model that created the interaction.
- **Veo 3.1 / Fast (high-fidelity, Vertex AI only)** — up to 1080p/4K, seeds for reproducibility, first/last-frame control, reference images, and extension. The path for final renders. Reached via `generate_video` (default) and `loop_extend`.

Typical workflows (`plan_generation` recommends these as ordered `workflow` steps, with both figures quoted):
- **cheap draft → finalize**: run `generate_video_omni(omni_model="gemini-omni-1.2-flash", resolution="360p")` for a preview at roughly a third of the 720p price, then re-run at `720p`/`1080p`/`4K` — or on Veo — once the shot is right. The equivalent shortcuts are `generate_video(draft=true, draft_resolution="360p")` and `generate_clip(animatic=true, animatic_resolution="360p")`.
- **draft → upscale** (Vertex AI): approve a 360p or 720p render, then pass it to `upscale_video_omni` for a 1080p copy (or 4K from 720p) that keeps its frames, timing and audio. The copy comes back inline unless you pass `output_gcs_uri`, the safer delivery for it. `plan_generation` plans this for an intent like `a 6 second video of a fox in snow, then upscale it to 4K`.
- **two stills → interpolation**: give `generate_video_omni` a `first_frame_uri` and a `last_frame_uri` on 1.2 and it renders the motion between them — the same job `generate_transition` does on Veo, and the planner now offers both. The same URI for both loops the clip.
- **draft → finalize**: run `generate_video(draft=true)` to preview quickly on omni, then re-run the same prompt with `draft=false` to render the final on Veo.
- **animatic → final**: run `generate_clip(animatic=true)` to render each beat via `gemini-omni-1.2-flash` as a fast storyboard preview of the whole reel, then re-run with `animatic=false` (the default) to commit to full Veo renders.

> **Note:** `generate_clip`'s new `animatic` parameter (default `false`) renders each beat through `gemini-omni-1.2-flash` instead of Veo, so you can preview an entire reel quickly before committing to full Veo renders (Omni 1.2 costs less per second than either Veo tier).

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

**Note**: On the Gemini API, video runs on Gemini Omni. Google switched off Veo 3.1 there (all tiers, including Lite) on **October 22, 2026**, so Veo needs Vertex AI, which also adds GCS output and Vertex-only features such as controllable audio.


## Contributing

### Development Setup

```bash
uv sync
```

### Running Tests

```bash
uv run pytest
```

This is hermetic and free: the paid live tests below skip unless you opt in.

### Live tests

`tests/test_live_omni_1_2.py` checks, against the real API, the questions about `gemini-omni-1.2-flash` that only a real render can settle. Every test in it spends money, so each one skips unless its credentials are set **and** `OMNI_LIVE_SPEND=1`. L3 and L3b, the 4K ones, also need `OMNI_LIVE_EXPENSIVE=1`. Credentials alone never run a paid test.

Run from the repo root; `-s` prints the request each test sent and what it measured:

```bash
# Vertex AI: L1, L2, L4-L7, L9-L13 (Vertex)
GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT=<project> \
OMNI_LIVE_SPEND=1 uv run pytest tests/test_live_omni_1_2.py -v -s

# Gemini API: L2, L4, L5, L7, L9-L13 (Gemini API)
GEMINI_API_KEY=<key> OMNI_LIVE_SPEND=1 \
uv run pytest tests/test_live_omni_1_2.py -v -s

# Both backends, plus L3 and L3b, and L8 with a bucket
GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT=<project> \
GEMINI_API_KEY=<key> OMNI_LIVE_SPEND=1 OMNI_LIVE_EXPENSIVE=1 \
OMNI_LIVE_GCS_BUCKET=<bucket> \
uv run pytest tests/test_live_omni_1_2.py -v -s
```

Vertex credentials are whatever the server takes: Application Default Credentials (for example `GOOGLE_APPLICATION_CREDENTIALS`) or `GOOGLE_SERVICE_ACCOUNT_JSON`. `GOOGLE_API_KEY` works in place of `GEMINI_API_KEY`. Each test runs the real server in-process on a scratch `DATA_FOLDER`.

| Test | What it settles | Runs on | Cost, at the published 1.2 rates |
|---|---|---|---|
| L1 | Upscaling an interaction by `previous_interaction_id`: a 3s 360p render, then an upscale to 720p | Vertex AI | about $0.18 |
| L1b | Upscaling an upscale's interaction: the same, then that upscale to 1080p, sent the clip it saved | Vertex AI | about $0.28 |
| L2 | Whether an extension turn that continues an interaction honours `duration_seconds`: a 3s 360p render, then two 3s turns, which bill the assembled 6s and 9s | each backend that has credentials | about $0.42 per backend |
| L3 | Whether a large output arrives inline on Vertex AI with no `output_gcs_uri`: a 10s 720p render, then an upscale to 4K | Vertex AI, with `OMNI_LIVE_EXPENSIVE=1` | about $2.10 |
| L3b | The same for a fresh render, since one switch covers all four Omni tools: a 3s 4K render | Vertex AI, with `OMNI_LIVE_EXPENSIVE=1` | about $0.63 |
| L4 | A call naming `gemini-omni-1.1-flash` sends 1.2, reports 1.2, and says why: a 3s 360p render | the backend the server's Omni default picks | about $0.07 |
| L5 | Extending an uploaded clip (`input_video_uri`) by 3s: the turn measures the source plus 3s | each backend | about $0.21 per backend |
| L6 | Upscaling an uploaded file through `upscale_video_omni` | Vertex AI | about $0.18 |
| L7 | A 12s render in one call: URI delivery on the Gemini API, inline on Vertex AI | each backend | about $0.28 per backend |
| L8 | Delivery to `output_gcs_uri`, then an upscale read from that `gs://` object | Vertex AI, with `OMNI_LIVE_GCS_BUCKET` | about $0.18 |
| L9a, L9b | A first frame, and a first/last-frame pair: the render's ends must resemble the images given for them | each backend | about $0.07 each per backend |
| L9c | A loop: one image as first and last frame is sent once, and both ends of the render resemble it. If it fails, send the image twice again | each backend | about $0.07 per backend |
| L10a, L10b | Two reference images, and a 2s reference clip, bound to their roles | each backend | about $0.07 each per backend |
| L11a, L11b | A night edit of an interaction, and of an uploaded clip: no duration or aspect ratio sent, and the result is darker | each backend | about $0.14 each per backend (up to $0.42 if the service makes the edit 15s) |
| L12a, L12b | `generate_video(draft=true)` and `generate_clip(animatic=true)` render on 1.2; the animatic measures portrait at 9:16 | the backend the server's Omni default picks | about $0.07 each |
| L12c | An animatic with `add_bridges`: the bridge is a first/last-frame render on 1.2 between the beats' frames, measured like a beat | the backend the server's Omni default picks | about $0.28 |
| L13 | A render whose call times out after the create is collected with `get_omni_interaction`, with nothing created | each backend | about $0.07 per backend |

Everything, with both backends and a bucket: about $7.24. L5-L13 alone: about $3.16. L2 costs about $0.91 per backend instead if the service ignores the duration, since `max_cost_usd` checks the estimate, not the bill.

The module docstring says what each outcome means for the code. Two of them decide a constant in `src/omni.py`: if L2 fails on either backend because the duration was refused or ignored, set `OMNI_CONVERSATIONAL_EXTENSION_SENDS_DURATION = False`; if L3 or L3b fails with a delivery error, set `OMNI_REFUSE_LARGE_INLINE_ON_VERTEX = True`, and every Omni tool then refuses a large Vertex AI output without `output_gcs_uri` before rendering, dry run included.

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

