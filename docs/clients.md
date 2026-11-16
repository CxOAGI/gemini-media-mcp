# Client setup

These setup features are unreleased. Run `uv run gemini-media-mcp` from this
checkout until the next Python release; afterwards use `uvx --python 3.13
gemini-media-mcp`. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
first. A released wizard pins generated client commands to its own installed
version. Docker placeholder versions older than the bundled public pin and
development checkouts use the public fallback in
`src/client_release.json` (currently 0.1.14); unreleased versions are not fetched
from PyPI. The Claude and Gemini native plugins have their own explicit pins,
checked against that fallback in CI.

## One wizard, several clients

```bash
uv run gemini-media-mcp setup --client codex --write
```

| `--client` | Destination / behavior |
|---|---|
| `claude-code` | Installs `gemini-media@cxoagi` in user scope, with non-secret settings supplied at install time and secrets sent over stdin. Tested with Claude Code 2.1.325; requires the marketplace files published on GitHub. |
| `claude-desktop` | macOS Application Support or Windows `%APPDATA%` Claude config. Other platforms require `--config-path`. This remains the default client. |
| `codex` | `$CODEX_HOME/config.toml` or `~/.codex/config.toml`; preserves TOML comments and unrelated tables. |
| `cursor` | `~/.cursor/mcp.json` |
| `copilot` | `$COPILOT_HOME/mcp-config.json` or `~/.copilot/mcp-config.json`; native Copilot in VS Code and CLI. |
| `vscode` | Editor-managed `servers` format; requires `--config-path` to install. Secrets and local paths become inputs; `uvx` resolves through each machine's PATH. This older Local harness format does not forward password inputs to Copilot. |
| `gemini-cli` | `$GEMINI_CLI_HOME/.gemini/settings.json`, or `~/.gemini/settings.json` when unset or empty; uses a 30-minute tool timeout. |
| `cline` | CLI `~/.cline/data/settings/cline_mcp_settings.json`, or `CLINE_MCP_SETTINGS_PATH`. For the IDE, use the path opened by Cline's Configure MCP Servers action. |

Use one installation method per client to avoid duplicate tools. Restart the
client or reload its MCP configuration after writing. First launch may take a
minute while uv downloads Python and dependencies. Codex gets a 120-second
startup timeout and a 30-minute tool timeout. Keep the client's approval prompts
enabled: actual image/video generation can incur Google charges.

Current format references: [Codex](https://learn.chatgpt.com/docs/extend/mcp?surface=cli),
[Cursor](https://cursor.com/docs/mcp),
[Copilot and VS Code](https://code.visualstudio.com/docs/agents/reference/mcp-configuration),
[Gemini CLI](https://geminicli.com/docs/tools/mcp-server/),
[Cline](https://docs.cline.bot/mcp/mcp-overview).

## Credentials and defaults

Gemini API mode only needs an API key and an output folder. The wizard reuses
`GEMINI_API_KEY` if set and announces that it will be saved; otherwise it prompts
in your terminal with hidden input.
Do not paste credentials into a chat. Gemini CLI's own account login does not
automatically supply credentials to this server.

Vertex mode needs a billing-enabled Google Cloud project, the Vertex AI API,
appropriate IAM permissions, and credentials. The simplest local path is:

```bash
gcloud auth application-default login
uv run gemini-media-mcp setup --client codex --mode vertex --adc --project-id YOUR_PROJECT --write
```

`gcloud auth login` alone is not ADC. The wizard does not enable billing, create
cloud resources, change IAM, or run browser login for you. Existing
`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and credential file settings can
be reused. Interactive setup shows the inherited project and location as prompt
defaults so you can confirm or change them. It asks before reusing an inherited service-account
credential and explains whether the private key or a file path will be saved.
Scripted Vertex setup requires an explicit `--adc`, `--sa-path`, or `--sa-json`;
empty credential arguments and non-object inline JSON (including `null`) fail
instead of falling back to an ambient identity, even with `--skip-validation`.
`--adc` deliberately chooses ADC discovery instead of an inherited
service-account override. Managed runtime ADC and existing impersonation-based
ADC also use this path. See [Google's ADC setup](https://cloud.google.com/docs/authentication/provide-credentials-adc).

For a service-account file, use `--sa-path /absolute/path/key.json` instead of
`--adc`. Inline JSON remains available through `--sa-json` for compatibility,
but shell arguments may be recorded; prefer a file. Claude Code's adapter
supports ADC and files, not inline JSON; unsupported inline arguments fail before
credential prompts, including for previews and exports. Vertex defaults to `us-central1` and
asks about GCS only on that branch. A bucket is optional for onboarding; some
video workflows require one. Omni is explicitly set to `vertex` or `gemini_api`
so an unrelated shell API key cannot silently change the billing route. The wizard
and native plugins set both `GOOGLE_GENAI_USE_VERTEXAI` and the SDK's older
`GOOGLE_GENAI_USE_ENTERPRISE` switch to the selected mode.

The output folder becomes an absolute path outside plugin caches. Local media
inputs must also be under `DATA_FOLDER`. Keep service-account keys elsewhere.

## Preview, export, and scripts

Destination checks and parsing of an existing JSON/TOML config happen before
credential entry. With `--write`, a missing
destination or missing Claude Code executable fails immediately. Without
`--write`, an unavailable install destination (including the default Claude
Desktop target on Linux) offers an export path before collecting credentials.
Pressing Enter accepts `~/gemini-media-config.json` for that export.

Without `--write`, interactive setup otherwise offers to install. Scripted CLI
setup requires `--write`, `--output PATH`, or `--preview`. The older command
`setup --non-interactive --mode gemini --api-key ...` now fails with migration
instructions: add `--output PATH` for a usable private config, or `--preview`
for redacted output. Credentials are never printed implicitly.
`--output` takes precedence over installing. Keep exports and `.bak` files out of Git.

```bash
# GEMINI_API_KEY is already in the environment; no key in shell arguments
uv run gemini-media-mcp setup --client cursor --mode gemini --non-interactive --write

# Export for manual merging, including JSONC files with comments
uv run gemini-media-mcp setup --client cursor --mode gemini --output ~/gemini-media-config.json

# Explicit redacted preview for scripts
uv run gemini-media-mcp setup --client cursor --mode gemini --non-interactive --preview

# Explicit destination for an older VS Code Local configuration
uv run gemini-media-mcp setup --client vscode --config-path /path/to/project/.vscode/mcp.json --write
```

JSON and TOML writes are atomic, preserve other servers/settings, and create an
owner-only, byte-identical backup, including any UTF-8 BOM and original line
endings. Symlink destinations and backups are refused. Malformed JSON or JSONC is left untouched: export and merge
using the client's editor. POSIX files use mode 0600; on Windows use a profile
directory protected by your account's ACLs. VS Code stores API keys using its
password-input facility and prompts for absolute media/credential paths on each
machine; the wizard's local paths are not embedded in the workspace file.
The other JSON/TOML clients store credentials in their private
config files. Claude Code uses its native sensitive-setting storage.

Claude Code previews and exports contain native plugin option names, matching
what the wizard supplies to `claude plugin configure --values-stdin`. They are
not generic MCP config files. An exported options file can be passed to that
command after installing the plugin.

**Clearing a saved Claude key:** Claude Code ignores empty sensitive values.
Use the wizard with `--client claude-code --mode vertex --adc --write` to switch
an existing Gemini installation to Vertex only. If an API key is stored, the
wizard uses Claude's native uninstall/reinstall with `--keep-data`, restores the
selected options, and verifies that the key is no longer configured. Generated
media and plugin data remain intact. If another scope also has this plugin,
setup stops before changing it because the scopes share saved credentials.
Resolve those installations first. An exported empty key alone cannot clear it.

If install, configure, or verification fails, setup attempts to disable the
user-scope plugin and reports whether that succeeded. Rerun setup to finish and
enable it. A timeout produces recovery instructions rather than a traceback.

`--skip-validation` supports offline setup. Otherwise the wizard calls
`models.list()`; failure can mean missing model-list permission or network
access, not necessarily invalid generation credentials. Interactive setup lets
you explicitly choose to continue; the default is to stop on a failed check.
Missing ADC reports the login command instead of suggesting the credentials
may be valid. No generation is performed to validate credentials.

## Diagnose and verify

Run `doctor` in the same environment that starts the server. It checks Python,
existing output-directory writability, credential configuration, and the same
Omni backend rules used at server startup. It does not create directories: an
explicit missing output path is an error; an absent default `./data` is marked
unchecked. Correct a typo or create the intended directory explicitly.
Missing `uvx` is
informational for pip and Docker installations. Inline credentials and their
`project_id` follow the server's precedence over a credential file. `doctor --live`
also authenticates with model listing. It does not load secrets back out of
client configuration, and does not prove access to every model or sufficient
generation quota.

In the client, confirm tools appear and ask: “Plan an image of a ceramic mug;
show the estimate and do not generate it.” Expect `plan_generation` to return
routes and costs without producing a file. For a real render, request a quote
and approve the cost first. GUI connection and paid rendering checks remain
separate from the repository's offline tests.

## Gemini CLI extension

After these files are published to GitHub:

```bash
gemini extensions install https://github.com/CxOAGI/gemini-media-mcp
```

For local development, use `gemini extensions link /absolute/path/to/checkout`.
The extension asks for a Gemini API key and an absolute output folder. The key
is declared sensitive for Gemini CLI's keychain. Use a full path, not `~`, for
the folder. Restart Gemini CLI and check `/mcp`. Change values with
`gemini extensions config gemini-media`.

The extension provides Gemini API images and Omni video. For Vertex, use the
wizard's `gemini-cli` target instead; do not keep both enabled. The extension
version is independent of the Python package and currently pins 0.1.14.
The [extension schema and settings behavior](https://geminicli.com/docs/extensions/reference/)
are documented upstream.

Native installation was verified with Gemini CLI **0.68.0** on Linux ARM64
on 2026-11-13, using an isolated `GEMINI_CLI_HOME` and a local package containing
extension version 1.0.0, its README and license. The normal installer prompted for the two
settings, masked the dummy API key, and enabled the extension. A fresh
`gemini mcp list` reported **Connected** using the saved settings, including an
output path with spaces. Reconfiguration and uninstall also passed; no media
was generated. The test used the CLI's encrypted-file credential backend,
not a desktop OS keychain. The disposable CLI and profile were removed.
Version 1.0.1 additionally sets `GOOGLE_GENAI_USE_ENTERPRISE` explicitly; the
package smoke tests verify its launch with an inherited conflicting value.

To repeat the installation check with Gemini CLI installed, run these commands
in an interactive terminal, using an isolated `GEMINI_CLI_HOME` if you want to
keep your normal configuration separate:

```bash
gemini extensions install /absolute/path/to/extension
gemini extensions list
gemini mcp list
gemini extensions uninstall gemini-media
```

Supply a dummy key and an absolute temporary output folder for a connection-only
test. A conversational planning check requires Gemini CLI's own authenticated
session; actual generation additionally requires a valid extension API key.
Installation from the public GitHub URL remains a post-publication check.
