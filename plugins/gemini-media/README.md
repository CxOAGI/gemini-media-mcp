# Gemini Media for Claude Code

An MCP plugin for planning, generating, composing, and editing images and video
with Gemini, Veo, and Omni. Generation tools support `dry_run: true` for a free
cost estimate. Real generation uses your Google credentials and incurs API costs.

Distribution is through **CxOAGI's GitHub-hosted marketplace**. We are not
submitting this plugin to Anthropic's official catalog. See
[distribution eligibility](https://github.com/CxOAGI/gemini-media-mcp/blob/main/docs/distribution.md#anthropic-directory-eligibility).

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first and
make sure `uvx --version` works in the environment that starts Claude Code.
These instructions are validated with Claude Code 2.1.325.

Inside Claude Code:

```text
/plugin marketplace add https://github.com/CxOAGI/gemini-media-mcp
/plugin install gemini-media@cxoagi
```

Choose **User** scope, then fill in just two values:

- **Gemini API key:** get it from [Google AI Studio](https://aistudio.google.com/apikey).
  Enter the key in Claude's configuration form, not the chat.
- **Media output folder:** an absolute, writable path such as
  `/Users/you/gemini-media` or `C:/Users/you/gemini-media`.

Leave the other options at their defaults. This enables Gemini images and Omni
video. Adding the marketplace also makes the plugin appear in `/plugin` →
**Discover**. No repository clone or manual MCP JSON editing is needed.

For Vertex AI instead, enable **Use Vertex AI**, enter your Google Cloud project and service
account file (or use Application Default Credentials), and leave the API key
blank unless you also want Omni to use the Gemini API. Select the `vertex` Omni
backend to keep Omni usage in your Vertex project.

Change these options with `/plugin configure gemini-media@cxoagi`. The API key
is marked sensitive for Claude's credential storage. Settings in this form
override their corresponding environment variables. An empty sensitive value
does not clear a saved API key. To switch an existing installation to Vertex
only, use the [setup wizard](https://github.com/CxOAGI/gemini-media-mcp/blob/main/docs/clients.md#preview-export-and-scripts).
It resets the user-scope plugin through Claude's native uninstall/reinstall with
`--keep-data`, restores the chosen options, and verifies that the old key is gone.
It stops before changing credentials shared with another installed scope.

Open `/mcp` to check the server connection. First startup downloads Python 3.13
if needed and the pinned `gemini-media-mcp==0.1.14` package through `uvx`.
Generated media stays in your chosen folder across plugin updates and removal.

See the [full setup, troubleshooting, and update instructions](https://github.com/CxOAGI/gemini-media-mcp#claude-code-plugin).

## Maintainers

The plugin wrapper is versioned independently of the Python package. To upgrade
the server, update the shared public fallback in `src/client_release.json` at
the repository root, both native plugin pins and their documented versions.
Change this plugin's package pin in `.mcp.json`, bump the version in
`.claude-plugin/plugin.json`, update the documented pin, and run the plugin
validation and smoke checks described in the repository's Contributing section.
Only files inside this plugin directory are installed; the server is fetched
from PyPI, so local changes to the repository's `src/` do not change this plugin.

License: [Apache-2.0](LICENSE).
