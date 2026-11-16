# Install Gemini Media MCP in Cline

Install uv, then follow [the client setup guide](docs/clients.md). The new
multi-client setup flags are currently available from this source checkout:

```bash
uv run gemini-media-mcp setup --client cline --write
```

For the Cline IDE extension, first open MCP Servers → Configure → Configure MCP
Servers and pass that file's absolute path with `--config-path`. The default
destination is Cline CLI's settings, not an inferred IDE extension directory.

Choose Gemini API for the shortest setup, or Vertex AI with ADC and a Google
Cloud project. Run the wizard directly in the user's terminal so API keys never
enter a chat. It prints a redacted preview and writes private config/backups.
Set a persistent output folder; local input media must also be inside it.

After restarting Cline, confirm `gemini-media` tools appear. Ask for an image
plan and estimate without generation. Only render after the user approves the
cost; confirm the returned media file exists. Leave paid tools out of automatic
approval lists. If tools time out, retain the configured 1800-second timeout.

For a published-runtime manual install, configure `uvx` with arguments
`["--python", "3.13", "--from", "gemini-media-mcp==0.1.14", "gemini-media-mcp", "stdio"]`
under `mcpServers.gemini-media`, and supply `GEMINI_API_KEY`, absolute
`DATA_FOLDER`, `GOOGLE_GENAI_USE_VERTEXAI=false`, and `OMNI_BACKEND=gemini_api`
in that server's environment. Preserve other servers and protect the config
file because it contains the API key.
