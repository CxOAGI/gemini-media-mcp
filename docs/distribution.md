# Distribution and listings

This checkout prepares integrations; it does not publish packages, submit
directory applications, or claim that a new listing is live.

For Claude Code, the chosen distribution route is **our own CxOAGI marketplace**.
Users add its GitHub URL and install `gemini-media@cxoagi`. Do not submit this
plugin to Anthropic's official catalog as part of this release.

| Channel | Prepared here | Remaining publication step |
|---|---|---|
| Claude Code | Marketplace, plugin manifest, native configuration and smoke test | Publish repository files; users add the repository URL. |
| Gemini CLI | Root `gemini-extension.json`; native local install, settings, connection and uninstall verified | Publish files and verify installation from the repository URL. |
| Official MCP Registry | Root `server.json` and README ownership marker | Release matching PyPI metadata, authenticate the CxOAGI namespace and publish. |
| Glama | Root `glama.json` names `JamesAntisdel` as maintainer | Merge/push and let Glama ingest; verify the existing repository entry under that account. |
| Cline Marketplace | [Install instructions](../llms-install.md) and submission text below | Perform the Cline acceptance test, supply the required logo, submit the issue. |
| Docker MCP Catalog | Existing Dockerfile and catalog submission recipe below | Build/test catalog configuration and submit a reviewed PR upstream. |
| Anthropic directory | Excluded from this release's distribution plan; policy reviewed | No submission planned. |

## Release order

1. Run `uv run ruff check src tests`, `uv run pytest -q`, and `uv build`.
   Run the opt-in pinned-package smoke tests documented in the README.
2. Publish these repository files. The native installers clone the public repo,
   so an unpushed local manifest cannot be installed from its GitHub URL.
3. Choose and tag the next Python release. `server.json` currently reserves
   **0.1.15**; update both version fields if the actual release differs. PyPI
   0.1.14 does not contain this checkout's new onboarding commands or
   the registry ownership marker. Do not publish registry metadata against it.
4. Publish the wheel/sdist through the existing release workflow. Verify that
   the PyPI description includes `mcp-name: io.github.CxOAGI/gemini-media-mcp`.
5. Verify the new wheel's setup and doctor commands. Released setup
   wizards automatically pin their own installed release. When upgrading the
   native plugins and development fallback, update `src/client_release.json`,
   `plugins/gemini-media/.mcp.json`, `gemini-extension.json`, and the documented
   pins together; `tests/test_client_config.py` checks their consistency.
   Bump plugin/extension versions when their bundled configuration changes.
6. Authenticate and publish the official registry entry using
   [mcp-publisher](https://modelcontextprotocol.io/registry/quickstart).
   GitHub authentication must authorize the **CxOAGI** organization namespace;
   Glama's individual-maintainer identity is a different setting.

The [registry PyPI ownership rule](https://modelcontextprotocol.io/registry/package-types)
checks the released package description. Local schema validation alone does
not verify ownership or publish anything. The entry intentionally gives a
simple Gemini API install; its documentation links to the richer Vertex wizard.

## Glama

`glama.json` follows [Glama's schema](https://glama.ai/mcp/schemas/server.json).
The existing repository entry is
[CxOAGI/gemini-media-mcp](https://glama.ai/mcp/servers/CxOAGI/gemini-media-mcp).
The maintainer is the GitHub user `JamesAntisdel`, not the organization name.
Repository publication and Glama account verification are still required for
the remote entry to reflect this local change.

## Directory description

**Name:** Gemini Media

**Repository:** https://github.com/CxOAGI/gemini-media-mcp

**License:** Apache-2.0

Plan, generate, compose and edit images and video with Gemini, Veo and Omni.
Compare routes and costs before generation, create storyboards and multi-beat
clips, and optionally have a director check each take against your notes.
Use a Gemini API key for images and Omni video, or a Vertex AI project for Veo
and additional workflows. Generated files stay in a user-selected directory.

## Cline submission draft

Use the current [MCP Marketplace submission template](https://github.com/cline/mcp-marketplace/issues/new/choose).

**Title:** Add Gemini Media MCP

**GitHub URL:** https://github.com/CxOAGI/gemini-media-mcp

**Benefit:** Image and video planning, generation, editing and composition from
Cline, with cost estimates before paid calls. Supports Gemini API and Vertex
AI, and writes media to an explicitly configured local folder.

**Install instructions:** `llms-install.md` in the repository root.

Before submitting: attach a 400×400 PNG logo and complete the marketplace's
testing confirmation. The repository tests exercise configuration and MCP
planning, but that is not a claim of a tested Cline GUI install. Verify tools
appear in Cline, run a no-generation plan, then an explicitly approved render
and confirm the file exists. Do not include credentials in the issue.

## Docker Catalog submission recipe

Follow the [Docker MCP Registry contributor guide](https://github.com/docker/mcp-registry/blob/main/CONTRIBUTING.md)
in a fork of that repository. Use `task wizard` with this repository URL and
choose a Docker-built local server using the root Dockerfile.

Use title **Gemini Media**, the description above, and Apache-2.0 source.
Configure `GEMINI_API_KEY` as a secret, `GOOGLE_GENAI_USE_VERTEXAI=false`,
`OMNI_BACKEND=gemini_api`, and `DATA_FOLDER=/data`. Mount a user-selected
persistent host directory at `/data`; without a volume, generated media is
lost with the container. Do not publish a host's credential path as a default.
Keep the first catalog entry focused on Gemini API; document Vertex separately.

Run the upstream generator/validator and tool-discovery checks against the
chosen immutable source commit. Verify output persists across container
replacement, and that local input access is limited to `/data`. Submit the
generated `servers/gemini-media-mcp/server.yaml` and supporting files in a PR
titled **Add Gemini Media MCP**. Catalog approval and any test-credential
exchange use Docker's official review process, not repository metadata alone.

## Anthropic directory eligibility

Checked against the [Software Directory Policy](https://support.claude.com/en/articles/13145358-anthropic-software-directory-policy)
on 2026-11-13. Section 4.B excludes AI video/audio generation and general-purpose
image generation from Anthropic's directories unless Anthropic expressly permits
them in writing. Its narrow image exception covers exclusively design-focused
workflows; our standalone image tools and video capabilities do not qualify.

Do not submit the full Gemini Media plugin as directory-eligible. Seek a written
exception or independently evaluate a separate product whose implemented
capabilities actually fit the policy. Changing descriptions, hiding tools, or
calling unrestricted generation a design workflow does not change eligibility.

Anthropic's [distribution documentation](https://code.claude.com/docs/en/plugins/publish)
separately supports developer-hosted marketplaces. Our GitHub URL installation
uses that mechanism. It is not Anthropic directory inclusion or endorsement;
normal usage policies and client/organization restrictions still apply.
`claude plugin validate --strict` validates packaging and does not confer policy
approval. Nothing has been submitted to Anthropic.
