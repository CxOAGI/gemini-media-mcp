# syntax=docker/dockerfile:1

# Build stage
FROM ghcr.io/astral-sh/uv:python3.13-alpine AS builder

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Set fallback version for builds without .git
ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}

# Install dependencies first (cached layer)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project --no-dev

# Copy source and install the project
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ src/

# Install the project itself (non-editable)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# Runtime stage
FROM python:3.13-alpine

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Indicate running in container for path handling
ENV RUNNING_IN_CONTAINER=true

# Expose port for HTTP/SSE transport
EXPOSE 8000

# Run MCP server (default: stdio)
ENTRYPOINT ["gemini-media-mcp"]
CMD ["stdio"]

# Usage:
# stdio:           docker run <image>
# sse:             docker run -p 8000:8000 <image> sse
# streamable-http: docker run -p 8000:8000 <image> streamable-http
