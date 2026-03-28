# Photo Renamer (Vision Model)

Uses a local vision model to describe a photo and generate a descriptive, unique filename.

## Features

- Uses Ollama `@vision` (or other providers) to "see" what's in your photo.
- Generates a 4-6 word slug: `2026-03-golden-gate-bridge-fog-sunset.jpg`.
- Appends a short hash suffix to avoid name collisions.
- Multi-provider support (`--provider`, `--model`).
- Standard local-first tracking and LLM run logging.

## Usage

```bash
# Rename using local Ollama vision model
uv run rename-photo photo.jpg

# Rename all photos in a directory
uv run rename-photo ./photos/

# Use a specific provider/model
uv run rename-photo photo.jpg --provider anthropic --model claude-3-5-sonnet

# Dry run (see proposed names without renaming)
uv run rename-photo photo.jpg --dry-run
```

## Development

```bash
# Run tests (using MockProvider)
uv run pytest
```

## Part of the Photo Pipeline
`photo-renamer` → `photo-metadata-scrubber` → `photo-scaler` → `unsplash-uploader`
