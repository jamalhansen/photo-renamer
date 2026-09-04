# Photo Renamer (Vision Model)

Uses a local vision model to describe a photo and generate a descriptive, unique filename.

## Features

- Uses Ollama `@vision` (or other providers) to "see" what's in your photo.
- Generates a 4-6 word slug: `2026-03-golden-gate-bridge-fog-sunset.jpg`.
- Appends a short hash suffix to avoid name collisions.
- One vision call also classifies the photo into a fixed category (people, landscape, event, food, product, screenshot, other) and, with `--catalog`, records it in a SQLite catalog alongside the description -- no second LLM call needed.
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

# Also record description + category in the photo catalog
# (default path: ~/sync/photo-catalog/store.db)
uv run rename-photo ./photos/ --catalog

# Use a different catalog file
uv run rename-photo ./photos/ --catalog --catalog-db ~/some/other/store.db

# Pipe-friendly (reads paths from stdin, prints new paths to stdout)
scrub-photo ./photos --pipe | rename-photo --pipe --catalog
```

## Development

```bash
# Run tests (using MockProvider)
uv run pytest
```

## Part of the Photo Pipeline
`photo-metadata-scrubber` → `photo-renamer` (+ catalog) → `photo-scaler` → `unsplash-uploader`

Strip location data first, then rename/classify/catalog, then resize for the newsletter/blog. All three intermediate tools support `--pipe` and chain directly via stdin/stdout.
