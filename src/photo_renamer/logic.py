import base64
import hashlib
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel

from local_first_common.cli import (
    debug_option,
    dry_run_option,
    model_option,
    no_llm_option,
    provider_option,
    resolve_dry_run,
    resolve_provider,
    verbose_option,
    pipe_option,
    init_config_option,
)
from local_first_common.config import get_setting
from local_first_common.db import init_db, resolve_sync_path
from local_first_common.llm import parse_json_response, try_xml_parse
from local_first_common.tracking import register_tool, timed_run

TOOL_NAME = "photo-renamer"
DEFAULTS = {"provider": "ollama", "model": "llama3"}
_TOOL = register_tool(TOOL_NAME)

# Fixed taxonomy so downstream catalog queries (e.g. "give me newsletter-ready
# landscape photos") have a stable, small set of values to filter on.
CATEGORIES = ("people", "landscape", "event", "food", "product", "screenshot", "other")

CATALOG_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY,
    original_path TEXT NOT NULL,
    current_path TEXT NOT NULL,
    category TEXT,
    description TEXT,
    scrubbed INTEGER DEFAULT 0,
    scaled INTEGER DEFAULT 0,
    max_dimension INTEGER,
    processed_at TEXT
);
"""

console = Console(stderr=True)  # Rich output to stderr
app = typer.Typer(
    help="Uses a vision model to generate descriptive filenames for photos."
)


class PhotoRenamerError(Exception):
    """Base error for rename-photo core operations."""


class ProviderCallError(PhotoRenamerError):
    """Raised when the model call fails."""


class EmptyDescriptionError(PhotoRenamerError):
    """Raised when the model returns no usable description."""


class FileRenameError(PhotoRenamerError):
    """Raised when file renaming fails."""


@dataclass
class RenamePhotoResult:
    path: Path
    action: str
    description: str = ""
    category: str = "other"


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")


def parse_description_and_category(raw: str) -> tuple[str, str]:
    """Extract (description, category) from a vision-model response.

    Tries JSON first, then an XML-tag fallback (local models under token
    pressure are often more reliable with XML than strict JSON -- same
    pattern as local_first_common.scoring.BaseScorer). If both fail, the
    raw text is treated as the description with category "other" -- this
    keeps the tool's original plain-description behavior as the final
    fallback rather than raising, since renaming should not fail just
    because categorization did.
    """
    try:
        data = parse_json_response(raw)
        description = str(data.get("description", "")).strip()
        category = str(data.get("category", "other")).strip().lower()
        if description:
            return description, category if category in CATEGORIES else "other"
    except (ValueError, KeyError, TypeError):
        pass

    xml = try_xml_parse(raw, ["description", "category"])
    if xml:
        description = xml.get("description", "").strip()
        category = xml.get("category", "other").strip().lower()
        if description:
            return description, category if category in CATEGORIES else "other"

    return raw.strip(), "other"


def write_catalog_entry(
    catalog_db: Path,
    original_path: Path,
    current_path: Path,
    category: str,
    description: str,
) -> None:
    """Insert one row into the photo catalog. Fire-and-forget: never raises,
    so a catalog write failure never breaks a rename that already succeeded.
    """
    try:
        init_db(catalog_db, CATALOG_SCHEMA_SQL)
        conn = sqlite3.connect(str(catalog_db))
        try:
            conn.execute(
                "INSERT INTO photos (original_path, current_path, category, "
                "description, processed_at) VALUES (?, ?, ?, ?, ?)",
                (
                    str(original_path),
                    str(current_path),
                    category,
                    description,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]Catalog write failed (rename still succeeded): {e}[/yellow]")


def encode_image_base64(image_path: Path) -> str:
    """Read an image file and return its base64-encoded contents.

    The BaseProvider abstraction's `images=` parameter expects already-encoded
    base64 data (see AnthropicProvider._build_messages and
    OllamaProvider._build_payload), not a file path.
    """
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def get_short_hash(file_path: Path) -> str:
    """Generate a short hash for the file content to avoid collisions."""
    hasher = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            buf = f.read(1024 * 1024)
            hasher.update(buf)
        return hasher.hexdigest()[:6]
    except Exception:
        return "000000"


def rename_photo_or_raise(
    image_path: Path,
    llm,
    dry_run: bool = False,
    verbose: bool = False,
    silent: bool = False,
    catalog_db: Optional[Path] = None,
) -> RenamePhotoResult:
    """Analyze photo with vision LLM, rename based on description, and
    optionally record a catalog row (description + category) if catalog_db
    is given.

    Raises typed errors for provider and file-operation failures.
    """
    if verbose and not silent:
        console.print(f"[dim]Analyzing {image_path.name}...[/dim]")

    system_prompt = (
        "You are a helpful assistant that describes images for file naming "
        "and cataloging purposes."
    )
    user_prompt = (
        "Look at this image and respond with a JSON object with exactly two "
        'fields: "description" (4 to 6 descriptive words suitable for a '
        f'filename) and "category" (exactly one of: {", ".join(CATEGORIES)}). '
        "Return ONLY the JSON object, no other text."
    )

    try:
        encoded_image = encode_image_base64(image_path)
    except OSError as e:
        raise PhotoRenamerError(f"Could not read {image_path.name}: {e}") from e

    with timed_run(
        "photo-renamer", llm.model, source_location=image_path.as_posix()
    ) as _run:
        try:
            raw_response = llm.complete(
                system_prompt,
                user_prompt,
                images=[encoded_image],
            )
        except Exception as e:  # noqa: BLE001
            raise ProviderCallError(
                f"Model call failed for {image_path.name}: {e}"
            ) from e
        _run.item_count = 1

    if not raw_response:
        raise EmptyDescriptionError(
            f"Model returned no description for {image_path.name}"
        )

    description, category = parse_description_and_category(raw_response)
    if not description:
        raise EmptyDescriptionError(
            f"Model returned no usable description for {image_path.name}"
        )

    slug = slugify(description)
    h = get_short_hash(image_path)
    new_name = f"{slug}-{h}{image_path.suffix}"
    new_path = image_path.parent / new_name

    if dry_run:
        if not silent:
            console.print(
                f"[yellow][dry-run] Would rename {image_path.name} -> {new_name} "
                f"(category: {category})[/yellow]"
            )
        return RenamePhotoResult(
            path=new_path, action="dry_run", description=description, category=category
        )

    if image_path.name == new_name:
        if not silent:
            console.print(f"[dim]{image_path.name} is already correctly named.[/dim]")
        return RenamePhotoResult(
            path=image_path, action="unchanged", description=description, category=category
        )

    try:
        os.rename(image_path, new_path)
    except OSError as e:
        raise FileRenameError(f"Could not rename {image_path.name}: {e}") from e

    if catalog_db is not None:
        write_catalog_entry(catalog_db, image_path, new_path, category, description)

    if not silent:
        console.print(
            f"[green]Renamed {image_path.name} -> {new_name}[/green] "
            f"[dim](category: {category})[/dim]"
        )
    return RenamePhotoResult(
        path=new_path, action="renamed", description=description, category=category
    )


def rename_photo(
    image_path: Path,
    llm,
    dry_run: bool = False,
    verbose: bool = False,
    silent: bool = False,
    catalog_db: Optional[Path] = None,
) -> Optional[Path]:
    """Compatibility wrapper for callers that expect Optional[Path]."""
    try:
        result = rename_photo_or_raise(
            image_path,
            llm,
            dry_run=dry_run,
            verbose=verbose,
            silent=silent,
            catalog_db=catalog_db,
        )
        return result.path
    except PhotoRenamerError as e:
        if not silent:
            console.print(f"[red]Error processing {image_path.name}: {e}[/red]")
        return None


@app.command()
def rename(
    path: Optional[Path] = typer.Argument(None, help="File or directory to rename"),
    provider: Annotated[str, provider_option()] = os.environ.get(
        "MODEL_PROVIDER", "ollama"
    ),
    model: Annotated[Optional[str], model_option()] = None,
    dry_run: Annotated[bool, dry_run_option()] = False,
    no_llm: Annotated[bool, no_llm_option()] = False,
    verbose: Annotated[bool, verbose_option()] = False,
    debug: Annotated[bool, debug_option()] = False,
    pipe: Annotated[bool, pipe_option()] = False,
    init_config: Annotated[bool, init_config_option(TOOL_NAME, DEFAULTS)] = False,
    catalog: Annotated[
        bool,
        typer.Option(
            "--catalog", help="Record a catalog row (description + category) for each rename"
        ),
    ] = False,
    catalog_db: Annotated[
        Optional[Path],
        typer.Option(
            "--catalog-db", help="Catalog database path (default: ~/sync/photo-catalog/store.db)"
        ),
    ] = None,
):
    """Analyze photos and rename them with descriptive slugs."""
    actual_provider = get_setting(
        TOOL_NAME, "provider", cli_val=provider, default="ollama"
    )
    actual_model = get_setting(TOOL_NAME, "model", cli_val=model)
    dry_run = resolve_dry_run(dry_run, no_llm)
    llm = resolve_provider(
        provider_name=actual_provider,
        model=actual_model,
        no_llm=no_llm,
        verbose=verbose,
        debug=debug,
    )
    resolved_catalog_db = None
    if catalog:
        resolved_catalog_db = catalog_db or resolve_sync_path(
            "photo-catalog", "store.db"
        )

    # Handle stdin for piping
    files_to_process = []
    if path is None:
        if not sys.stdin.isatty():
            for line in sys.stdin:
                p = Path(line.strip())
                if p.exists():
                    files_to_process.append(p)
        else:
            console.print("[red]Error: No path provided and no stdin detected.[/red]")
            raise typer.Exit(1)
    else:
        if not path.exists():
            console.print(f"[red]Path does not exist: {path}[/red]")
            raise typer.Exit(1)
        if path.is_file():
            files_to_process.append(path)
        elif path.is_dir():
            for ext in (".jpg", ".jpeg", ".png", ".tiff", ".webp"):
                files_to_process.extend(path.glob(f"*{ext}"))
                files_to_process.extend(path.glob(f"*{ext.upper()}"))

    if not files_to_process:
        if not pipe:
            console.print("No photos found.")
        return

    if not pipe:
        console.print(
            Panel(
                f"Analyzing {len(files_to_process)} photos with {llm.model}...",
                title="Photo Renamer",
                border_style="cyan",
            )
        )

    renamed_count = 0
    for file in files_to_process:
        try:
            result = rename_photo_or_raise(
                file,
                llm,
                dry_run=dry_run,
                verbose=verbose,
                silent=pipe,
                catalog_db=resolved_catalog_db,
            )
        except PhotoRenamerError as e:
            if not pipe:
                console.print(f"[red]Error processing {file.name}: {e}[/red]")
            continue

        if result.path:
            renamed_count += 1
            if pipe:
                print(result.path.absolute())

    if not pipe:
        if not dry_run:
            console.print(
                f"\n[bold green]Done! Renamed {renamed_count} photos.[/bold green]"
            )
        else:
            console.print(
                f"\n[yellow][dry-run] Would have renamed {renamed_count} photos.[/yellow]"
            )


if __name__ == "__main__":
    app()
