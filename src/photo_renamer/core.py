import base64
import hashlib
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from local_first_common.db import init_db
from local_first_common.llm import parse_json_response, try_xml_parse
from local_first_common.tracking import timed_run
from rich.console import Console

console = Console(stderr=True)

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
    """Extract (description, category) from a vision-model response."""
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
    """Insert one row into the photo catalog."""
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
                    datetime.now(UTC).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]Catalog write failed (rename still succeeded): {e}[/yellow]")


def encode_image_base64(image_path: Path) -> str:
    """Read an image file and return its base64-encoded contents."""
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def get_short_hash(file_path: Path) -> str:
    """Generate a short hash for the file content to avoid collisions."""
    hasher = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            buf = f.read(1024 * 1024)
            hasher.update(buf)
        return hasher.hexdigest()[:6]
    except Exception:  # noqa: BLE001 - hash is a best-effort collision-avoidance suffix; any read failure should fall back, not crash the rename
        return "000000"


def rename_photo_or_raise(
    image_path: Path,
    llm,
    dry_run: bool = False,
    verbose: bool = False,
    silent: bool = False,
    catalog_db: Path | None = None,
) -> RenamePhotoResult:
    """Analyze photo with vision LLM, rename based on description, and
    optionally record a catalog row (description + category) if catalog_db
    is given.
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
        except Exception as e:
            raise ProviderCallError(
                f"Model call failed for {image_path.name}: {e}"
            ) from e
        _run.item_count = 1
        # llm.model was "" at timed_run() call time for a GatewayProvider with
        # no explicit --model -- re-read both now that the call has resolved
        # it (provider too, in case a FallbackProvider switched legs mid-call).
        _run.model = llm.model
        _run.provider = getattr(llm, "provider_name", None)

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
    catalog_db: Path | None = None,
) -> Path | None:
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
