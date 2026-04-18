import hashlib
import os
import re
import sys
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
)
from local_first_common.tracking import register_tool, timed_run

_TOOL = register_tool("photo-renamer")
console = Console(stderr=True) # Rich output to stderr
app = typer.Typer(help="Uses a vision model to generate descriptive filenames for photos.")

def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")

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

def rename_photo(
    image_path: Path,
    llm,
    dry_run: bool = False,
    verbose: bool = False,
    silent: bool = False,
) -> Optional[Path]:
    """Analyze photo with vision LLM and rename based on description."""
    try:
        if verbose and not silent:
            console.print(f"[dim]Analyzing {image_path.name}...[/dim]")

        system_prompt = "You are a helpful assistant that describes images for file naming purposes."
        user_prompt = "Describe this image in 4 to 6 descriptive words. Return ONLY the description, no other text."

        with timed_run("photo-renamer", llm.model, source_location=image_path.as_posix()) as _run:
            description = llm.complete(
                system_prompt,
                user_prompt,
                images=[image_path.as_posix()],
            )
            _run.item_count = 1

        if not description:
            if not silent:
                console.print(f"[red]Failed to get description for {image_path.name}[/red]")
            return None

        slug = slugify(description)
        h = get_short_hash(image_path)
        new_name = f"{slug}-{h}{image_path.suffix}"
        new_path = image_path.parent / new_name

        if dry_run:
            if not silent:
                console.print(f"[yellow][dry-run] Would rename {image_path.name} -> {new_name}[/yellow]")
            return new_path

        if image_path.name == new_name:
            if not silent:
                console.print(f"[dim]{image_path.name} is already correctly named.[/dim]")
            return image_path

        os.rename(image_path, new_path)
        if not silent:
            console.print(f"[green]Renamed {image_path.name} -> {new_name}[/green]")
        return new_path

    except Exception as e:
        if not silent:
            console.print(f"[red]Error processing {image_path.name}: {e}[/red]")
        return None

@app.command()
def rename(
    path: Annotated[Optional[Path], typer.Argument(help="File or directory to rename")] = None,
    provider: Annotated[str, provider_option()] = "ollama",
    model: Annotated[Optional[str], model_option()] = "@vision",
    dry_run: Annotated[bool, dry_run_option()] = False,
    no_llm: Annotated[bool, no_llm_option()] = False,
    verbose: Annotated[bool, verbose_option()] = False,
    debug: Annotated[bool, debug_option()] = False,
    pipe: Annotated[bool, pipe_option()] = False,
):
    """Analyze photos and rename them with descriptive slugs."""
    dry_run = resolve_dry_run(dry_run, no_llm)
    llm = resolve_provider(provider_name=provider, model=model, no_llm=no_llm, verbose=verbose, debug=debug)

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
        console.print(Panel(f"Analyzing {len(files_to_process)} photos with {llm.model}...", title="Photo Renamer", border_style="cyan"))

    renamed_count = 0
    for file in files_to_process:
        new_path = rename_photo(file, llm, dry_run=dry_run, verbose=verbose, silent=pipe)
        if new_path:
            renamed_count += 1
            if pipe:
                print(new_path.absolute())

    if not pipe:
        if not dry_run:
            console.print(f"\n[bold green]Done! Renamed {renamed_count} photos.[/bold green]")
        else:
            console.print(f"\n[yellow][dry-run] Would have renamed {renamed_count} photos.[/yellow]")

if __name__ == "__main__":
    app()
