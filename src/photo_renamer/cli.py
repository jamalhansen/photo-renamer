import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from local_first_common.cli import (
    debug_option,
    dry_run_option,
    init_config_option,
    model_option,
    no_llm_option,
    pipe_option,
    provider_option,
    resolve_dry_run,
    resolve_provider,
    verbose_option,
)
from local_first_common.config import get_setting
from local_first_common.db import resolve_sync_path
from local_first_common.tracking import register_tool
from rich.console import Console
from rich.panel import Panel

from .core import (
    PhotoRenamerError,
    rename_photo_or_raise,
)

TOOL_NAME = "photo-renamer"
DEFAULTS = {"provider": "ollama", "model": "llama3"}
_TOOL = register_tool(TOOL_NAME)

console = Console(stderr=True)  # Rich output to stderr
app = typer.Typer(
    help="Uses a vision model to generate descriptive filenames for photos."
)


@app.command()
def rename(
    path: Annotated[Path | None, typer.Argument(help="File or directory to rename")] = None,
    provider: Annotated[str, provider_option()] = os.environ.get(
        "MODEL_PROVIDER", "ollama"
    ),
    model: Annotated[str | None, model_option()] = None,
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
        Path | None,
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
