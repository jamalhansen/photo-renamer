from pathlib import Path

from local_first_common.testing import MockProvider
from PIL import Image
from typer.testing import CliRunner

from photo_renamer.cli import app
from photo_renamer.core import (
    EmptyDescriptionError,
    ProviderCallError,
    get_short_hash,
    parse_description_and_category,
    rename_photo,
    rename_photo_or_raise,
    slugify,
)

runner = CliRunner()


def create_test_image(tmp_path: Path, name: str = "test.jpg"):
    img_path = tmp_path / name
    img = Image.new("RGB", (100, 100), color="green")
    img.save(img_path)
    return img_path


def test_slugify():
    assert slugify("Golden Gate Bridge Fog") == "golden-gate-bridge-fog"
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("multiple   spaces") == "multiple-spaces"


def test_rename_photo_mock(tmp_path):
    img_path = create_test_image(tmp_path, "original.jpg")

    # Mock LLM response
    llm = MockProvider(response="Golden Gate Bridge Fog")

    new_path = rename_photo(img_path, llm)

    assert new_path is not None
    assert "golden-gate-bridge-fog" in new_path.name
    assert new_path.suffix == ".jpg"
    assert new_path.exists()
    assert not img_path.exists()


def test_rename_photo_dry_run(tmp_path):
    img_path = create_test_image(tmp_path, "original.jpg")

    llm = MockProvider(response="San Francisco Skyline")

    new_path = rename_photo(img_path, llm, dry_run=True)

    assert new_path is not None
    assert "san-francisco-skyline" in new_path.name
    assert img_path.exists()
    assert not new_path.exists()


def test_get_short_hash_for_missing_file_returns_zeros(tmp_path):
    missing = tmp_path / "missing.jpg"

    assert get_short_hash(missing) == "000000"


def test_rename_photo_logs_the_model_resolved_after_the_call(tmp_path):
    """Regression 2026-09-20: a GatewayProvider with no explicit --model has
    .model == "" until the gateway resolves it server-side -- that happens
    *during* llm.complete(), but timed_run()'s model argument was already
    evaluated (as "") before the call ran. photo-renamer was one of the real
    tools whose processing_log rows showed up empty because of this."""
    from local_first_common.tracking import get_tracking_db_path

    class ResolvesModelDuringCall(MockProvider):
        default_model = ""  # matches GatewayProvider's own real default when no model is specified

        def _complete(self, system, user, response_model=None, images=None):
            result = super()._complete(system, user, response_model, images)
            self.model = "phi4-mini"  # simulates the gateway resolving an unspecified model
            return result

    img_path = create_test_image(tmp_path, "original.jpg")
    llm = ResolvesModelDuringCall(response="Golden Gate Bridge Fog")
    assert llm.model == ""

    rename_photo(img_path, llm)

    import duckdb

    conn = duckdb.connect(str(get_tracking_db_path()))
    row = conn.execute(
        "SELECT model, provider FROM processing_log WHERE tool_name = 'photo-renamer' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row[0] == "phi4-mini"
    assert row[1] == "mock"


def test_rename_photo_returns_none_when_description_missing(tmp_path):
    img_path = create_test_image(tmp_path, "original.jpg")
    llm = MockProvider(response="")

    result = rename_photo(img_path, llm, silent=True)

    assert result is None
    assert img_path.exists()


def test_rename_photo_or_raise_raises_on_missing_description(tmp_path):
    img_path = create_test_image(tmp_path, "original.jpg")
    llm = MockProvider(response="")

    try:
        rename_photo_or_raise(img_path, llm, silent=True)
    except EmptyDescriptionError:
        pass
    else:
        raise AssertionError("Expected EmptyDescriptionError")


def test_rename_photo_returns_same_path_when_already_named(tmp_path):
    llm = MockProvider(response="Golden Gate Bridge Fog")
    original = create_test_image(tmp_path, "seed.jpg")
    expected = rename_photo(original, llm, silent=True)
    assert expected is not None

    same_path = rename_photo(expected, llm, silent=True)

    assert same_path == expected
    assert expected.exists()


def test_rename_photo_returns_none_on_llm_exception(tmp_path):
    class BrokenProvider:
        model = "broken-model"

        def complete(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    img_path = create_test_image(tmp_path, "original.jpg")

    result = rename_photo(img_path, BrokenProvider(), silent=True)

    assert result is None
    assert img_path.exists()


def test_rename_photo_or_raise_raises_on_llm_exception(tmp_path):
    class BrokenProvider:
        model = "broken-model"

        def complete(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    img_path = create_test_image(tmp_path, "original.jpg")

    try:
        rename_photo_or_raise(img_path, BrokenProvider(), silent=True)
    except ProviderCallError:
        pass
    else:
        raise AssertionError("Expected ProviderCallError")


def test_rename_command_missing_path_exits_nonzero(tmp_path):
    missing = tmp_path / "does-not-exist"

    result = runner.invoke(app, [str(missing), "--no-llm"])

    assert result.exit_code == 1


def test_parse_description_and_category_json():
    raw = '{"description": "golden gate bridge fog", "category": "landscape"}'

    description, category = parse_description_and_category(raw)

    assert description == "golden gate bridge fog"
    assert category == "landscape"


def test_parse_description_and_category_json_in_fences():
    raw = '```json\n{"description": "birthday cake candles", "category": "event"}\n```'

    description, category = parse_description_and_category(raw)

    assert description == "birthday cake candles"
    assert category == "event"


def test_parse_description_and_category_unknown_category_falls_back_to_other():
    raw = '{"description": "a cat on a windowsill", "category": "not-a-real-category"}'

    description, category = parse_description_and_category(raw)

    assert description == "a cat on a windowsill"
    assert category == "other"


def test_parse_description_and_category_xml_fallback():
    raw = "<description>san francisco skyline</description><category>landscape</category>"

    description, category = parse_description_and_category(raw)

    assert description == "san francisco skyline"
    assert category == "landscape"


def test_parse_description_and_category_plain_text_fallback():
    """Matches the tool's original pre-catalog behavior: a plain-text response
    (no JSON, no XML) is used directly as the description with category "other"."""
    raw = "Golden Gate Bridge Fog"

    description, category = parse_description_and_category(raw)

    assert description == "Golden Gate Bridge Fog"
    assert category == "other"


def test_rename_photo_writes_catalog_entry(tmp_path):
    import sqlite3

    img_path = create_test_image(tmp_path, "original.jpg")
    catalog_db = tmp_path / "catalog" / "store.db"
    llm = MockProvider(
        response='{"description": "golden gate bridge fog", "category": "landscape"}'
    )

    result = rename_photo_or_raise(img_path, llm, catalog_db=catalog_db)

    assert catalog_db.exists()
    conn = sqlite3.connect(str(catalog_db))
    try:
        row = conn.execute(
            "SELECT original_path, current_path, category, description FROM photos"
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row[0] == str(img_path)
    assert row[1] == str(result.path)
    assert row[2] == "landscape"
    assert row[3] == "golden gate bridge fog"


def test_rename_photo_dry_run_does_not_write_catalog_entry(tmp_path):
    img_path = create_test_image(tmp_path, "original.jpg")
    catalog_db = tmp_path / "catalog" / "store.db"
    llm = MockProvider(
        response='{"description": "golden gate bridge fog", "category": "landscape"}'
    )

    rename_photo_or_raise(img_path, llm, dry_run=True, catalog_db=catalog_db)

    assert not catalog_db.exists()
