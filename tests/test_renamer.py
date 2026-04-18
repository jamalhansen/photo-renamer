from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from photo_renamer.logic import app, get_short_hash, rename_photo, slugify
from local_first_common.testing import MockProvider


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


def test_rename_photo_returns_none_when_description_missing(tmp_path):
    img_path = create_test_image(tmp_path, "original.jpg")
    llm = MockProvider(response="")

    result = rename_photo(img_path, llm, silent=True)

    assert result is None
    assert img_path.exists()


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


def test_rename_command_missing_path_exits_nonzero(tmp_path):
    missing = tmp_path / "does-not-exist"

    result = runner.invoke(app, [str(missing), "--no-llm"])

    assert result.exit_code == 1
