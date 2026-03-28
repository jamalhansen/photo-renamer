from pathlib import Path

from PIL import Image
from photo_renamer.logic import rename_photo, slugify
from local_first_common.testing import MockProvider

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
