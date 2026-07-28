import subprocess
from pathlib import Path

import modal

TEST_DIRECTORY = Path(__file__).parent.resolve()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(TEST_DIRECTORY))
    .add_local_file(
        TEST_DIRECTORY / "test_flash_attention.py",
        remote_path="/gpu-tests/test_flash_attention.py",
    )
)

app = modal.App("astral-build-flash-attention-gpu-tests")


@app.function(image=image, gpu="A10G", timeout=600)
def test() -> None:
    subprocess.run(
        ["python", "-m", "pytest", "-v", "/gpu-tests/test_flash_attention.py"],
        check=True,
    )
