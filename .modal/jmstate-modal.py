"""Run one jmstate notebook on a T4, repo untouched."""

import argparse
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent

app = modal.App("jmstate-t4")
image = modal.Image.debian_slim(python_version="3.14").uv_pip_install(
    "nbconvert",
    "ipykernel",
    requirements=[str(REPO / "scripts" / "requirements.txt")],
)
repo = modal.Volume.from_name("jmstate", create_if_missing=True)


@app.function(gpu="T4", image=image, volumes={"/mnt/jmstate": repo}, timeout=36000)
def run_notebook(notebook: str):
    """Execute one repo notebook headless and save it with outputs."""
    subprocess.run(  # noqa: S603
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            f"/mnt/jmstate/scripts/{notebook}.ipynb",
            f"--output=/mnt/jmstate/results/{notebook}-executed.ipynb",
            "--ExecutePreprocessor.timeout=-1",
        ],
        cwd="/mnt/jmstate/scripts",
        check=True,
    )
    repo.commit()


def main() -> None:
    """Parse CLI args and launch the notebook on Modal."""
    parser = argparse.ArgumentParser(description="Run a jmstate notebook on a Modal T4.")
    parser.add_argument(
        "--notebook", default="fitting-test", help="notebook in scripts/, without .ipynb"
    )
    parsed = parser.parse_args()
    with app.run():
        run_notebook.remote(parsed.notebook)
    print("Executed notebook is on the volume. Pull it with:")  # noqa: T201
    print("  modal volume ls jmstate results")  # noqa: T201
    print(  # noqa: T201
        "  modal volume get jmstate "
        "results/<notebook>-executed.ipynb <local-path>"
    )


if __name__ == "__main__":
    main()
