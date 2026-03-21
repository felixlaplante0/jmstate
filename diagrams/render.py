import subprocess
from pathlib import Path

for qmd_file in Path(__file__).parent.glob("*.qmd"):
    subprocess.run(["quarto", "render", str(qmd_file), "--to", "pdf"], check=True)
