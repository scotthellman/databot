import subprocess
import sys
from pathlib import Path

from .filesystem import resolve_path


def register_execution_tools(mcp, workdir: Path):
    @mcp.tool()
    def run_python_file(path: str, timeout_seconds: int = 120) -> dict:
        """Run a Python script inside the working directory. Returns stdout, stderr, and the return code."""
        target = resolve_path(workdir, path)
        if not target.exists():
            raise FileNotFoundError(f"Script not found: {path}")
        if target.suffix != ".py":
            raise ValueError(f"Only .py files can be executed, got: {path}")

        result = subprocess.run(
            [sys.executable, str(target)],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
