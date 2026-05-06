from pathlib import Path


def resolve_path(root: Path, user_path: str) -> Path:
    resolved_root = root.resolve()
    resolved_target = (root / user_path).resolve()
    if not str(resolved_target).startswith(str(resolved_root)):
        raise ValueError(f"Path '{user_path}' escapes the working directory")
    return resolved_target


def register_filesystem_tools(mcp, workdir: Path):
    @mcp.tool()
    def list_files(subdir: str = ".") -> list[str]:
        """List files in the working directory (or a subdirectory)."""
        target = resolve_path(workdir, subdir)
        results = []
        for p in target.rglob("*"):
            if p.is_file() and not any(part.startswith(".") for part in p.parts):
                results.append(str(p.relative_to(workdir)))
        return sorted(results)

    @mcp.tool()
    def read_file(path: str) -> str:
        """Read a text file from the working directory."""
        target = resolve_path(workdir, path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")
        size = target.stat().st_size
        if size > 1_000_000:
            raise ValueError(f"File too large to read ({size} bytes); max is 1 MB")
        return target.read_text(encoding="utf-8")

    @mcp.tool()
    def write_file(path: str, content: str) -> str:
        """Write content to a file in the working directory. Creates parent directories if needed."""
        target = resolve_path(workdir, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path}"
