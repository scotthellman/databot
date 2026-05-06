from pathlib import Path

from .filesystem import resolve_path


def register_data_tools(mcp, workdir: Path):
    @mcp.tool()
    def get_csv_summary(path: str) -> str:
        """Return a statistical summary of a CSV file: shape, column types, null counts, and descriptive statistics."""
        import polars as pl

        target = resolve_path(workdir, path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")

        df = pl.read_csv(target, infer_schema_length=10000)

        lines = [
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
            "",
            "Column types:",
        ]
        for col in df.columns:
            null_count = df[col].null_count()
            null_pct = 100 * null_count / df.shape[0]
            lines.append(f"  {col}: {df[col].dtype}  (nulls: {null_count} = {null_pct:.1f}%)")

        lines += ["", "Descriptive statistics:"]
        try:
            desc = df.describe()
            lines.append(str(desc))
        except Exception as e:
            lines.append(f"  (could not compute describe: {e})")

        lines += ["", "First 3 rows:"]
        lines.append(str(df.head(3)))

        return "\n".join(lines)
