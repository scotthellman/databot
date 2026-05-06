import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .tools.filesystem import register_filesystem_tools
from .tools.data import register_data_tools
from .tools.execution import register_execution_tools

mcp = FastMCP("databot-server")


def run():
    workdir_str = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DATABOT_WORKDIR", ".")
    workdir = Path(workdir_str).resolve()
    if not workdir.is_dir():
        print(f"Error: working directory does not exist: {workdir}", file=sys.stderr)
        sys.exit(1)

    register_filesystem_tools(mcp, workdir)
    register_data_tools(mcp, workdir)
    register_execution_tools(mcp, workdir)

    mcp.run(transport="stdio")
