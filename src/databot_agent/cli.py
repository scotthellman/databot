import asyncio
import sys
from pathlib import Path

import click
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import Command

from .agent import build_graph, build_llm
from .state import AgentState

_NODE_NAMES = {"explore", "write_scripts", "run_scripts", "evaluate", "human_review", "tools", "write_tools"}


async def _stream_graph(graph, input_, config):
    async for event in graph.astream_events(input_, config, version="v2"):
        kind = event["event"]
        name = event.get("name", "")
        if kind == "on_chain_start" and name in _NODE_NAMES:
            click.echo(f"\n\n[{name}]")
        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            content = chunk.content
            if isinstance(content, str):
                click.echo(content, nl=False)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        click.echo(block.get("text", ""), nl=False)
        elif kind == "on_tool_start" and name not in _NODE_NAMES:
            click.echo(f"\n  [tool: {name}]")


def _find_dataset_files(dataset_dir: Path) -> tuple[str, str]:
    csvs = list(dataset_dir.glob("*.csv"))
    txts = list(dataset_dir.glob("*.txt"))
    if len(csvs) != 1:
        raise click.UsageError(
            f"Expected exactly one .csv file in {dataset_dir}, found {len(csvs)}"
        )
    if len(txts) != 1:
        raise click.UsageError(
            f"Expected exactly one .txt file in {dataset_dir}, found {len(txts)}"
        )
    return csvs[0].name, txts[0].read_text(encoding="utf-8").strip()


def _display_scripts(dataset_dir: Path):
    for script in ("train.py", "eval.py"):
        path = dataset_dir / script
        if path.exists():
            click.echo(f"\n{'=' * 60}")
            click.echo(f"  {script}")
            click.echo(f"{'=' * 60}")
            click.echo(path.read_text(encoding="utf-8"))
        else:
            click.echo(f"\n[{script} not found yet]")


async def _run(dataset_dir: str, max_iterations: int, provider: str, model: str):
    dir_path = Path(dataset_dir).resolve()
    csv_filename, goal_description = _find_dataset_files(dir_path)

    click.echo(f"Dataset: {dir_path / csv_filename}")
    click.echo(
        f"Goal: {goal_description[:120]}{'...' if len(goal_description) > 120 else ''}"
    )
    click.echo(f"Model: {provider}/{model}  |  Max iterations: {max_iterations}\n")

    llm = build_llm(provider, model)

    print(sys.argv[0])
    server_cmd = "databot-server"

    client = MultiServerMCPClient(
        {
            "databot": {
                "command": server_cmd,
                "args": [str(dir_path)],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    graph = build_graph(
        tools, llm, max_iterations=max_iterations, dataset_dir=str(dir_path)
    )

    initial_state: AgentState = {
        "messages": [],
        "dataset_dir": str(dir_path),
        "goal_description": goal_description,
        "csv_filename": csv_filename,
        "iteration": 0,
        "last_eval_results": "",
        "pending_feedback": "",
        "final_script_paths": [],
    }

    config = {"configurable": {"thread_id": "1"}}
    await _stream_graph(graph, initial_state, config)
    state = graph.get_state(config)

    while state.tasks and any(t.interrupts for t in state.tasks):
        interrupt_val = state.tasks[0].interrupts[0].value
        click.echo(f"\n{interrupt_val['message']}")
        _display_scripts(dir_path)
        click.echo("")

        approved = click.confirm("Approve and run these scripts?", default=True)
        feedback = ""
        if not approved:
            feedback = click.prompt("Describe what needs to change", default="")

        await _stream_graph(graph, Command(resume={"approved": approved, "feedback": feedback}), config)
        state = graph.get_state(config)

    final_paths = state.values.get("final_script_paths", [])
    if final_paths:
        click.echo(f"\nDone! Final scripts saved to: {', '.join(final_paths)}")
        click.echo(f"MLflow artifacts are in: {dir_path / 'mlruns'}")
    else:
        click.echo("\nAgent finished (max iterations reached or stopped early).")
        click.echo(f"Last evaluation: {state.values.get('last_eval_results', 'N/A')}")


@click.command()
@click.argument("dataset_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--max-iterations", default=3, show_default=True, help="Maximum train/eval cycles"
)
@click.option(
    "--provider",
    default="anthropic",
    show_default=True,
    type=click.Choice(["anthropic", "ollama"]),
    help="LLM provider",
)
@click.option(
    "--model", default="claude-sonnet-4-6", show_default=True, help="Model name"
)
def main(dataset_dir: str, max_iterations: int, provider: str, model: str):
    """Run the databot ML agent on DATASET_DIR.

    DATASET_DIR must contain exactly one .csv file and one .txt file
    describing the prediction goal.
    """
    asyncio.run(_run(dataset_dir, max_iterations, provider, model))
