from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    dataset_dir: str
    goal_description: str
    csv_filename: str
    iteration: int
    last_eval_results: str
    pending_feedback: str
    final_script_paths: list[str]
