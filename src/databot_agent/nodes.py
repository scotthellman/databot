import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from .prompts import (
    EVALUATE_PROMPT,
    EXPLORE_PROMPT,
    FEEDBACK_SECTION,
    SYSTEM_PROMPT,
    WRITE_SCRIPTS_PROMPT,
)
from .state import AgentState


def make_explore_node(llm_with_tools):
    async def explore_node(state: AgentState) -> dict:
        existing = state["messages"]
        if not existing:
            prompt = EXPLORE_PROMPT.format(
                goal_description=state["goal_description"],
                csv_filename=state["csv_filename"],
            )
            new_msgs = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        else:
            new_msgs = []
        response = await llm_with_tools.ainvoke(existing + new_msgs)
        return {"messages": new_msgs + [response]}

    return explore_node


def make_write_scripts_node(llm_with_tools):
    async def write_scripts_node(state: AgentState) -> dict:
        messages = state["messages"]
        last_is_tool_result = messages and messages[-1].type == "tool"
        if not last_is_tool_result:
            feedback = state.get("pending_feedback", "")
            feedback_section = (
                FEEDBACK_SECTION.format(feedback=feedback) if feedback else ""
            )
            prompt = WRITE_SCRIPTS_PROMPT.format(feedback_section=feedback_section)
            new_message = HumanMessage(content=prompt)
            messages = messages + [new_message]
            state_update = {"pending_feedback": ""}
        else:
            new_message = None
            state_update = {}

        response = await llm_with_tools.ainvoke(messages)
        new_msgs = ([new_message, response] if new_message else [response])
        return {**state_update, "messages": new_msgs}

    return write_scripts_node


def make_human_review_node(workdir: str):
    async def human_review_node(state: AgentState) -> dict:
        decision = interrupt({
            "message": "The agent has written train.py and eval.py. Review them before execution.",
            "workdir": workdir,
        })
        feedback = decision.get("feedback", "") if not decision.get("approved", True) else ""
        return {"pending_feedback": feedback}

    return human_review_node


def make_run_scripts_node(tools: list):
    run_python_file = next(t for t in tools if t.name == "run_python_file")

    async def run_scripts_node(state: AgentState) -> dict:
        train_result = await run_python_file.ainvoke({"path": "train.py"})
        eval_result = await run_python_file.ainvoke({"path": "eval.py"})

        summary = (
            f"## train.py result (returncode={train_result['returncode']})\n"
            f"STDOUT:\n{train_result['stdout']}\n"
            f"STDERR:\n{train_result['stderr']}\n\n"
            f"## eval.py result (returncode={eval_result['returncode']})\n"
            f"STDOUT:\n{eval_result['stdout']}\n"
            f"STDERR:\n{eval_result['stderr']}"
        )
        return {"messages": [HumanMessage(content=summary)]}

    return run_scripts_node


def make_evaluate_node(llm, max_iterations: int):
    async def evaluate_node(state: AgentState) -> dict:
        prompt = EVALUATE_PROMPT.format(max_iterations=max_iterations)
        new_message = HumanMessage(content=prompt)
        messages = state["messages"] + [new_message]

        response = await llm.ainvoke(messages)
        content = response.content.strip()

        try:
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content)
            satisfied = bool(result.get("satisfied", False))
            reason = result.get("reason", "")
        except (json.JSONDecodeError, IndexError):
            satisfied = False
            reason = "Could not parse evaluation response"

        iteration = state.get("iteration", 0) + 1
        last_eval = f"Iteration {iteration}: satisfied={satisfied}. {reason}"

        update: dict = {
            "messages": [new_message, response],
            "iteration": iteration,
            "last_eval_results": last_eval,
        }
        if satisfied:
            update["final_script_paths"] = ["train.py", "eval.py"]

        return update

    return evaluate_node
