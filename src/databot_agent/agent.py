from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .nodes import (
    make_evaluate_node,
    make_explore_node,
    make_human_review_node,
    make_run_scripts_node,
    make_write_scripts_node,
)
from .state import AgentState


def build_llm(provider: str, model: str):
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model)
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, base_url="http://localhost:11434")
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose 'anthropic' or 'ollama'.")


def _human_review_router(state: AgentState) -> str:
    feedback = state.get("pending_feedback", "")
    return "write_scripts" if feedback else "run_scripts"


def _evaluate_router(state: AgentState, max_iterations: int) -> str:
    if state.get("final_script_paths"):
        return END
    if state.get("iteration", 0) >= max_iterations:
        return END
    return "write_scripts"


def build_graph(tools: list, llm, max_iterations: int = 3, dataset_dir: str = "."):
    llm_with_tools = llm.bind_tools(tools)

    explore = make_explore_node(llm_with_tools)
    write_scripts = make_write_scripts_node(llm_with_tools)
    human_review = make_human_review_node(dataset_dir)
    run_scripts = make_run_scripts_node(tools)
    evaluate = make_evaluate_node(llm, max_iterations)
    tool_executor = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("explore", explore)
    graph.add_node("tools", tool_executor)
    graph.add_node("write_scripts", write_scripts)
    graph.add_node("human_review", human_review)
    graph.add_node("run_scripts", run_scripts)
    graph.add_node("evaluate", evaluate)

    graph.add_edge(START, "explore")
    graph.add_conditional_edges("explore", tools_condition, {"tools": "tools", END: "write_scripts"})
    graph.add_edge("tools", "explore")

    graph.add_conditional_edges(
        "write_scripts",
        tools_condition,
        {"tools": "write_tools", END: "human_review"},
    )

    write_tool_executor = ToolNode(tools)
    graph.add_node("write_tools", write_tool_executor)
    graph.add_edge("write_tools", "write_scripts")

    graph.add_conditional_edges(
        "human_review",
        _human_review_router,
        {"run_scripts": "run_scripts", "write_scripts": "write_scripts"},
    )
    graph.add_edge("run_scripts", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        lambda s: _evaluate_router(s, max_iterations),
        {"write_scripts": "write_scripts", END: END},
    )

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
