# %% [markdown]
# <span style="font-size: 5em">🦜</span>

# %% [markdown]
# # __LangGraph Essentials__
# ## Building a Tool-Calling Agent with LangGraph

# %% [markdown]
# An "agent" is just a graph with a loop: a model node that can decide to call a tool, a tool node that executes it, and a conditional edge that sends control back to the model until it's ready to answer. This notebook builds that loop directly with `StateGraph`
# 
# The model is a local Ollama model.

# %% [markdown]
# ## Setup

# %% [markdown]
# Load and/or check for needed environmental variables

# %%
#%%capture --no-stderr
#%pip install --quiet -U langgraph>=1.0.0 langchain_core langchain_ollama langfuse>=4
#%pip install --upgrade python-dotenv

# %%
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Ollama server URL - override with OLLAMA_BASE_URL env var
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

os.environ.setdefault("LANGFUSE_BASE_URL", "http://localhost:3000")
os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "labs")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-local-dev-key")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-local-dev-secret")

# %%
from typing import Annotated, Literal, TypedDict

from IPython.display import Image, display
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# %% [markdown]
# ## State
# 
# <a id='state_definition'></a>
# 
# The agent's state is a running list of messages. `add_messages` is a reducer: each node returns new messages, and LangGraph appends them to the existing list rather than overwriting it — this is what gives the agent conversational memory within a single run.

# %%
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# %% [markdown]
# ## Define a tool
# 
# <a id='tool_definition'></a>
# 
# A single, simple tool the agent can call.

# %%
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    print(f"🌤️ Looking up weather for {city}")
    return f"It's always sunny in {city}!"


tools = [get_weather]
tools_by_name = {t.name: t for t in tools}

# %% [markdown]
# ## Model node
# 
# <a id='model_node'></a>
# 
# `bind_tools` tells the model which tools are available so it can request a tool call instead of (or in addition to) answering directly. The node itself is just a function: read the message history from state, call the model, and return the new message.

# %%
SYSTEM_PROMPT = "You are a helpful assistant. only use the tools data dont generatate your own"

llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0.7,
    base_url=OLLAMA_BASE_URL,
)
llm_with_tools = llm.bind_tools(tools)


def call_model(state: AgentState) -> AgentState:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return AgentState(messages=[response])

# %% [markdown]
# ## Tool node
# 
# <a id='tool_node'></a>
# 
# When the model responds with one or more tool calls, this node executes each one and wraps the result in a `ToolMessage` so the model can read it on the next turn.

# %%
def call_tools(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        results.append(
            ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
        )
    return AgentState(messages=results)

# %% [markdown]
# ## Conditional routing
# 
# <a id='conditional_edge_func'></a>
# 
# After the model node runs, this function decides where to go next: back out to `tools` if the model asked for a tool call, or `END` if it produced a final answer.

# %%
def should_continue(state: AgentState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# %% [markdown]
# ## Build the graph
# 
# <a id='agent_graph'></a>
# 
# Wire the nodes together: `model` runs first, `should_continue` decides whether to route to `tools` or stop, and `tools` always loops back to `model` so it can use the tool's result.

# %%
builder = StateGraph(AgentState)

builder.add_node("model", call_model)
builder.add_node("tools", call_tools)

builder.add_edge(START, "model")
builder.add_conditional_edges("model", should_continue, ["tools", END])
builder.add_edge("tools", "model")
#builder.add_edge("model", "tools")
#builder.add_edge("tools", END)

graph = builder.compile()

# %%
display(Image(graph.get_graph().draw_mermaid_png()))

# %% [markdown]
# ## Run the agent
# 
# <a id='graph_invoke'></a>

# %%
result = graph.invoke(
    {"messages": [{"role": "user", "content": "What's the weather like in Bengaluru?"}]}
)
print(result["messages"][-1].content)

# %% [markdown]
# Streaming shows each step of the loop: the model deciding to call the tool, the tool node running it, and the model producing the final answer.

# %%
for step in graph.stream(
    {"messages": [{"role": "user", "content": "What's the weather like in Tokyo?"}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# %% [markdown]
# ## Try your own
# 
# Ask a question that doesn't need the weather tool at all. `should_continue` should route straight to `END` on the first pass, since the model has no tool calls to make.

# %%
result = graph.invoke(
    {"messages": [{"role": "user", "content": "What is 12 * 7?"}]}
)

print(result["messages"][-1].content)

# %% [markdown]
# ## Takeaways
# 
# Setup:
# 
# - State: [AgentState](#state_definition) uses the `add_messages` reducer so each node's output is appended to history, not overwritten
# - Tool: A plain [`@tool`-decorated function](#tool_definition), bound to the model with `llm.bind_tools(tools)`
# - Model node: [call_model](#model_node) sends the full message history to the model and returns its response
# - Tool node: [call_tools](#tool_node) executes every tool call the model requested and returns the results as `ToolMessage`s
# - Routing: [should_continue](#conditional_edge_func) inspects the model's last message and decides `tools` vs `END`
# - Graph: [model → tools → model](#agent_graph) forms the loop; it only exits through `should_continue` returning `END`
# 
# This is exactly what higher-level agent constructors (like `create_agent`) build for you automatically — but wiring it by hand means you can change any piece: swap the routing logic, add more nodes before/after the model call, or run multiple tool calls differently.
# 
# Try Next:
# 
# - Add a second tool and ask a question that requires both
# - Add a [checkpointer](#memory_setup) (see Lab 4) so the agent remembers earlier turns across multiple `invoke` calls
# - Add a node before `model` that validates or rewrites the user's question


