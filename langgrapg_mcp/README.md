# LangGraph + MCP

This folder contains four progressively-evolving Python scripts that demonstrate how to build a [LangGraph](https://github.com/langchain-ai/langgraph) chatbot agent, moving from a plain synchronous tool-calling agent to an agent whose tools are supplied dynamically by one or more **MCP (Model Context Protocol)** servers.

| Script | Sync/Async | Tool source | MCP servers wired in |
|---|---|---|---|
| `chatbot.py` | Synchronous | Local `@tool`-decorated Python function | None |
| `chatbot_async.py` | Asynchronous | Local `@tool`-decorated Python function | None |
| `chatbot_mcp.py` | Asynchronous | MCP servers via `MultiServerMCPClient` | `arith` (stdio) + `expense` (streamable HTTP) |
| `chatbot_mcp_1.py` | Asynchronous | MCP servers via `MultiServerMCPClient` | `expense` (streamable HTTP) only |

## 1. What is MCP, and why does it matter for LangGraph agents

The **Model Context Protocol (MCP)** is a standardized client-server protocol for exposing tools, resources, and prompts to LLM-based applications. Instead of an agent author writing every tool as in-process Python code (as in `chatbot.py`/`chatbot_async.py`), an MCP **server** is a separate process (or hosted HTTP endpoint) that advertises a set of callable tools with typed schemas. An MCP **client** embedded in the agent process connects to one or more of these servers, discovers their tools at runtime, and hands them to the LLM.

This decoupling matters for a few reasons visible directly in this codebase:

- **Tool implementation is separated from the agent.** In `chatbot_mcp.py` the arithmetic tool no longer lives in this repository at all — it is served by an external process (`/Users/SB/Desktop/mcp-math-server/main.py`) launched over stdio. The LangGraph script only ever sees the tool's name/schema, not its implementation.
- **Tools can live anywhere.** The `expense` server in both `chatbot_mcp.py` and `chatbot_mcp_1.py` is not a local process at all — it is a remote, already-running MCP endpoint (`https://splendid-gold-dingo.fastmcp.app/mcp`) reached over `streamable_http`. The agent code is identical whether the tool executes on the same machine or across the network.
- **Multiple independent tool providers can be combined transparently.** `chatbot_mcp.py` merges tools from two servers using two different transports (`stdio` and `streamable_http`) into a single flat tool list that is bound to the LLM exactly like the local tool in `chatbot.py` was.
- **The agent graph itself does not change.** Across all four scripts the LangGraph `StateGraph`/`chat_node`/`ToolNode`/`tools_condition` wiring is identical. MCP only changes *where the tool list comes from* (`await client.get_tools()` instead of a Python list literal); it does not change how LangGraph reasons about tool calls, so switching an agent from local tools to MCP tools (or back) requires no changes to the graph logic.

## 2. Architectural overview: how MCP plugs into LangGraph here

All four scripts share the same underlying LangGraph pattern:

```
START -> chat_node -> (tools_condition) -> tools -> chat_node -> ... -> END
```

- **State**: a single `TypedDict` (`ChatState`) holding `messages: Annotated[list[BaseMessage], add_messages]`. `add_messages` is LangGraph's reducer that appends new messages to the running conversation rather than overwriting it.
- **`chat_node`**: calls an LLM (`ChatOpenAI(model="gpt-5")`) that has been bound to the current tool list via `.bind_tools(tools)`, and returns the LLM's response message.
- **`tools` node**: a prebuilt `ToolNode(tools)` from `langgraph.prebuilt` that executes whichever tool call(s) the LLM requested.
- **Routing**: `graph.add_conditional_edges("chat_node", tools_condition)` — LangGraph's prebuilt `tools_condition` inspects the last AI message and routes to the `tools` node if it contains tool calls, otherwise to `END`. After a tool runs, the graph unconditionally routes `tools -> chat_node` so the LLM can process the tool result and produce a final answer (or call another tool).

What changes in the MCP scripts is exclusively **where the `tools` list comes from**:

- In `chatbot.py`/`chatbot_async.py`, `tools = [calculator]` — a hardcoded, in-process LangChain tool created with the `@tool` decorator.
- In `chatbot_mcp.py`/`chatbot_mcp_1.py`, an instance of `MultiServerMCPClient` (from the `langchain_mcp_adapters` package) is constructed with a dict describing one or more MCP servers. Calling `await client.get_tools()` connects to each configured server, performs MCP tool discovery, and returns a list of LangChain-compatible `BaseTool` objects — one per MCP tool exposed by those servers. This adapter is the piece that converts MCP's wire-protocol tool definitions into the same `BaseTool` interface that `.bind_tools()` and `ToolNode` already understand, which is exactly why the rest of the graph code needs no modification.

Because `client.get_tools()` is a coroutine, both MCP scripts must build their graph inside an `async def build_graph()` function and run everything (`build_graph()`, `chatbot.ainvoke(...)`) from an `async def main()` driven by `asyncio.run(main())`. There is no synchronous MCP variant in this folder — MCP tool discovery is only demonstrated with the async execution model.

Note: all four scripts import `DuckDuckGoSearchRun` from `langchain_community.tools` at the top, but none of them actually adds it to their `tools` list or uses it anywhere in the file — it is dead/unused code in every script as written.

## 3. Script-by-script breakdown

### `chatbot.py` — baseline synchronous tool-calling agent

- **Execution model**: fully synchronous. Builds the graph at module scope and calls `chatbot.invoke(...)` directly (no `async`/`await`, no `asyncio`).
- **Tools**: a single local tool, `calculator(first_num, second_num, operation)`, supporting `add`, `sub`, `mul`, `div`, returning a dict with either the computed result or an `{"error": ...}` payload (including explicit division-by-zero handling).
- **Graph**: `ChatState -> chat_node -> tools_condition -> tools -> chat_node -> END`, compiled once at module scope as `chatbot = graph.compile()`.
- **Invocation**: sends one `HumanMessage` ("Find the modulus of 132354 and 23 and give answer like a cricket commentator.") and prints `result['messages'][-1].content`.
- **Purpose demonstrated**: the minimal LangGraph ReAct-style tool-calling loop with a locally defined tool and no MCP involvement at all — the baseline every other script builds on.

### `chatbot_async.py` — same agent, asynchronous execution model

- **Execution model**: identical graph and identical `calculator` tool to `chatbot.py`, but every layer is converted to async: `chat_node` is `async def` and calls `llm_with_tools.ainvoke(messages)`; graph construction is wrapped in `def build_graph()` (itself synchronous, since building the graph requires no I/O) returning a compiled graph; running the graph happens inside `async def main()` via `await chatbot.ainvoke(...)`, launched with `asyncio.run(main())`.
- **Tools/graph structure**: unchanged from `chatbot.py` — same state, same nodes, same edges, same test prompt.
- **Purpose demonstrated**: shows how to convert a synchronous LangGraph agent to the async API (`ainvoke`, `async def` nodes) without changing behavior, as a stepping stone toward the MCP scripts, which require async because MCP tool discovery is itself async.

### `chatbot_mcp.py` — multi-server MCP agent (stdio + streamable HTTP)

- **Execution model**: async throughout, same shape as `chatbot_async.py`, but `build_graph()` is now itself `async def` because it must `await client.get_tools()` before it can bind tools to the LLM.
- **MCP client**: constructs `MultiServerMCPClient` with two servers:
  - `arith`: `transport: "stdio"`, launched as a local subprocess via `command: "python3"`, `args: ["/Users/SB/Desktop/mcp-math-server/main.py"]`. This is a hardcoded absolute path to a separate FastMCP-based arithmetic server script that is **not included in this repository** and must exist and be runnable at that path (and `python3` must be on `PATH`) for this script to work.
  - `expense`: `transport: "streamable_http"`, pointing at a hosted MCP endpoint `https://splendid-gold-dingo.fastmcp.app/mcp` (the inline comment notes `"sse"` as a fallback transport if `streamable_http` fails).
- **Tool binding**: `tools = await client.get_tools()` merges the tool sets discovered from *both* servers into one list; `print(tools)` is left in as a debug statement showing what was discovered; `llm_with_tools = llm.bind_tools(tools)` binds the combined set.
- **Graph**: identical shape to the other scripts (`chat_node` / `ToolNode(tools)` / `tools_condition`), but the tool node now dispatches to whichever MCP tool (arithmetic or expense) the LLM selects.
- **Invocation**: sends the prompt "Give me all my expenses for the month of Nov from 1 Nov to 30 Nov" — a query aimed at the `expense` MCP server's tools, not the `arith` server (the arithmetic server is wired up but not exercised by this particular test prompt).
- **Purpose demonstrated**: combining multiple MCP servers over different transports (local stdio subprocess and remote HTTP) into a single LangGraph agent's tool set.

### `chatbot_mcp_1.py` — single-server MCP agent (streamable HTTP only)

- **Execution model**: identical async structure to `chatbot_mcp.py` (`async def build_graph()`, `async def main()`, `asyncio.run(main())`).
- **MCP client**: constructs `MultiServerMCPClient` with only the `expense` server (`transport: "streamable_http"`, same `https://splendid-gold-dingo.fastmcp.app/mcp` URL). The `arith` stdio server present in `chatbot_mcp.py` has been removed entirely.
- **Tools/graph/prompt**: otherwise byte-for-byte the same as `chatbot_mcp.py` — same `ChatState`, same node/edge wiring, same `print(tools)` debug line, same expense-report test prompt.
- **Purpose demonstrated**: an isolated, minimal single-MCP-server example — useful for testing/demoing the hosted `expense` server on its own, without the extra complexity (and extra local-process prerequisite) of the `arith` stdio server. This is the simplest of the two MCP scripts and a natural first step before `chatbot_mcp.py`'s multi-server setup.

## 4. Configuration files in the repository root

The repository root (`C:\Users\debab\Desktop\IIT+SELF LEARNING\CODING\LangGraph\`) contains two JSON files, `mcp_config.json` and `browser_mcp.json`. Both files currently contain **identical content**: an `mcpServers` map describing nine unrelated MCP servers — `google-search`, `wikipedia`, `playwright`, `airbnb`, `github`, `git`, `filesystem`, `postgres`, and `memory` — each launched via `npx`/`uvx` with its own `command`/`args`/`env`.

Important: **none of the four scripts in this folder read, import, or reference either `mcp_config.json` or `browser_mcp.json`.** There is no config-loading code (no `json.load`, no path reference to these filenames) anywhere in `chatbot.py`, `chatbot_async.py`, `chatbot_mcp.py`, or `chatbot_mcp_1.py`. Both MCP scripts instead define their MCP server list **inline in Python** as a dict literal passed directly to `MultiServerMCPClient(...)`. The two root-level JSON files appear to belong to a different, more general MCP setup (the schema matches the config format used by MCP-aware editors/clients such as Claude Desktop or similar tools), not to the servers these four scripts actually connect to (`arith`, `expense`). Anyone trying to run `chatbot_mcp.py` or `chatbot_mcp_1.py` should not expect editing `mcp_config.json`/`browser_mcp.json` to have any effect on them.

**Security note**: both `mcp_config.json` and `browser_mcp.json` currently contain what appear to be live-looking credential values in plaintext (an API key/CSE ID under `google-search`, and a GitHub personal access token under `github`). This README intentionally does not reproduce those values. If these are real, active credentials, they should be rotated and moved to environment variables / a `.env` file (consistent with how `load_dotenv()` is already used for the OpenAI key in these scripts) rather than committed to a JSON file in the repo.

### Environment variables

All four scripts call `load_dotenv()` at import time and construct `ChatOpenAI(model="gpt-5")` with no explicit API key argument, which means an OpenAI-compatible API key (conventionally `OPENAI_API_KEY`) is expected to be present in a `.env` file or the process environment. No `.env` file is read directly by this README's author; its exact contents are out of scope and not inspected here. No other environment variables are referenced by any of the four scripts.

## 5. Prerequisites and limitations observed in the code

- **Python packages**: `langgraph`, `langchain-openai`, `langchain-community` (for the unused `DuckDuckGoSearchRun` import), `python-dotenv`, and — for the two MCP scripts — `langchain_mcp_adapters`.
- **OpenAI-compatible API key**: required in the environment (via `.env`/`load_dotenv()`) for `ChatOpenAI(model="gpt-5")` to function in every script.
- **`chatbot_mcp.py` requires a running/reachable `arith` MCP server**: the `command`/`args` (`python3 /Users/SB/Desktop/mcp-math-server/main.py`) is a hardcoded macOS-style path outside this repository. On a different machine (including this Windows environment) this path will not exist, `python3` may not be on `PATH`, and the script will fail to start the `arith` server unless this is corrected to a valid local path/interpreter for the machine it is run on.
- **Both MCP scripts depend on network reachability** of the hosted `expense` MCP endpoint (`https://splendid-gold-dingo.fastmcp.app/mcp`) over `streamable_http`; the inline comment in the code notes `"sse"` as a fallback transport if `streamable_http` does not work against that endpoint.
- **No error handling around MCP connection failures**: neither MCP script wraps `client.get_tools()` or the server connections in try/except, so a server being unreachable (stdio process missing, HTTP endpoint down) will surface as an unhandled exception.
- **Dead import**: `DuckDuckGoSearchRun` is imported in every script but never used or added to any tool list.
- **Debug output left in place**: both MCP scripts contain a bare `print(tools)` call inside `build_graph()`, printing the discovered MCP tool objects to stdout on every run.
- **Single-turn, script-based invocation only**: none of the four scripts implement an interactive loop (e.g. reading from stdin repeatedly); each script sends exactly one hardcoded `HumanMessage` and prints the final response, then exits.
