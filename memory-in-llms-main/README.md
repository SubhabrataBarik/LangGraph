# Memory in LLM Applications with LangGraph

This repository is a progressive, notebook-driven tutorial series on implementing memory in LLM-powered agents using **LangGraph**. It walks from the most basic form of conversational memory to advanced management strategies, and finally to durable, cross-session memory backed by Postgres.

## Short-Term Memory vs. Long-Term Memory

LLM agent architectures generally need two distinct kinds of memory, because they solve different problems:

- **Short-term memory (STM)** is the memory of a single conversation thread. It is the list of messages (`HumanMessage`, `AIMessage`, etc.) exchanged within one session, held in the graph's **state** (`MessagesState`) and made durable across invocations by a **checkpointer** (e.g. `InMemorySaver`, `PostgresSaver`). STM answers the question "what has been said so far in *this* conversation?" It is scoped to a `thread_id`. Without STM, every `graph.invoke()` call would be stateless and the model would have no memory of the previous turn (as shown in notebook 1 before a checkpointer is added).
- **Long-term memory (LTM)** is memory that persists **across threads and sessions**, tied to an entity such as a `user_id` rather than a `thread_id`. It is implemented with LangGraph's **store** abstraction (`BaseStore`, `InMemoryStore`, `PostgresStore`), which is a key-value/document store organized into **namespaces** (e.g. `("user", "u1", "details")`). LTM answers "what do we know about this user in general, regardless of which conversation we're in?" — facts like their name, preferences, or ongoing projects.

Both matter together: STM keeps a single conversation coherent turn-to-turn, while LTM lets an agent recognize and personalize for a user even in a brand-new thread. STM without management (trimming/deletion/summarization) grows unbounded and blows past context-window/token budgets; LTM turns per-thread knowledge into durable, queryable facts that outlive any single thread.

## Series Progression

The notebooks build up in two arcs:

1. **STM fundamentals and management** (notebooks 1-5): start with raw stateless invocation, add a checkpointer for persistence, then layer on three different strategies to keep the growing message history under control — trimming (token-budget windowing), deletion (explicit `RemoveMessage` pruning), and summarization (compress old turns into a running summary).
2. **LTM fundamentals and implementation** (notebooks 6-8): introduce the `Store` abstraction and its CRUD/search API (including semantic/embedding-based search) in isolation, then wire a store into an actual graph so nodes can read and write user-level memories, and finally swap the in-memory store for a Postgres-backed store for real persistence, paralleling the Postgres-backed checkpointer already shown for STM.

---

## Notebook-by-Notebook Breakdown

### 1. `1_stm.ipynb` — Basic STM (stateless baseline)

Introduces the minimal LangGraph chat graph: a single `call_model` node built on `MessagesState`, using `ChatOpenAI` (`gpt-3.5-turbo`), wired as `StateGraph(MessagesState)` with `START -> call_model -> END`.

- First calls `graph.invoke(...)` **without a checkpointer**. Each invocation is independent, so a follow-up "What is my name?" fails — the model has no memory of the prior turn. This demonstrates that a compiled graph alone has no persistence.
- Then rebuilds the graph and compiles it with `checkpointer=InMemorySaver()`, and calls `graph.invoke(..., config)` where `config = {"configurable": {"thread_id": "thread-1"}}`.
- With the checkpointer and a `thread_id`, the second invoke on the same thread correctly recalls "Nitish"; a different `thread_id` (`thread-2`) starts with a blank slate — proving STM is scoped per-thread.
- Shows `graph.get_state(config)` to inspect the accumulated `messages` list held for a thread.

**Mechanism introduced:** `MessagesState`, `StateGraph`, `InMemorySaver` checkpointer, `thread_id`-scoped `config` for state isolation.

### 2. `2_stm_persistence.ipynb` — Durable STM via Postgres checkpointer

Same minimal `call_model` graph as notebook 1, but swaps the in-memory checkpointer for `PostgresSaver` from `langgraph.checkpoint.postgres`, connecting to `DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres"`.

- Uses `PostgresSaver.from_conn_string(DB_URI)` as a context manager, calls `checkpointer.setup()` once to create the required tables, then compiles the graph with `builder.compile(checkpointer=checkpointer)`.
- Demonstrates that `thread-1`'s state (the name "Nitish") is recalled correctly, while `thread-2` starts fresh — the same thread-isolation behavior as notebook 1, but now backed by a real database instead of process memory, meaning state survives process restarts.
- Shows that state can be reloaded in an entirely new `with PostgresSaver.from_conn_string(...)` block/graph object and `graph.get_state(t1)` still pulls the prior conversation from Postgres — i.e., persistence is decoupled from the Python process's lifetime.

**Mechanism introduced:** `PostgresSaver` (Postgres-backed checkpointer) as a drop-in replacement for `InMemorySaver`, `checkpointer.setup()` for schema creation, and the same `postgres` container defined in `docker-compose.yml` (see below) as the backing database.

### 3. `3_stm_trimming.ipynb` — Trimming: token-budget windowing

Introduces message **trimming** to keep the prompt sent to the LLM within a token budget, while still growing the full history in the checkpointer.

- Imports `trim_messages` and `count_tokens_approximately` from `langchain_core.messages.utils`.
- Sets `MAX_TOKENS = 150`.
- Inside `call_model`, before invoking the model, calls:
  ```python
  messages = trim_messages(
      state["messages"],
      strategy="last",
      token_counter=count_tokens_approximately,
      max_tokens=MAX_TOKENS,
  )
  ```
  This produces a trimmed window containing only the most recent messages that fit under 150 tokens (`strategy="last"`), and only that trimmed window — not the full state history — is sent to `model.invoke(messages)`.
- Prints the token count of the trimmed window on every turn to make the effect visible; as the conversation grows, older turns fall out of what's sent to the model, while `graph.get_state(config)` shows the **full, untrimmed** history is still retained in the checkpointer.
- Notably, this causes an observed regression: after enough turns, "What is my name?" fails because the trimmed window no longer includes the turn where the user's name was given — illustrating the trade-off trimming makes (bounded token cost vs. potential loss of older but relevant facts).

**Mechanism introduced:** `trim_messages(strategy="last", token_counter=count_tokens_approximately, max_tokens=...)` applied inside the model-calling node, trimming what's sent to the LLM without truncating the persisted state.

### 4. `4_stm_deletion.ipynb` — Deletion: pruning state itself

Introduces **actual removal of messages from the persisted state**, using `RemoveMessage` from `langchain.messages`.

- Graph has two nodes: `chat` (calls the model) and `cleanup` (`delete_old_messages`), wired `START -> chat -> cleanup -> __end__`.
- `delete_old_messages` inspects `state["messages"]`; if there are more than 10 messages, it takes the earliest 6 (`msgs[:6]`) and returns:
  ```python
  {"messages": [RemoveMessage(id=m.id) for m in to_remove]}
  ```
  Because `MessagesState`'s reducer understands `RemoveMessage` objects (matched by message `id`), returning them causes LangGraph to delete those specific messages from the checkpointed state — unlike trimming, this is a permanent, structural mutation of stored history, not just a view used for one LLM call.
- After seven user turns, `graph.get_state(config)` confirms the message count in the checkpointer is reduced (8 messages remain instead of continuing to grow unbounded), verifying the earliest messages were physically removed rather than merely hidden from the prompt.

**Mechanism introduced:** `RemoveMessage(id=...)` returned from a dedicated cleanup node to physically prune old messages out of the checkpointed state (contrast with notebook 3's non-destructive trimming).

### 5. `5_stm_summarization.ipynb` — Summarization: compress history into a running summary

Introduces **summarization** as a third strategy: instead of discarding old context (deletion) or just hiding it from one call (trimming), old turns are compressed into a persistent natural-language `summary` field, and only that summary plus the most recent turns are kept verbatim.

- Extends state with a custom schema: `class ChatState(MessagesState): summary: str`.
- `summarize_conversation(state)`: builds a prompt that either summarizes from scratch or extends an `existing_summary` ("Extend the summary using the new conversation above."), invokes the LLM to produce a new summary string, and **also** deletes all but the last 2 messages via `RemoveMessage(id=m.id) for m in state["messages"][:-2]`. It returns both `{"summary": ..., "messages": [RemoveMessage(...), ...]}` in one node — so summarization here is implemented by combining an LLM summarization call with the same `RemoveMessage` deletion mechanism from notebook 4.
- `chat_node(state)`: if `state["summary"]` is non-empty, prepends it as a `SystemMessage`-like dict (`{"role": "system", "content": f"Conversation summary:\n{state['summary']}"}`) before the recent verbatim messages, then calls the model — so the summary substitutes for the messages that were deleted.
- `should_summarize(state)`: a conditional-edge predicate, `len(state["messages"]) > 6`, routing to the `summarize` node when the thread has grown past 6 messages, otherwise routing straight to `__end__`.
- Graph: `START -> chat`, then a conditional edge from `chat` keyed on `should_summarize` to either `summarize` or `__end__`, and `summarize -> __end__`.
- Demonstration: across four turns about quantum physics and Einstein, once the message count exceeds 6, the `summarize` node fires — the `summary` field is now populated (a running paragraph condensing the entire conversation so far) and `messages` collapses down to just the last 2 (the newest human/AI pair), confirmed via a `show_state()` helper that prints both `summary` and the remaining `messages`.

**Mechanism introduced:** a custom state field (`summary: str`) alongside `MessagesState`, a conditional edge (`add_conditional_edges` with a boolean-returning predicate) to trigger summarization only once history exceeds a threshold, an LLM-driven summarization node that both produces/extends a summary and prunes old messages via `RemoveMessage`, and injecting the summary back into the prompt as a system message on subsequent turns.

### 6. `6_ltm_basics.ipynb` — LTM store fundamentals (`InMemoryStore`)

Introduces LangGraph's **store** abstraction in isolation, without a graph — pure API exploration of `InMemoryStore` from `langgraph.store.memory`.

- Creates `store = InMemoryStore()` and a **namespace** tuple `("user", "u1")`, illustrating that stores organize memories hierarchically per-entity rather than per-thread.
- CRUD/search operations:
  - `store.put(namespace, key, value_dict)` — write a memory item (e.g. `store.put(("user","u1"), "1", {"data": "User likes pizza"})`).
  - `store.get(namespace, key)` — fetch a single item by key, returning an `Item(namespace=..., key=..., value=..., created_at=..., updated_at=...)`.
  - `store.search(namespace)` — list all items under a namespace (used to demonstrate that `("user","u2")` is isolated from `("user","u1")`).
- **Semantic search**: rebuilds the store with an embedding index — `InMemoryStore(index={'embed': embedding_model, 'dims': 1536})` using `OpenAIEmbeddings(model='text-embedding-3-small')`. After `put`-ing ten free-text memory facts about a user, calls `store.search(namespace, query="what is the user currently learning", limit=1)` and `store.search(namespace, query="what are user's preferences", limit=3)`, showing the store performs vector-similarity retrieval over stored memories rather than exact key lookup — the results returned are the semantically closest facts, not necessarily the most recently written ones.

**Mechanism introduced:** `InMemoryStore` as a namespaced key-value store with `put`/`get`/`search`, and its optional embedding-backed semantic search index (`index={'embed': ..., 'dims': ...}`) for natural-language memory retrieval, all independent of any `StateGraph`.

### 7. `7_ltm_implementation.ipynb` — Wiring an LTM store into a graph

Shows how to actually connect a store to a running graph via the `store=` compile argument and the `store: BaseStore` node parameter, progressing through four variants:

1. **Read-only memory ("Chatbot Reading Existing Memories")**: an `InMemoryStore` is pre-seeded (before the graph runs) with facts under `("user", "u1", "details")` (name, profession, preferences, project). A single `chat_node(state, config: RunnableConfig, store: BaseStore)` reads `config["configurable"]["user_id"]`, calls `store.search(user_details_ns)`, formats the retrieved facts into `{user_details_content}` inside a personalization-oriented `SYSTEM_PROMPT_TEMPLATE`, and answers using `SystemMessage(system_prompt) + state["messages"]`. The graph is compiled with `builder.compile(store=store)`, and `user_id` is threaded through `config["configurable"]`. This shows LTM being read but never updated by the conversation.

2. **Write-only memory ("Chatbot Creating New Memories")**: a `remember_only_node` uses a second, *structured-output* LLM (`memory_extractor = extractor_llm.with_structured_output(MemoryDecision)`, where `MemoryDecision` is a Pydantic model with `should_write: bool` and `memories: List[str]`) to decide what atomic facts from the latest user message are worth persisting, then `store.put(namespace, str(uuid.uuid4()), {"data": mem})` for each. The node returns a fixed "Noted." reply and does not use memory to answer — isolating the "extract and persist" concern.

3. **Deduplicated writes ("...without Duplication")**: extends the `MemoryDecision`/`MemoryItem` schema with an `is_new: bool` field. The extractor LLM is given the *existing* memories (`store.search(namespace)` joined into `user_details_content`) as context in `MEMORY_PROMPT`, and is instructed to mark `is_new=false` for anything that duplicates existing content; only items with `is_new=True` are written via `store.put`. This prevents the same fact ("likes Python") from being stored twice across turns.

4. **Merged workflow**: combines all of the above into a two-node graph — `remember` (extracts and deduplicates new facts into the store, returns `{}`, i.e. no message change) followed by `chat` (reads the store's current facts and answers with personalization), wired `START -> remember -> chat -> END`. This is the complete read+write LTM pattern: every turn first updates long-term memory, then answers using the now-current memory.

**Mechanism introduced:** the `store=` argument to `builder.compile(...)`, the `store: BaseStore` (and `config: RunnableConfig`) parameters injected into node functions, `config["configurable"]["user_id"]` as the key that scopes the memory namespace, structured-output extraction (`with_structured_output` + Pydantic models) to decide *what* to remember, and an `is_new` dedup flag to avoid storing repeated facts.

### 8. `8_ltm_postgres.ipynb` — Postgres-backed LTM store

Takes the "merged workflow" pattern from notebook 7 and swaps `InMemoryStore` for `PostgresStore` (`langgraph.store.postgres`), making long-term memory durable across process restarts — the LTM analogue of what notebook 2 did for STM checkpointing.

- Installs `psycopg[binary,pool]`, `langgraph`, `langgraph-checkpoint-postgres`.
- Identical `remember_node` / `chat_node` graph structure as notebook 7's merged workflow (structured-output `MemoryDecision`/`MemoryItem` extraction with `is_new` dedup, `SYSTEM_PROMPT_TEMPLATE` personalization).
- Connection: `DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"`, used as:
  ```python
  with PostgresStore.from_conn_string(DB_URI) as store:
      store.setup()   # run once to create the store's tables
      graph = builder.compile(store=store)
      ...
  ```
- Runs several turns for `user_id="u1"` ("Hi, my name is Nitish", "I teach AI on YouTube", "Explain GenAI simply") inside the `with` block, then prints `store.search(("user", "u1", "details"))` to show the extracted facts ("Nitish teaches AI on YouTube.", "User's name is Nitish.") are present.
- **Persistence check**: opens a **new** `PostgresStore.from_conn_string(DB_URI)` context (a fresh connection/object, no shared Python state with the previous block) and calls `store.search(("user", "u1", "details"))` again — the same two facts are returned, confirming the memories survived independently of the original store object/process, i.e. real durability backed by the Postgres database rather than by anything held in memory.

**Mechanism introduced:** `PostgresStore.from_conn_string(DB_URI)` as a context-managed, durable replacement for `InMemoryStore`; `store.setup()` to provision the store's schema/tables once; otherwise the exact same `store=` graph-compile wiring and node signatures as the in-memory version in notebook 7 — demonstrating the store backend is swappable without changing graph or node logic.

---

## `docker-compose.yml`

```yaml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: postgres
    ports:
      - "5442:5432"
```

This provisions a single service: a `postgres:16` container, exposing Postgres on host port `5442` (mapped to the container's default `5432`), with database `postgres` and user `postgres` created via the standard Postgres image environment variables.

This one container backs **both** of the Postgres-based notebooks in this series:

- Notebook 2 (`2_stm_persistence.ipynb`) connects `PostgresSaver.from_conn_string("postgresql://postgres:postgres@localhost:5442/postgres")` to it for STM **checkpointing** (conversation/thread state).
- Notebook 8 (`8_ltm_postgres.ipynb`) connects `PostgresStore.from_conn_string("postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable")` to the same database for LTM **store** persistence (cross-thread user memories).

Both notebooks call `.setup()` on their respective Postgres object (`checkpointer.setup()` / `store.setup()`) the first time they run against a fresh database, which creates the tables each abstraction needs (checkpoint tables for `PostgresSaver`, store tables for `PostgresStore`). Since both notebooks target the same `postgres` database inside the same container, the checkpointer tables and the store tables coexist side by side in that one database.

## Setup Notes (from the notebooks themselves)

- **Start Postgres before running notebooks 2 or 8**: `docker compose up -d` (or equivalent) in this directory, using the `docker-compose.yml` above. This exposes Postgres on `localhost:5442`.
- **Environment variables**: every notebook calls `load_dotenv()` at the top and instantiates `ChatOpenAI` / `OpenAIEmbeddings` with no explicit API key argument, meaning an OpenAI API key (conventionally `OPENAI_API_KEY`) must be available via a `.env` file or the environment for `load_dotenv()`/the LangChain OpenAI clients to pick up. No other environment variables are referenced in the notebook code.
- **Python packages**: install cells reference `langgraph`, `langgraph-checkpoint-postgres`, `psycopg[binary,pool]`, and `langchain-openai` (notebooks 2 and 8); other notebooks assume these plus `langchain-core` and `pydantic` are already installed (no explicit install cells beyond notebooks 2 and 8).
- **Run-once schema setup**: both `PostgresSaver.setup()` (notebook 2) and `PostgresStore.setup()` (notebook 8) are explicitly commented as needing to run only once per fresh database to create their respective tables; re-running is harmless but unnecessary once tables exist.
- **Connection string**: both Postgres notebooks hardcode `DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres"` (notebook 8 appends `?sslmode=disable`), matching the credentials and port defined in `docker-compose.yml`. No `.env`-driven connection string is used for the database in the notebooks as written.
