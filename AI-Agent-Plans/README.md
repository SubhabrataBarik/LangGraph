# AI-Agent-Plans: The Blog Writing Agent ("bwa")

## What "bwa" Stands For

`bwa` stands for **Blog Writing Agent**. This is confirmed directly in the code:

- `bwa_backend.py` contains the header comment:
  ```
  # Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
  ```
- `bwa_frontend.py` sets the Streamlit page title to `"Blog Writing Agent"` (`st.title("Blog Writing Agent")`) with page config `page_title="LangGraph Blog Writer"`.

The agent is a LangGraph-based multi-node pipeline that turns a single topic string into a complete, saved Markdown blog post — optionally grounded in live web research (via Tavily) and optionally illustrated with generated images (via Gemini). It uses an **orchestrator/worker (map-reduce) pattern**: one LLM call plans the blog as a list of sections, then one LLM call per section writes that section in parallel, and a final reducer stitches everything into one document.

## Development Arc Across the 5 Notebooks

The notebooks form a strictly incremental build-up, each one keeping the orchestrator → fan-out-workers → reducer skeleton from the previous step and adding one new capability:

| # | Notebook | Adds |
|---|----------|------|
| 1 | `1_bwa_basic.ipynb` | The foundational map-reduce graph: orchestrator plans, workers write sections in parallel via `Send`, reducer concatenates and saves to disk. Minimal prompts, minimal schema. |
| 2 | `2_bwa_improved_prompting.ipynb` | Same graph topology, but the `Plan`/`Task` schemas and system prompts are substantially richer (audience, tone, goals, word-count targets, bullet constraints, a mandatory `common_mistakes` section, and detailed "technical quality bar" instructions for both planning and writing). No new nodes — purely a prompt/schema engineering pass. |
| 3 | `3_bwa_research.ipynb` | Introduces web research: a new `router` node classifies the topic into `closed_book` / `hybrid` / `open_book` mode and (if needed) emits search queries; a new `research` node calls Tavily and synthesizes results into structured `EvidenceItem`s; the orchestrator and workers are updated to consume that evidence and cite sources. |
| 4 | `4_bwa_research_fine_tuned.ipynb` | Refines the research behavior from #3 rather than adding nodes: introduces `as_of` / `recency_days` state for time-awareness, a hard recency filter that discards evidence outside the freshness window for `open_book` mode, richer `RouterDecision` (adds a `reason` field and `max_results_per_query`), and forced `blog_kind="news_roundup"` enforcement so the plan can't drift into a tutorial when the mode is news-oriented. |
| 5 | `5_bwa_image.ipynb` | Adds image generation. The reducer becomes a 3-node **subgraph** (`merge_content` → `decide_images` → `generate_and_place_images`): an LLM decides whether diagrams would help (max 3), inserts `[[IMAGE_N]]` placeholders and prompts, and a Gemini image-generation call (`gemini-2.5-flash-image` via `google-genai`) renders each image to `images/`, replacing placeholders with Markdown image embeds (with a graceful text-fallback block if generation fails). |

`bwa_backend.py` is the productionized descendant of notebook 5 (same graph, hardened with `.env` loading and defensive Tavily-key checking), and `bwa_frontend.py` wraps it in a Streamlit UI.

---

## Notebook 1: `1_bwa_basic.ipynb`

**State (`State` TypedDict):**
- `topic: str`
- `plan: Plan`
- `sections: Annotated[List[str], operator.add]` — reducer-annotated list; LangGraph auto-concatenates each worker's returned section list
- `final: str`

**Schemas:**
- `Task(id, title, brief)` — a bare-bones section spec.
- `Plan(blog_title, tasks: List[Task])`

**Nodes:**
- `orchestrator` — single `SystemMessage` instructing the LLM (`ChatOpenAI(model="gpt-4.1-mini")`) with `.with_structured_output(Plan)` to "Create a blog plan with 5-7 sections."
- `fanout` — a conditional-edge function returning a list of `Send("worker", {...})` objects, one per task (LangGraph's map/fan-out primitive).
- `worker` — writes one Markdown section per task with a single-line system prompt ("Write one clean Markdown section.").
- `reducer` — joins all sections under an `# {title}` heading and writes the result to `<slugified_title>.md` on disk.

**Graph:** `START → orchestrator → (fanout) → worker → reducer → END`.

No tools are bound; this is a pure LLM planning/writing pipeline with no external data access.

---

## Notebook 2: `2_bwa_improved_prompting.ipynb`

**Graph topology is identical to Notebook 1** — same 3 nodes, same edges. The entire delta is in prompting and schema strictness.

**Schema changes:**
- `Task` gains: `goal` (one-sentence outcome statement), `bullets` (3–5 items, `min_length`/`max_length` enforced by Pydantic), `target_words` (120–450), and `section_type` (a `Literal` enum of `intro/core/examples/checklist/common_mistakes/conclusion`, with the instruction that `common_mistakes` must appear **exactly once**).
- `Plan` gains `audience` and `tone` fields.

**Prompting changes:**
- The orchestrator's system prompt becomes a multi-paragraph spec: "senior technical writer and developer advocate" persona, hard requirements (section count, bullet count, word count, exactly-one `common_mistakes`), an explicit structural ordering guide (problem → intuition → approach → implementation → trade-offs → testing/observability → conclusion), and a ban on vague bullets like "Explain X."
- The worker's system prompt adds hard constraints (cover all bullets in order, stay within ±15% of `target_words`, output only the section body) and a "technical quality bar" (must include a code snippet, example I/O, checklist, or text-described diagram; call out trade-offs and edge cases).

This notebook demonstrates that capability growth doesn't always mean new nodes — much of an agent's quality comes from schema constraints (Pydantic `Field` validation) and system-prompt specificity alone.

---

## Notebook 3: `3_bwa_research.ipynb`

This is where the agent gains tool use. Two nodes are added and the state grows substantially.

**New/changed schemas:**
- `EvidenceItem(title, url, published_at, snippet, source)`
- `RouterDecision(needs_research: bool, mode: Literal["closed_book","hybrid","open_book"], queries: List[str])`
- `EvidencePack(evidence: List[EvidenceItem])`
- `Task` gains `tags`, `requires_research`, `requires_citations`, `requires_code` (boolean flags workers use to decide whether to cite sources / include code).
- `Plan` gains `blog_kind` (`explainer/tutorial/news_roundup/comparison/system_design`) and `constraints`.

**State additions:** `mode`, `needs_research`, `queries: List[str]`, `evidence: List[EvidenceItem]`. `sections` changes from `List[str]` to `List[tuple[int, str]]` (task id + markdown) so the reducer can restore deterministic ordering after parallel fan-out.

**New nodes:**
- `router_node` — classifies the topic into `closed_book` (evergreen, no research), `hybrid` (evergreen but wants fresh examples), or `open_book` (volatile/news, e.g. weekly roundups, pricing, "latest"); if research is needed it also produces 3–10 scoped search queries. A conditional edge (`route_next`) sends state to `research` or straight to `orchestrator`.
- `research_node` — for each query, calls `_tavily_search()`, which wraps `langchain_community.tools.tavily_search.TavilySearchResults` (`max_results=6` per query). Raw results are then passed to the LLM with `.with_structured_output(EvidencePack)` to normalize/dedupe by URL into `EvidenceItem` objects (dates kept only if explicitly present — the prompt forbids guessing dates).

**Tool bound:** `TavilySearchResults` (LangChain community Tavily wrapper) — the only external tool in the whole project.

**Orchestrator/worker changes:** The orchestrator prompt now branches its grounding rules by `mode` (closed_book stays evergreen; hybrid must mark sections `requires_research/requires_citations`; open_book forces `blog_kind="news_roundup"` and forbids tutorial drift). The worker prompt adds a citation policy: in `open_book` mode every event/company/model claim must link to a supplied evidence URL or say "Not found in provided sources," and `requires_code` triggers a mandatory code snippet.

**Graph:** `START → router → (route_next) → [research → orchestrator | orchestrator] → (fanout) → worker → reducer → END`.

---

## Notebook 4: `4_bwa_research_fine_tuned.ipynb`

Same 5 nodes and graph shape as Notebook 3. This is a **refinement pass on the research subsystem**, not a new capability.

**Key changes:**
- **Time-awareness:** state adds `as_of: str` (ISO date) and `recency_days: int`. The router now derives `recency_days` from the chosen mode: `open_book` → 7 days, `hybrid` → 45 days, `closed_book` → 3650 days (effectively unbounded).
- **`RouterDecision`** gains a `reason` field (rationale for the mode choice) and `max_results_per_query`.
- **Hard recency filter:** `research_node` adds `_iso_to_date()` and, specifically for `open_book` mode, drops any evidence item whose `published_at` can't be parsed or falls before `as_of - recency_days` — so stale sources are mechanically excluded, not just discouraged by prompt.
- **Forced genre consistency:** the orchestrator now explicitly forces `plan.blog_kind = "news_roundup"` in code (not just prompt instruction) whenever `mode == "open_book"`, closing a gap where notebook 3 only asked the LLM to set it.
- The `run()` helper function is expanded to print a structured debug summary (mode, queries, evidence count/sample, task count) after each invocation — better observability for tuning the router/research behavior.

Both `bwa_backend.py` and notebook 5 build on this fine-tuned version of the research/recency logic (they retain `as_of`/`recency_days` in state, though `bwa_backend.py`'s router itself uses the simpler notebook-3-style prompt while keeping the recency-day derivation).

---

## Notebook 5: `5_bwa_image.ipynb`

Adds visual content generation. The router/research/orchestrator/worker nodes are unchanged from notebook 3's logic (state still carries `as_of`/`recency_days` fields but the router prompt doesn't reference them as heavily as notebook 4). The major addition is that **the reducer is replaced by a compiled subgraph**.

**New schemas:**
- `ImageSpec(placeholder, filename, alt, caption, prompt, size, quality)` — one entry per proposed image.
- `GlobalImagePlan(md_with_placeholders, images: List[ImageSpec])`

**State additions:** `merged_md`, `md_with_placeholders`, `image_specs: List[dict]`.

**Reducer subgraph (`reducer_graph` → compiled as `reducer_subgraph`, plugged into the main graph as the `"reducer"` node):**
1. `merge_content` — joins ordered sections into one Markdown document with the `# {title}` heading (same as prior reducers, but without saving to disk yet).
2. `decide_images` — LLM call with `.with_structured_output(GlobalImagePlan)`; system prompt caps images at 3, requires each to "materially improve understanding," and requires exact `[[IMAGE_1]]`/`[[IMAGE_2]]`/`[[IMAGE_3]]` placeholders. If no images are warranted, `md_with_placeholders` must equal the input and `images=[]`.
3. `generate_and_place_images` — for each `ImageSpec`, calls `_gemini_generate_image_bytes(prompt)`, which uses `google.genai.Client` with model `"gemini-2.5-flash-image"` and `response_modalities=["IMAGE"]` (requires `GOOGLE_API_KEY` and the `google-genai` package). Generated bytes are written under `images/<filename>`; each placeholder is replaced with a Markdown image embed (`![alt](images/filename)` + italic caption). If generation raises an exception, the placeholder is replaced with a `> **[IMAGE GENERATION FAILED]**` blockquote containing the caption, alt text, prompt, and error — so the document stays usable even without image credentials. The final document is written to `<blog_title>.md`.

**Graph:** identical top-level shape to notebook 3/4, but `"reducer"` now points at `reducer_subgraph` instead of a single function node:
`START → router → (route_next) → [research → orchestrator | orchestrator] → (fanout) → worker → reducer(subgraph) → END`.

---

## `tavily_test.ipynb`

A minimal, standalone smoke test for the Tavily integration used by notebooks 3–5 and `bwa_backend.py`. It:

1. Resolves the project root one level above the notebook's directory and calls `load_dotenv()` against `<root>/.env`, then asserts `TAVILY_API_KEY` is present — verifying the key is discoverable regardless of which folder the kernel starts in.
2. Instantiates `TavilySearchResults(max_results=2)` directly (no LangGraph, no LLM) and invokes it with a plain-text query (`"ChatGPT version releases and updates from 2022 to 2026"`).
3. Prints each result's `content` field.

This confirms the Tavily API key is valid and that `TavilySearchResults.invoke({"query": ...})` returns a list of dicts with `title`, `url`, `content`, and a relevance `score` — the exact shape `_tavily_search()` in the research nodes normalizes. Note: an unrelated LangSmith tracing error (`403 Forbidden` posting to `api.smith.langchain.com/runs/multipart`) appears in the output; it's a background tracing/telemetry failure, not a Tavily error, and does not affect the search result returned.

---

## `bwa_backend.py` and `bwa_frontend.py`: The Productionized App

### `bwa_backend.py`

This is the deployable version of the notebook 5 graph, structured as a plain `.py` module so it can be imported. Differences from the notebook:

- Calls `load_dotenv()` at import time (loads `.env` from the working directory).
- `_tavily_search()` is defensive: it checks `os.getenv("TAVILY_API_KEY")` and returns `[]` immediately if unset (rather than raising), and wraps the whole Tavily call in `try/except Exception: return []` — so a missing/broken Tavily key degrades to `closed_book`-style behavior instead of crashing the graph.
- Adds `_safe_slug()` (regex-based filename sanitizer: lowercases, strips non `[a-z0-9 _-]` characters, collapses whitespace to underscores) used when writing both the final `.md` file — this fixes a rough edge in the notebooks, where `Plan.blog_title` was written directly as a filename without sanitization.
- Otherwise reproduces notebook 5's schemas (`Task`, `Plan`, `EvidenceItem`, `RouterDecision`, `EvidencePack`, `ImageSpec`, `GlobalImagePlan`), `State` TypedDict, and all 5 nodes (`router_node`, `research_node`, `orchestrator_node`, `worker_node`, and the `merge_content → decide_images → generate_and_place_images` reducer subgraph), compiled into the module-level `app` object that the frontend imports.

**Graph exposed:** `app = g.compile()` with nodes `router → research?/orchestrator → worker (fanned out) → reducer(subgraph) → END`.

### `bwa_frontend.py`

A Streamlit application (`st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")`, `st.title("Blog Writing Agent")`) that drives `bwa_backend.app`:

- **Sidebar:** a topic `st.text_area`, an `as_of` date picker (`st.date_input`, defaults to today), and a "Generate Blog" button. Below that, a "Past blogs" browser that lists every `*.md` file in the working directory (newest first via mtime), lets the user pick one by radio button (parsing the first `# ` heading as a display title), and load it back into `st.session_state["last_out"]` for viewing (with `plan`/`evidence`/`image_specs` empty since old files don't carry that metadata).
- **Execution:** `try_stream()` first attempts `app.stream(inputs, stream_mode="updates")`, falling back to `stream_mode="values"`, and finally to a plain `app.invoke()` if streaming isn't supported — then always calls `app.invoke(inputs)` again to get the authoritative final state. Node transitions and a live JSON summary (mode, queries, evidence count, task count, images, sections done) are shown inside an `st.status(...)` block while the graph runs.
- **Results, rendered in 5 tabs:**
  - **Plan** — title/audience/tone/blog_kind plus a `pandas.DataFrame` of tasks (id, title, target words, requires_research/citations/code flags, tags) and a raw JSON expander.
  - **Evidence** — a DataFrame of `EvidenceItem`s (title, published_at, source, url), or an info message if none were returned (closed_book mode or missing Tavily key).
  - **Markdown Preview** — renders the final Markdown via a custom `render_markdown_with_local_images()` (regex-splits on `![alt](src)` image syntax, resolves local `images/...` paths with `st.image`, and picks up an italic caption line immediately following an image). Offers a **Download Markdown** button and a **Download Bundle (MD + images) zip** built by `bundle_zip()`.
  - **Images** — displays the `image_specs` plan JSON and every file under `images/`, with a zip download (`images_zip()`).
  - **Logs** — a scrolling text area of the last 80 logged stream events, persisted across reruns in `st.session_state["logs"]`.

**How they connect:** `bwa_frontend.py` does `from bwa_backend import app` and treats the compiled LangGraph `app` as a black box — it only calls `.stream()`/`.invoke()` on it and reads back the `State` TypedDict fields (`plan`, `evidence`, `final`, `image_specs`, etc.) to populate the UI. All blog-writing logic (routing, research, planning, section writing, image generation) lives entirely in the backend module.

---

## Setup Notes (from code, not inferred)

- **Environment variables required:**
  - `TAVILY_API_KEY` — required for the `research_node` / `TavilySearchResults` tool used in notebooks 3–5 and `bwa_backend.py`. Loaded via `python-dotenv`'s `load_dotenv()`. Without it, `bwa_backend.py`'s `_tavily_search()` silently returns no results (evidence stays empty) rather than failing; `tavily_test.ipynb` will raise an `AssertionError` if it's missing.
  - `GOOGLE_API_KEY` — required by `_gemini_generate_image_bytes()` (notebook 5 and `bwa_backend.py`) for image generation via `google.genai.Client`. If unset, a `RuntimeError` is raised per image, which is caught and turned into a "IMAGE GENERATION FAILED" placeholder block in the final Markdown — the app still produces a usable text-only blog.
  - An OpenAI-compatible API key is implicitly required by `ChatOpenAI(model="gpt-4.1-mini")` (via `langchain_openai`), though the exact env var (`OPENAI_API_KEY`) is never referenced explicitly in this code — it's read by the `langchain_openai`/`openai` SDK itself.
  - `.env` file location: `tavily_test.ipynb` looks for it one directory above the notebook (i.e., the repository root); `bwa_backend.py` calls `load_dotenv()` with no path argument, so it depends on the process's working directory containing (or having on its dotenv search path) the `.env` file.
- **Python packages referenced in code:** `langgraph`, `langchain-openai`, `langchain-core`, `langchain-community` (for `TavilySearchResults`), `pydantic`, `python-dotenv`, `google-genai` (imported as `from google import genai`), `streamlit`, `pandas`.
- **Required package install for images:** the code comment in `_gemini_generate_image_bytes()` states `pip install google-genai`.
- **Running the frontend:** since `bwa_frontend.py` is a Streamlit script that imports the compiled graph from `bwa_backend.py`, it is run with Streamlit's CLI (e.g. `streamlit run bwa_frontend.py`) from within the `AI-Agent-Plans` directory so that relative paths (`images/`, generated `*.md` files, `.env` lookup) resolve correctly.
- **Output artifacts:** every run writes the finished blog as `<slugified_blog_title>.md` in the current working directory, and any generated images under an `images/` subdirectory — both of which the frontend's "Past blogs" and "Images" tabs scan for at runtime.
