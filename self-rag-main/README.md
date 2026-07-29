# Self-RAG

A progressive, notebook-by-notebook implementation of **Self-RAG** (Self-Reflective Retrieval-Augmented Generation) built with LangGraph and LangChain. Each notebook (`self_rag_step1.ipynb` through `self_rag_step7.ipynb`) adds one new self-grading or control-flow capability on top of the previous step, culminating in a full pipeline with retrieval-need detection, document-relevance filtering, groundedness verification with a revise loop, answer-usefulness checking, and query rewriting. `self_rag_web.ipynb` is a separate branch of the same idea that swaps the "no relevant docs" dead end for a live web-search fallback.

## What Self-RAG Is

Self-RAG is a RAG pattern in which the generating model is also used to **critique its own pipeline** at multiple checkpoints, using structured (Pydantic/JSON) outputs to make each critique machine-routable inside a LangGraph graph. In this project the model is asked, at different nodes:

- **Should I retrieve at all** for this question, or can I answer from general knowledge? (`RetrieveDecision` / `decide_retrieval`)
- **Is each retrieved document actually relevant** to the question? (`RelevanceDecision` / `is_relevant`)
- **Is my generated answer supported by the retrieved context** (groundedness/hallucination check), or does it contain unsupported claims or interpretive language? (`IsSUPDecision` / `is_sup`)
- **Is the (grounded) answer actually useful**, i.e. does it address what was asked? (`IsUSEDecision` / `is_use`)

Each of these checks produces a structured decision that a conditional edge in the `StateGraph` uses to route execution — to skip retrieval, to discard irrelevant documents, to loop back and revise an under-supported answer, or to rewrite the retrieval query and try again. This is what distinguishes Self-RAG from plain RAG (retrieve-then-generate with no verification) — the pipeline can refuse to retrieve, refuse to answer from bad documents, revise its own output, and retry retrieval, all driven by the LLM's own structured self-assessment rather than a fixed, non-adaptive chain.

### Contrast with Corrective RAG (sibling project)

The sibling folder `corrective-rag-main` in this repository implements Corrective RAG (CRAG), a related but distinct pattern: CRAG typically grades retrieved documents and, when they are judged insufficient, **corrects** the retrieval set externally (e.g., by triggering a web search or query transformation to supplement/replace the documents) before generation ever happens. Self-RAG, as built here, centers instead on **post-generation self-critique** — it grades the retrieved docs too (`is_relevant`), but its distinguishing steps (`is_sup`, `is_use`) happen *after* an answer has already been generated, checking the answer itself for groundedness and usefulness and looping the generation (not just the retrieval) when it fails. `self_rag_web.ipynb` is the one notebook in this set that moves closer to CRAG's document-side correction, by adding a web-search fallback when vector-retrieved documents are judged irrelevant — see its section below.

## Common Architecture (steps 1-7)

All seven step notebooks share the same ingestion and retrieval scaffolding:

- **Corpus loading**: three PDFs are loaded with `PyPDFLoader` — `Company_Policies.pdf`, `Company_Profile.pdf`, `Product_and_Pricing.pdf` — and concatenated into one `docs` list.
- **Chunking**: `RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)`.
- **Embeddings / vector store**: `OpenAIEmbeddings(model="text-embedding-3-large")` into a `FAISS` vector store built with `FAISS.from_documents(chunks, embeddings)`.
- **Retriever**: `vector_store.as_retriever(search_kwargs={"k": 4})` — always top-4 documents.
- **LLM**: `ChatOpenAI(model="gpt-4o-mini", temperature=0)`, used both for generation and (via `llm.with_structured_output(...)`) for every grading decision.
- **Graph**: a LangGraph `StateGraph(State)` with a `TypedDict` state, built up incrementally across the steps, compiled with `g.compile()` into `app`, invoked via `app.invoke(initial_state, ...)`.

The self-grading nodes that recur across the steps (introduced progressively, detailed per-step below):

| Node | Structured schema | What it grades |
|---|---|---|
| `decide_retrieval` | `RetrieveDecision(should_retrieve: bool)` | Whether the question needs document retrieval at all, vs. answerable from general knowledge |
| `is_relevant` | `RelevanceDecision(is_relevant: bool)` | Per-document relevance of each of the top-4 retrieved chunks to the question |
| `is_sup` | `IsSUPDecision(issup: Literal["fully_supported","partially_supported","no_support"], evidence: List[str])` | Whether the generated answer's claims are grounded in the retrieved context (hallucination/groundedness check), including flagging unsupported interpretive/qualitative language |
| `is_use` | `IsUSEDecision(isuse: Literal["useful","not_useful"], reason: str)` | Whether the (grounded) answer actually addresses the question, independent of grounding |
| `rewrite_question` (step7) | `RewriteDecision(retrieval_query: str)` | Reformulates the original question into a retrieval-optimized query when the answer was judged not useful |

## Step-by-Step Breakdown

### `self_rag_step1.ipynb` — Retrieval-need routing only

- **State**: `question`, `need_retrieval`, `docs`, `answer`.
- **New node**: `decide_retrieval`, using `RetrieveDecision(should_retrieve: bool)` with a prompt instructing the LLM to return `True` when the question needs specific facts/citations, `False` for general knowledge, and to default to `True` when unsure.
- **Routing**: `route_after_decide` sends the graph to `retrieve` if `need_retrieval` is true, else to `generate_direct`.
- **Nodes**: `generate_direct` answers purely from the LLM's general knowledge (explicitly instructed not to assume external documents); `retrieve` calls the retriever and stores `docs`, but the retrieved documents are never used to generate an answer yet — the `retrieve` path just goes straight to `END` with `answer` left as-is. This step establishes only the retrieval-decision gate, no relevance grading or generation-from-context yet.
- **Graph shape**: `START -> decide_retrieval -> {generate_direct, retrieve} -> END`.

### `self_rag_step2.ipynb` — Adds document-relevance grading

- **State adds**: `relevant_docs: List[Document]`.
- **New node**: `is_relevant`, using `RelevanceDecision(is_relevant: bool)`, invoked once per retrieved document (loop over `state["docs"]`) with a prompt asking whether the document "contains information useful for answering the question." Documents graded relevant are collected into `relevant_docs`.
- **Graph change**: `retrieve -> is_relevant -> END` (unconditional edge to `END` after filtering — `relevant_docs` is computed and exposed but not yet used to generate a context-grounded answer; the retrieval path's final `answer` is still whatever was set earlier, demonstrated in the notebook by inspecting `result['docs']` vs `result['relevant_docs']` directly rather than by producing a new answer).

### `self_rag_step3.ipynb` — Adds context-grounded generation and a no-relevant-docs branch

- **State adds**: `context: str`.
- **New nodes**:
  - `generate_from_context`: joins `relevant_docs` page contents with `"\n\n---\n\n"` into `context`, and answers strictly from that context using a prompt that instructs the model to say `"No relevant document found."` if the context is insufficient and never use outside knowledge. If `relevant_docs` is empty, it directly returns `"No relevant document found."` without calling the LLM.
  - `no_relevant_docs`: a hardcoded fallback node returning `{"answer": "No relevant document found.", "context": ""}`.
- **New routing**: `route_after_relevance` sends the graph to `generate_from_context` if `relevant_docs` is non-empty, else to `no_relevant_docs`.
- **Graph shape**: `retrieve -> is_relevant -> {generate_from_context, no_relevant_docs} -> END`. This is the first step where retrieval actually produces a final grounded answer.

### `self_rag_step4.ipynb` — Adds IsSUP groundedness verification (single pass)

- **State adds**: `issup: Literal["fully_supported","partially_supported","no_support"]`, `evidence: List[str]`.
- **Relevance prompt tightened**: `is_relevant` is now framed as a "TOPIC level" judgment — a document is relevant if it discusses the same entity/topic as the question, without needing to fully answer it; the prompt explicitly defers exact-answer verification to "IsSUP", and defaults to `is_relevant=true` when unsure.
- **Generation prompt simplified**: `generate_from_context` no longer says "No relevant document found" on missing context; it returns `"No answer found."` instead, and the no-docs fallback node is renamed `no_answer_found`.
- **New node**: `is_sup`, using `IsSUPDecision(issup, evidence)`. The prompt defines three grades:
  - `fully_supported` — every claim is explicitly backed by context and the answer introduces no unsupported qualitative/interpretive words (examples given: "culture", "generous", "robust", "best-in-class").
  - `partially_supported` — facts are supported but the answer adds unsupported abstraction/interpretation.
  - `no_support` — key claims are not supported by context.
  It also asks for up to 3 quoted `evidence` snippets from the context.
- **Graph shape**: `generate_from_context -> is_sup -> END`. At this step `is_sup`'s verdict is computed and returned to the caller but there is **no loop back to revise the answer yet** — verification is diagnostic only.

### `self_rag_step5.ipynb` — Adds the IsSUP revise-and-reloop cycle

- **State adds**: `retries: int`.
- **New nodes**:
  - `accept_answer`: a no-op pass-through node (`return {}`) reached when the answer is accepted.
  - `revise_answer`: uses a "STRICT reviser" prompt that forces the model to output **only direct quotes from the context** as bullet points, with no explanatory language, and increments `retries`.
- **New routing**: `route_after_issup` with `MAX_RETRIES = 10` — returns `"accept_answer"` if `issup == "fully_supported"` **or** `retries >= MAX_RETRIES` (give up and accept whatever is there), otherwise `"revise_answer"`.
- **Loop edges**: `is_sup -> {accept_answer -> END, revise_answer -> is_sup}` — i.e. `revise_answer` loops back into `is_sup` for re-verification, forming the actual "generate → verify → revise → re-verify" self-reflection loop. The notebook invokes the graph with `config={"recursion_limit": 80}` to allow enough loop iterations.
- **Observed behavior in the notebook**: one query ("Do NexaAI plans include a free trial?") terminates with `issup: no_support` after retries are exhausted; another ("Describe NexaAI's company culture") converges to `fully_supported` after 1 revision, producing a quote-only bulleted answer.

### `self_rag_step6.ipynb` — Adds IsUSE usefulness grading after IsSUP

- **State adds**: `isuse: Literal["useful","not_useful"]`, `use_reason: str`.
- **New node**: `is_use`, using `IsUSEDecision(isuse, reason)`. Its prompt explicitly separates concerns from `is_sup`: "Do NOT re-check grounding (IsSUP already did that). Only check: 'Did we answer the question?'" — `useful` means the answer directly addresses the question; `not_useful` means it's generic/off-topic/background-only.
- **Routing change**: `route_after_issup`'s `"accept_answer"` branch is redirected to `is_use` instead of `END` (fully-supported or retry-exhausted answers now pass through a usefulness check before finishing). `route_after_isuse` sends `"useful"` to `END` and `"not_useful"` to `no_answer_found`.
- **Graph shape**: `is_sup -> {is_use (via accept_answer), revise_answer -> is_sup}`, then `is_use -> {END, no_answer_found}`. This is the first step where an answer can be grounded (`fully_supported`) yet still be discarded for being unhelpful.
- **Observed behavior**: a refund-policy query with no supporting document content ends with `issup: no_support` after 10 retries, then `isuse: not_useful`, final answer `"No answer found."`.

### `self_rag_step7.ipynb` — Adds retrieval-query rewriting and a second self-correction loop

- **State adds**: `retrieval_query: str`, `rewrite_tries: int`.
- **Retrieve node changed**: `retrieve` now uses `state.get("retrieval_query") or state["question"]` as the vector-search query, so a rewritten query can override the raw question on subsequent passes.
- **New node**: `rewrite_question`, using `RewriteDecision(retrieval_query: str)`. Its prompt asks for a short (6-16 word) query preserving key entities and adding likely policy/pricing keywords, given the original question, the previous retrieval query, and the (unhelpful) answer produced so far. It resets `docs`, `relevant_docs`, and `context` to empty and increments `rewrite_tries`.
- **Routing change**: `route_after_isuse` now has three outcomes governed by `MAX_REWRITE_TRIES = 3`: `"useful"` -> `END`; `not_useful` and `rewrite_tries < MAX_REWRITE_TRIES` -> `"rewrite_question"`; `not_useful` and `rewrite_tries >= MAX_REWRITE_TRIES` -> `"no_answer_found"` (give up).
- **New loop edge**: `rewrite_question -> retrieve`, so an unhelpful answer causes the whole retrieve → relevance-filter → generate → IsSUP → IsUSE pipeline to re-run with a better-targeted query, bounded by `MAX_REWRITE_TRIES`. Combined with step5's `MAX_RETRIES` cap on IsSUP revisions, the graph now has two independent, bounded self-correction loops: an inner groundedness-revision loop and an outer retrieval-rewrite loop.
- **Observed behavior**: with a seeded (deliberately wrong) initial `retrieval_query`, the pipeline still resolves "Describe NexaAI's company culture" correctly via the normal retrieve step (query passed as the direct question in that particular invocation), converging to `issup: fully_supported`, `isuse: useful` after 1 IsSUP revision.

### `self_rag_web.ipynb` — Web-search fallback instead of the "no docs" dead end

This notebook diverges from the step1-7 progression at the point reached in `self_rag_step3.ipynb` (it has `decide_retrieval`, `is_relevant`, `generate_from_context`, but **no IsSUP or IsUSE nodes and no retry/rewrite counters** — it is a separate, simpler branch focused specifically on adding external web search).

- **State**: `question`, `need_retrieval`, `docs`, `relevant_docs`, `context`, `answer`, plus **`web_query: str`** (new). Note: no `issup`/`isuse`/`retries` fields exist in this notebook's state.
- **New dependency**: `from langchain_community.tools.tavily_search import TavilySearchResults` (deprecated but used as-is; the notebook logs a `LangChainDeprecationWarning` recommending `langchain_tavily.TavilySearch` instead). Requires a Tavily API key available to the environment (loaded via `load_dotenv()`, consistent with the other notebooks' OpenAI key requirement).
- **New nodes**:
  - `rewrite_query_node`: uses `WebQuery(query: str)` structured output to turn the question into a short web-search keyword query, optionally appending a recency hint ("last 30 days") if the question implies it.
  - `web_search_node`: calls `tavily.invoke({"query": ...})` (`TavilySearchResults(max_results=5)`), and wraps each result's title/url/content into a `Document` with `metadata={"source": "web", "url": ..., "title": ...}`, replacing `state["docs"]`.
- **Routing change**: `route_after_relevance` no longer has a `no_relevant_docs`/`no_answer_found` terminal branch — instead, when no vector-retrieved documents are judged relevant, it routes to `"rewrite_query"`.
- **New loop edge**: `rewrite_query -> web_search -> is_relevant` — the same `is_relevant` node is reused to grade the freshly fetched web documents, and if they pass, execution proceeds to `generate_from_context` as normal. **This loop has no retry counter or maximum-iteration guard in the code** — if the web-search documents are also graded irrelevant, `route_after_relevance` would route back to `rewrite_query` again indefinitely (bounded in practice only by LangGraph's default recursion limit, which is not overridden in this notebook's `invoke` call).
- **Observed behavior**: for a live, current-events question ("Who won the Aus vs Zim World T20 match 2026 and who was the top scorer") that the internal PDFs cannot answer, the vector path fails relevance, the graph falls through to `rewrite_query -> web_search`, and all 5 Tavily results are judged relevant, producing a grounded answer citing ESPN/BBC/ICC sources with player-level statistics pulled directly from the web content.

## The `documents/` Corpus

Three PDFs form a synthetic company knowledge base for a fictional company, "NexaAI Solutions Pvt. Ltd.":

- **`Company_Policies.pdf`** — HR policies: equal-opportunity/anti-harassment statements, performance reviews, leave policy (annual/sick/casual/maternity leave day counts), and related workplace rules.
- **`Company_Profile.pdf`** — company overview (founded 2021, Bengaluru HQ, 85+ employees, operating regions), founder/leadership bios (Aarav Mehta as CEO & Founder, CTO, Head of Product, Head of Operations & HR, Head of Sales & Partnerships), vision.
- **`Product_and_Pricing.pdf`** — product portfolio (`NexaChat` internal knowledge assistant, `NexaInsight` business analytics/reporting, `NexaSupport` customer support automation, and other modules) plus pricing/trial terms (e.g., a 14-day free trial referenced in step5's example).
- **`documents/readme.md`** — present but effectively empty (1 byte).

This corpus supports company-internal Q&A such as: leadership/founder questions ("Who is the CEO of NexaAI?"), HR-policy questions (leave entitlements, notice period, culture), and product/pricing questions (free trial length, refund policy, feature lists per product). The notebooks demonstrate that some queries (e.g., refund policy specifics) are **not actually answerable from this corpus**, which is precisely what the self-grading nodes (`is_relevant`, `is_sup`, `is_use`) are shown catching — refusing to fabricate an answer and instead returning "No answer found" after exhausting revise/rewrite retries.

## Setup / Prerequisites

Based strictly on imports and calls present in the notebooks:

- **Environment variables** are loaded via `dotenv.load_dotenv()` in every notebook; an OpenAI API key must be available (used by both `ChatOpenAI` and `OpenAIEmbeddings`). `self_rag_web.ipynb` additionally requires a Tavily API key for `TavilySearchResults`.
- **LLM**: `langchain_openai.ChatOpenAI` with `model="gpt-4o-mini"`, `temperature=0`.
- **Embeddings**: `langchain_openai.OpenAIEmbeddings` with `model="text-embedding-3-large"`.
- **Vector store**: `langchain_community.vectorstores.FAISS`, built fresh from the three local PDFs on every notebook run (no persistence to disk is shown — each notebook reloads and re-embeds the documents in its own cells).
- **PDF loading**: `langchain_community.document_loaders.PyPDFLoader`, reading from a relative `./documents/` path — notebooks must be run with the working directory set to this project's root (`self-rag-main/`).
- **Text splitting**: `langchain_text_splitters.RecursiveCharacterTextSplitter` (`chunk_size=600`, `chunk_overlap=150`).
- **Orchestration**: `langgraph.graph.StateGraph`, `START`, `END`; steps 5-7 invoke the compiled graph with an increased `config={"recursion_limit": 80}` to accommodate the revise/rewrite loops.
- **Web search** (`self_rag_web.ipynb` only): `langchain_community.tools.tavily_search.TavilySearchResults` — flagged as deprecated in favor of `langchain_tavily.TavilySearch` in the notebook's own runtime warning output, but used as-is in the code.
