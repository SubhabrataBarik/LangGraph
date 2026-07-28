# Corrective RAG (CRAG) with LangGraph

This project is a progressive, notebook-by-notebook implementation of **Corrective RAG (CRAG)** using LangGraph, LangChain, OpenAI models, FAISS, and Tavily web search. Each notebook builds directly on the previous one, adding one architectural capability at a time, starting from plain RAG and ending at a full self-correcting retrieval pipeline that grades its own context and falls back to the web when the local corpus is insufficient.

## 1. What Corrective RAG Is, and Why It Improves on Plain RAG

Plain Retrieval-Augmented Generation (RAG) retrieves the top-`k` chunks for a query (via vector similarity) and hands them to an LLM unconditionally, trusting that the retriever did its job. This has three recurring failure modes:

- **Irrelevant retrieval**: the top-`k` chunks may simply not contain the answer (e.g., the corpus doesn't cover the topic, as with a question about current events against a static book corpus).
- **Partial/noisy retrieval**: a chunk may be topically related but contain mostly irrelevant sentences mixed with the one or two sentences that actually matter, diluting the context the generator sees.
- **Silent hallucination risk**: because the generation step never inspects retrieval quality, the LLM cannot distinguish "I have good evidence" from "I have garbage," and it may confidently answer from bad context, or hedge indiscriminately even when the context was fine.

Corrective RAG (CRAG), as implemented across these notebooks, addresses this by inserting an explicit **grading/evaluation step** after retrieval and before generation, plus **corrective actions** that are conditionally triggered based on the grade:

- If retrieved context is judged sufficient, use it directly (optionally refined to remove noise).
- If it is judged insufficient, discard it and issue a web search (optionally with a rewritten, search-optimized query) to compensate.
- If it is ambiguous, either flag it explicitly or (in the final notebook) combine internal and external evidence rather than trusting either alone.

The corrective loop is what distinguishes CRAG from plain RAG: retrieval quality is checked and acted upon before generation, rather than assumed.

## 2. Common Architecture Across the Notebooks

Every notebook shares the same data-loading and indexing scaffold, and builds a `langgraph.graph.StateGraph` with a `TypedDict` state that grows monotonically from notebook to notebook:

- **Document loading**: three PDFs (`book1.pdf`, `book2.pdf`, `book3.pdf`) are loaded with `PyPDFLoader` and concatenated (2123 pages total).
- **Chunking**: `RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)`, producing 6396 chunks. Each chunk's `page_content` is re-encoded (`encode("utf-8","ignore").decode("utf-8","ignore")`) to strip surrogate characters left over from PDF text extraction.
- **Embeddings and vector store**: `OpenAIEmbeddings(model="text-embedding-3-large")` indexed into an in-memory `FAISS.from_documents(...)` store, rebuilt fresh on every notebook run (no persistence to disk).
- **Retriever**: `vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})` — always top-4 similarity search.
- **LLM**: `ChatOpenAI(model="gpt-4o-mini", temperature=0)` is used uniformly for generation, grading, filtering, and query rewriting (different prompts/structured-output schemas per role).
- **Graph**: built with `StateGraph(State)`, wired with `START`/`END` and, from notebook 3 onward, `add_conditional_edges` for grade-based routing.
- **Environment**: `load_dotenv()` is called in every notebook, implying a `.env` file supplying API keys (see [Setup](#5-setup-and-prerequisites)).

The state schema and node graph accumulate as follows (see per-notebook sections for exact detail):

1. `retrieve` — vector similarity search for the question.
2. `eval_each_doc` (from notebook 3) — LLM-graded relevance score per retrieved chunk, producing a `verdict` (`CORRECT` / `INCORRECT` / `AMBIGUOUS`).
3. Conditional routing on `verdict` to either use the retrieved docs, trigger web search (from notebook 4), or handle ambiguity (from notebook 3, revised in notebook 6).
4. `rewrite_query` (from notebook 5) — LLM rewrites the question into a keyword-style web search query before the actual web search.
5. `web_search` (from notebook 4) — Tavily web search, wrapping results as `Document` objects.
6. `refine` (from notebook 2 onward) — CRAG's decompose → filter → recompose knowledge refinement, applied to whichever source of documents the verdict selected.
7. `generate` — final answer synthesis from the (refined) context.

## 3. Per-Notebook Breakdown

### `1_basic_rag.ipynb` — Plain RAG baseline

The starting point, with no grading or correction logic.

- **State** (`TypedDict`): `question: str`, `docs: List[Document]`, `answer: str`.
- **Nodes**: `retrieve` (calls `retriever.invoke(question)`), `generate` (joins all `docs[i].page_content` with `"\n\n"` into a single `context` string and runs it through a prompt/LLM chain).
- **Prompt**: system message instructs "Answer only from the context. If not in context, say you don't know."
- **Graph**: linear — `START -> retrieve -> generate -> END`. No conditional edges, no grading, no fallback.
- This notebook establishes the baseline that every subsequent notebook improves upon.

### `2_retrieval_refinement.ipynb` — CRAG's decompose/filter/recompose refinement

Introduces the "knowledge refinement" operation from the CRAG paper: instead of feeding raw retrieved chunks straight to the generator, the context is broken into sentence-level units, each is judged independently for relevance, and only the relevant ones are recomposed.

- **State additions**: `strips: List[str]` (all decomposed sentences), `kept_strips: List[str]` (sentences that passed the filter), `refined_context: str` (the kept sentences rejoined).
- **New node**: `refine`, inserted between `retrieve` and `generate`.
- **Decomposition**: `decompose_to_sentences(text)` — collapses whitespace with `re.sub(r"\s+", " ", text)`, splits on sentence boundaries with `re.split(r"(?<=[.!?])\s+", text)`, and drops any fragment of length ≤ 20 characters.
- **Filtering**: a structured-output Pydantic model `KeepOrDrop(BaseModel): keep: bool`, combined with a "strict relevance filter" system prompt ("Return keep=true only if the sentence directly helps answer the question... Use ONLY the sentence."), invoked once per sentence via `llm.with_structured_output(KeepOrDrop)`.
- **Recomposition**: kept sentences are rejoined with `"\n"` into `refined_context`.
- **Generation prompt** changed to consume `refined_context` instead of raw `context`, with instructions to answer only from "the provided bullets" and to say "I don't know based on the provided books" if bullets are empty/insufficient.
- **Graph**: `START -> retrieve -> refine -> generate -> END`. Still no grading or conditional routing — refinement runs unconditionally on whatever was retrieved.

### `3_retrieval_evaluator.ipynb` — Document-level grading and verdict routing

This is where CRAG's defining feature — a retrieval evaluator that grades documents and drives conditional control flow — is introduced.

- **State additions**: `good_docs: List[Document]` (docs judged at least weakly relevant), `verdict: str`, `reason: str`.
- **Thresholds**: module-level constants `UPPER_TH = 0.7`, `LOWER_TH = 0.3`.
- **Structured grading schema**: `DocEvalScore(BaseModel): score: float; reason: str`, produced per chunk via `doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)`. The grading system prompt instructs the LLM to score each chunk in `[0.0, 1.0]` for whether it alone is sufficient to answer the question, and to "be conservative with high scores."
- **New node**: `eval_each_doc_node`, which scores every doc in `state["docs"]` individually and computes a verdict:
  - `CORRECT` if **any** chunk scores `> UPPER_TH` (0.7). `good_docs` = all chunks scoring `> LOWER_TH`.
  - `INCORRECT` if **all** chunks score `< LOWER_TH` (0.3) — `good_docs` is empty.
  - `AMBIGUOUS` otherwise (mixed signals — no chunk above 0.7, but not all below 0.3).
- **Routing function**: `route_after_eval(state)` maps `CORRECT -> "refine"`, `INCORRECT -> "web_search"` (though at this stage the target node is actually named `fail`, a stand-in for the not-yet-implemented web search branch — see below), `AMBIGUOUS -> "ambiguous"`.
- **New terminal nodes**: `fail_node` returns `{"answer": f"FAIL: {reason}"}`; `ambiguous_node` returns `{"answer": f"Ambiguous: {reason}"}`. Neither performs any real corrective action yet — this notebook proves out the grading/routing mechanism before wiring in an actual web-search fallback.
- **`refine`** is updated to build its context from `good_docs` (the CORRECT-path documents) rather than all retrieved docs.
- **Graph**: `START -> retrieve -> eval_each_doc`, then conditional edges to `{refine, fail, ambiguous}`; `refine -> generate -> END`; `fail -> END`. (No explicit edge from `ambiguous` to `END` is added in this notebook — it is a dead-end node reached only via the conditional map.)

### `4_web_search_refinement.ipynb` — Real web search as the corrective action

Replaces the `fail` placeholder with an actual web-search fallback, making the "corrective" part of CRAG concrete for the `INCORRECT` case.

- **State addition**: `web_docs: List[Document]`.
- **New tool**: `TavilySearchResults(max_results=5)` from `langchain_community.tools.tavily_search` (flagged as deprecated in favor of `langchain_tavily.TavilySearch`, but still what's used here).
- **New node**: `web_search_node` — calls `tavily.invoke({"query": question})` using the raw, un-rewritten question, and wraps each result (`title`, `url`, `content`/`snippet`) into a `Document` with a formatted `TITLE/URL/CONTENT` string as `page_content` and `{"url", "title"}` as metadata.
- **`refine`** is updated to branch on `state["verdict"]`: if `"CORRECT"`, refine `good_docs`; otherwise (i.e. for the web-search path), refine `web_docs`. This is the first point where `refine` operates over externally sourced content rather than only the vector store's output.
- **Routing update**: `route_after_eval` now sends `INCORRECT -> "web_search"` for real (mapped to the actual `web_search_node`), and `web_search -> refine` is added as a new edge so the web results also pass through the decompose/filter/recompose refinement step before generation.
- **`ambiguous` path**: still terminal, but now has an explicit `g.add_edge("ambiguous", END)`.
- **Graph**: `START -> retrieve -> eval_each_doc` --conditional--> `{refine, web_search, ambiguous}`; `web_search -> refine -> generate -> END`; `ambiguous -> END`.

### `5_query_rewrite.ipynb` — Query rewriting before web search

Adds a query-rewriting step so the web search tool receives a search-engine-friendly query rather than the user's raw natural-language question.

- **State addition**: `web_query: str`.
- **Structured schema**: `WebQuery(BaseModel): query: str`.
- **New node**: `rewrite_query_node`, using `rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)`. The rewrite prompt instructs the LLM to: keep the query short (6-14 words), append a recency constraint like "(last 30 days)" if the question implies recency (e.g. "recent", "latest", "last week"), never answer the question itself, and return JSON with a single `query` key.
- **`web_search_node`** updated to consume `state.get("web_query") or state["question"]` (falls back to the raw question if rewriting produced nothing).
- **Routing update**: `route_after_eval` now sends `INCORRECT -> "rewrite_query"` (instead of directly to `web_search`).
- **Graph**: `START -> retrieve -> eval_each_doc` --conditional--> `{refine, rewrite_query, ambiguous}`; `rewrite_query -> web_search -> refine -> generate -> END`; `ambiguous -> END` (verdict `AMBIGUOUS` is still a dead end here, just like in notebook 4).
- Example run shows the rewrite turning `"Recent AI news"` into `"recent AI news last 30 days"`.

### `6_ambiguous.ipynb` — Hybrid internal+web handling for the ambiguous case

The final notebook removes `ambiguous` as a dead-end/terminal branch and instead treats it as a case that also deserves a corrective web-search action, while additionally blending internal and external evidence rather than picking one source exclusively.

- **State**: same fields as notebook 5 (`good_docs`, `verdict`, `reason`, `strips`, `kept_strips`, `refined_context`, `web_query`, `web_docs`), no new fields — the change here is behavioral, not schema-level.
- **`refine`** is rewritten with three explicit branches keyed on `state["verdict"]`:
  - `CORRECT` -> use `good_docs` only.
  - `INCORRECT` -> use `web_docs` only.
  - `AMBIGUOUS` (the `else` branch) -> use `good_docs + web_docs` concatenated — the only notebook where internal retrieval and web search results are combined into one context for refinement.
- **Routing simplified**: `route_after_eval` now only distinguishes two cases — `CORRECT -> "refine"`, and everything else (`INCORRECT` or `AMBIGUOUS`) `-> "rewrite_query"`. There is no longer a separate `ambiguous_node` or `fail_node` in the graph; ambiguous questions flow through the same rewrite → web search → refine → generate pipeline as clearly-incorrect ones, but end up with hybrid (internal + web) context instead of web-only context.
- **Graph**: `START -> retrieve -> eval_each_doc` --conditional--> `{refine, rewrite_query}`; `rewrite_query -> web_search -> refine`; `refine -> generate -> END`. This is the most complete pipeline in the series: grade → route → (optionally rewrite + web search) → refine (source-dependent on verdict) → generate.
- Example run: the question "Batch normalization vs layer normalization" is graded `AMBIGUOUS`, triggers a query rewrite (`web_query` unchanged from the question in this case) and a web search, and the final answer is generated from the combined refined context.

## 4. The `documents/` Folder

`documents/` contains the static corpus indexed by every notebook:

- `book1.pdf`, `book2.pdf`, `book3.pdf` — three PDF textbooks covering machine learning and deep learning topics. Based on the content surfaced in retrieved chunks across the notebooks, the corpus spans topics such as deep learning fundamentals and representation learning, convolutional layers/pooling and CNN architectures (e.g. LeNet-5), regularization and dataset augmentation, bias-variance decomposition and generalization error, and deep/recurrent neural network training. The notebooks do not name the books explicitly; this description is inferred from the chunk text shown in each notebook's outputs.
- `readme.md` — a one-line placeholder file containing only the word "Books".
- All three PDFs are loaded, concatenated (2123 pages total), chunked (6396 chunks at `chunk_size=900`/`chunk_overlap=150`), and embedded into an in-memory FAISS index identically in every notebook — there is no incremental indexing or persistence across notebooks; each notebook rebuilds the index from these PDFs from scratch.

Note: a nested `corrective-rag-main/corrective-rag-main/` directory exists alongside these files as a leftover duplicate from an accidental double-extraction. It contains copies of the same notebooks and is not part of the project structure described here.

## 5. Setup and Prerequisites

Based strictly on imports and calls present in the notebooks:

- **Environment variables**: all notebooks call `load_dotenv()`, implying a `.env` file. The following API keys are required by the libraries used:
  - An OpenAI API key (`OPENAI_API_KEY`), consumed implicitly by `ChatOpenAI` and `OpenAIEmbeddings`.
  - A Tavily API key (`TAVILY_API_KEY`), consumed implicitly by `TavilySearchResults`, required starting from `4_web_search_refinement.ipynb` onward.
- **Models used**:
  - Chat/generation/grading/rewriting LLM: `gpt-4o-mini` via `ChatOpenAI(temperature=0)`.
  - Embeddings: `text-embedding-3-large` via `OpenAIEmbeddings`.
- **Vector store**: FAISS (`langchain_community.vectorstores.FAISS`), in-memory only, rebuilt on each notebook run via `FAISS.from_documents(...)` — no on-disk persistence or reuse of a saved index.
- **Web search tool**: `langchain_community.tools.tavily_search.TavilySearchResults` (used from notebook 4 onward). The notebooks emit a `LangChainDeprecationWarning` recommending migration to `langchain_tavily.TavilySearch`, but do not make that change themselves.
- **PDF loading**: `langchain_community.document_loaders.PyPDFLoader`, requiring the three PDFs to be present at `./documents/book1.pdf`, `./documents/book2.pdf`, `./documents/book3.pdf` relative to the notebook's working directory.
- **Key Python packages** implied by imports: `langchain`, `langchain-community`, `langchain-openai`, `langchain-text-splitters`, `langgraph`, `pydantic`, `python-dotenv`, `faiss` (via the FAISS vector store integration), and a PDF backend for `PyPDFLoader` (e.g. `pypdf`).
