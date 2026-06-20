# LangGraph

A collection of LangGraph experiments, workflows, and chatbot implementations — exploring sequential workflows, persistence, tool calling, and MCP integration.

## 📂 Project Structure

```
LangGraph/
├── langgraph_backend.py              # Basic LangGraph chatbot backend
├── langgraph_database_backend.py     # Backend with SQLite checkpointing
├── langgraph_tool_backend.py         # Backend with tool calling (search, stock)
├── langgraph_mcp_backend.py          # Backend with MCP tools integration
├── MCP_backend.py                    # MCP server backend
├── MCP_frontend.py                   # Streamlit frontend for MCP
├── streamlit_frontend.py             # Basic Streamlit chat UI
├── streamlit_frontend_streaming.py   # Streaming responses UI
├── streamlit_frontend_threading.py   # Multi-thread chat UI
├── streamlit_frontend_database.py    # DB-backed chat UI
├── streamlit_frontend_tool.py        # Tool-using chat UI
├── mcp_config.json                   # MCP server config
├── browser_mcp.json                  # Browser MCP config
├── requirements.txt                  # Python dependencies
├── installed_packages.txt            # Installed packages snapshot
└── Sequential Workflows in LangGraph/
    ├── 0_test_installation.ipynb
    ├── 1_bmi_workflow.ipynb
    ├── 2_simple_llm_workflow.ipynb
    ├── 3_prompt_chaining.ipynb
    ├── 4_batsman_workflow.ipynb
    ├── 5_UPSC_essay_workflow.ipynb
    ├── 6_quadratic_equation_workflow.ipynb
    ├── 7_review_reply_workflow.ipynb
    ├── 8_X_post_generator.ipynb
    ├── 9_basic_chatbot.ipynb
    ├── 10_persistence.ipynb
    └── 11_tools.ipynb
```

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/SubhabrataBarik/LangGraph.git
cd LangGraph
```

### 2. Create a virtual environment
```bash
python -m venv env
```

### 3. Activate the virtual environment
```bash
# Windows (PowerShell)
.\env\Scripts\Activate.ps1

# Windows (CMD)
.\env\Scripts\activate.bat

# macOS / Linux
source env/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Set up environment variables
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 🧩 What's Inside

| Component | Description |
|-----------|-------------|
| **Basic Chatbot** | Stateful LangGraph chatbot with memory |
| **Streaming UI** | Real-time token streaming via Streamlit |
| **Threading UI** | Multi-conversation management |
| **Database UI** | SQLite-backed persistent chat history |
| **Tool Calling** | DuckDuckGo search + Alpha Vantage stock prices |
| **MCP Integration** | Model Context Protocol tools (arith + expense servers) |

## 📓 Notebooks

The `Sequential Workflows in LangGraph/` folder contains step-by-step Jupyter notebooks:

- **0**: Installation test
- **1**: BMI workflow (state + conditional logic)
- **2**: Simple LLM workflow
- **3**: Prompt chaining
- **4**: Batsman stats workflow
- **5**: UPSC essay grading workflow
- **6**: Quadratic equation solver
- **7**: Review reply generator
- **8**: X (Twitter) post generator
- **9**: Basic chatbot
- **10**: Persistence / checkpointing
- **11**: Tool calling

## 🛠️ Tech Stack

- **[LangGraph](https://langchain-ai.github.io/langgraph/)** — Stateful LLM workflows
- **[LangChain](https://www.langchain.com/)** — LLM orchestration
- **[Streamlit](https://streamlit.io/)** — Chat UI
- **[OpenAI](https://openai.com/)** — LLM provider
- **[MCP](https://modelcontextprotocol.io/)** — Tool integration protocol
- **SQLite** — Persistence

## 📝 License

This project is for learning and experimentation.

## 🙋 Author

**Subhabrata Barik** — [GitHub](https://github.com/SubhabrataBarik)