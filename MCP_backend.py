# # backend.py
# from langgraph.graph import StateGraph, START, END
# from typing import TypedDict, Annotated
# from langchain_core.messages import BaseMessage, HumanMessage
# from langchain_openai import ChatOpenAI
# from langgraph.checkpoint.sqlite import SqliteSaver
# from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode, tools_condition
# from dotenv import load_dotenv
# import sqlite3
# import os
# import asyncio

# # Import MCP-related libraries
# from langchain_mcp_adapters.tools import load_mcp_tools
# from mcp_use import MCPClient

# load_dotenv()

# llm = ChatOpenAI()

# # Global variables for tools and chatbot
# _tools = []
# _llm_with_tools = None
# _chatbot = None
# _checkpointer = None

# async def get_mcp_tools():
#     """Initialize MCP tools asynchronously"""
#     try:
#         mcp_client = MCPClient.from_config_file(
#             os.path.join(os.path.dirname(__file__), "mcp_config.json")
#         )
#         mcp_tools = await load_mcp_tools(mcp_client)
#         print(f"Loaded {len(mcp_tools)} MCP tools.")
#         return mcp_tools
#     except Exception as e:
#         print(f"Failed to load MCP tools: {e}")
#         return []

# def initialize_backend():
#     """Initialize the backend synchronously for Streamlit"""
#     global _tools, _llm_with_tools, _chatbot, _checkpointer
    
#     _tools = []
#     _llm_with_tools = llm.bind_tools(_tools)
    
#     conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
#     _checkpointer = SqliteSaver(conn=conn)
    
#     _chatbot = create_graph()
#     return _chatbot, _checkpointer

# async def async_initialize_tools():
#     """Async initialization of MCP tools"""
#     global _tools, _llm_with_tools, _chatbot
    
#     _tools = await get_mcp_tools()
#     _llm_with_tools = llm.bind_tools(_tools)
#     _chatbot = create_graph()
    
#     return _tools

# class ChatState(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]

# def chat_node(state: ChatState):
#     """LLM node that may answer or request a tool call."""
#     messages = state["messages"]
#     response = _llm_with_tools.invoke(messages)
#     return {"messages": [response]}

# def create_graph():
#     """Create the LangGraph workflow"""
#     tool_node = ToolNode(_tools)
    
#     graph = StateGraph(ChatState)
#     graph.add_node("chat_node", chat_node)
#     graph.add_node("tools", tool_node)
    
#     graph.add_edge(START, "chat_node")
#     graph.add_conditional_edges("chat_node", tools_condition)
#     graph.add_edge('tools', 'chat_node')
    
#     return graph.compile(checkpointer=_checkpointer)

# def retrieve_all_threads():
#     """Retrieve all thread IDs from checkpointer"""
#     if _checkpointer is None:
#         return []
    
#     all_threads = set()
#     try:
#         for checkpoint in _checkpointer.list(None):
#             all_threads.add(checkpoint.config["configurable"]["thread_id"])
#     except Exception as e:
#         print(f"Error retrieving threads: {e}")
    
#     return list(all_threads)

# def get_chatbot():
#     """Get the initialized chatbot"""
#     return _chatbot

# # Initialize the backend when imported
# chatbot, checkpointer = initialize_backend()
