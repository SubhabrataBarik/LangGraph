# # app.py
# import streamlit as st
# from MCP_backend import get_chatbot, retrieve_all_threads, async_initialize_tools
# from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# import uuid
# import asyncio

# st.set_page_config(
#     page_title="LangGraph MCP Chatbot",
#     page_icon="🤖",
#     layout="wide"
# )

# def generate_thread_id():
#     return str(uuid.uuid4())

# @st.cache_resource
# def initialize_mcp_tools():
#     """Initialize MCP tools with caching"""
#     try:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         tools = loop.run_until_complete(async_initialize_tools())
#         loop.close()
#         return tools
#     except Exception as e:
#         st.error(f"Failed to initialize MCP tools: {e}")
#         return []

# # Initialize session state
# if "initialized" not in st.session_state:
#     with st.spinner("Initializing MCP tools..."):
#         tools = initialize_mcp_tools()
#         st.session_state["initialized"] = True
#         st.session_state["tools_count"] = len(tools)

# if "message_history" not in st.session_state:
#     st.session_state["message_history"] = []

# if "thread_id" not in st.session_state:
#     st.session_state["thread_id"] = generate_thread_id()

# # Main UI
# st.title("🤖 LangGraph MCP Chatbot")
# st.sidebar.write(f"Tools loaded: {st.session_state.get('tools_count', 0)}")

# # Chat interface
# for message in st.session_state["message_history"]:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# if user_input := st.chat_input("Type your message here..."):
#     st.session_state["message_history"].append({"role": "user", "content": user_input})
    
#     with st.chat_message("user"):
#         st.write(user_input)
    
#     chatbot = get_chatbot()
#     if chatbot:
#         config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
        
#         with st.chat_message("assistant"):
#             def stream_response():
#                 for message_chunk, metadata in chatbot.stream(
#                     {"messages": [HumanMessage(content=user_input)]},
#                     config=config,
#                     stream_mode="messages",
#                 ):
#                     if isinstance(message_chunk, AIMessage) and message_chunk.content:
#                         yield message_chunk.content
            
#             ai_response = st.write_stream(stream_response())
#             st.session_state["message_history"].append({"role": "assistant", "content": ai_response})
