from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage
from .legal_prompt import SYSTEM_PROMPT

_checkpointer = None
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatTongyi(model="qwen3-max")
    return _llm

def get_checkpointer():
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = InMemorySaver()
    return _checkpointer

def create_legal_agent(session_id: str, tools=None):
    llm = get_llm()
    checkpointer = get_checkpointer()
    agent = create_react_agent(
        model=llm,
        tools=tools or [],
        state_modifier=SYSTEM_PROMPT,
        checkpointer=checkpointer
    )
    return agent

def invoke_agent(agent, user_input: str, session_id: str) -> str:
    config = {"configurable": {"thread_id": session_id}}
    response = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    return response["messages"][-1].content