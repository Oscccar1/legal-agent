# -*- coding: utf-8 -*-
"""基于 LangGraph InMemorySaver 的 Agent 记忆管理"""

from typing import Dict, Any
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, AIMessage
from .legal_prompt import SYSTEM_PROMPT, LEGAL_SKILLS

# 全局存储：checkpointer（长期记忆）和工具列表（RAG）
_checkpointer = None
_llm = None
_agent = None
_session_tools = {}  # session_id -> tools


def get_llm():
    """获取通义千问大模型实例"""
    global _llm
    if _llm is None:
        _llm = ChatTongyi(model="qwen3-max")
    return _llm


def get_checkpointer():
    """获取 InMemorySaver 检查点管理器（长期记忆）"""
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = InMemorySaver()
    return _checkpointer


def create_legal_agent(session_id: str, rag_tools: list = None):
    """
    为每个会话创建专属的法律 Agent
    - session_id: 会话唯一标识（thread_id），用于隔离不同会话的记忆
    - rag_tools: 从知识库检索工具
    """
    llm = get_llm()
    checkpointer = get_checkpointer()

    # 合并系统提示词和法律技能
    full_system_prompt = SYSTEM_PROMPT + "\n" + LEGAL_SKILLS

    # 构建工具列表（RAG 工具 + 其他）
    tools = rag_tools if rag_tools else []

    # 使用 create_react_agent 创建 Agent，配置 checkpointer 实现长期记忆
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=full_system_prompt,
        checkpointer=checkpointer
    )
    return agent


def get_session_config(session_id: str) -> Dict[str, Any]:
    """
    获取会话配置，包含 thread_id
    thread_id 是 LangGraph Checkpointer 用于隔离不同会话记忆的关键
    """
    return {"configurable": {"thread_id": session_id}}


def invoke_agent(agent, user_input: str, session_id: str) -> str:
    """
    调用 Agent 并返回响应
    - 通过 config 中的 thread_id 实现长期记忆（跨对话保留上下文）
    """
    config = get_session_config(session_id)
    response = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    # 获取最后一条 AI 消息
    last_message = response["messages"][-1]
    return last_message.content if hasattr(last_message, 'content') else str(last_message)