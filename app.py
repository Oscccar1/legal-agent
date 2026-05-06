# -*- coding: utf-8 -*-
"""
法律助手 Agent
功能：
- 长期记忆（LangGraph InMemorySaver + thread_id）
- 短期记忆（同一会话内上下文）
- RAG 知识库检索（支持上传 PDF/TXT）
- Streamlit 界面
"""

import streamlit as st
from dotenv import load_dotenv
import os

from utils.memory_manager import create_legal_agent, invoke_agent, get_session_config
from utils.rag_manager import add_to_knowledge_base, get_retrieval_tool, load_or_create_vectorstore

# 加载环境变量
load_dotenv()

# 页面配置
st.set_page_config(
    page_title="法律助手｜专业法律咨询服务",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 session_state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "您好，我是您的专属法律助手。我可以帮您解答法律疑问、审查合同条款、提供法律咨询参考。\n\n⚠️ **免责声明**：我的回答仅供法律信息参考，不构成正式法律意见。如有具体法律问题，请咨询持证律师。\n\n请告诉我您需要什么帮助？"}
    ]

if "session_id" not in st.session_state:
    st.session_state.session_id = "legal_user_default"

if "agent" not in st.session_state:
    st.session_state.agent = None

if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False

# ========== 侧边栏 ==========
with st.sidebar:
    st.title("⚖️ 法律助手")
    st.markdown("---")

    # 会话管理
    st.subheader("👤 会话设置")
    new_session_id = st.text_input("会话 ID（切换 ID 将开启新会话）", value=st.session_state.session_id)
    if new_session_id != st.session_state.session_id:
        st.session_state.session_id = new_session_id
        st.session_state.agent = None
        st.session_state.messages = [
            {"role": "assistant", "content": f"已切换到新会话「{new_session_id}」。有什么法律问题可以帮您解答？"}
        ]
        st.rerun()

    st.markdown("---")

    # 长期记忆说明
    st.subheader("🧠 长期记忆")
    st.info(
        "当前会话 ID: `" + st.session_state.session_id + "`\n\n同 ID 下的对话会被自动记住，切换 ID 后可开启全新会话，互不干扰。",
        icon="💡")

    st.markdown("---")

    # 知识库管理
    st.subheader("📚 法律知识库")
    st.markdown("上传法律相关文档（PDF 或 TXT），助手将能从文档中检索信息。")

    uploaded_files = st.file_uploader(
        "选择文件上传",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="knowledge_uploader"
    )

    if uploaded_files and st.button("📥 添加到知识库", use_container_width=True):
        with st.spinner("正在处理文档..."):
            chunk_count, msg = add_to_knowledge_base(uploaded_files)
            if chunk_count > 0:
                st.success(msg)
                # 重新初始化 Agent，使新知识库生效
                st.session_state.agent = None
                st.session_state.rag_initialized = False
            else:
                st.error(msg)

    # 显示知识库状态
    vectorstore = load_or_create_vectorstore()
    if vectorstore and vectorstore.index.ntotal > 1:
        st.success(f"✅ 知识库已激活，当前包含 {vectorstore.index.ntotal - 1} 个文档块")
    else:
        st.warning("⚠️ 知识库为空，请上传法律文档。")

    st.markdown("---")
    st.markdown("**⚠️ 免责声明**：本助手提供的信息仅供参考，不构成正式法律意见。")
    st.caption("Powered by LangGraph + 通义千问")

# ========== 主对话界面 ==========
st.title("⚖️ 专业法律助手")
st.caption("具备长期记忆（跨对话保留上下文）+ RAG 法律知识库检索")

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 处理用户输入
if user_input := st.chat_input("请输入您的法律相关问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 获取或创建 Agent
    if st.session_state.agent is None:
        # 获取 RAG 检索工具
        retrieval_tool = get_retrieval_tool(st.session_state.session_id)
        st.session_state.agent = create_legal_agent(
            session_id=st.session_state.session_id,
            rag_tools=[retrieval_tool]
        )

    # 调用 Agent
    with st.chat_message("assistant"):
        with st.spinner("正在分析中..."):
            try:
                response = invoke_agent(
                    st.session_state.agent,
                    user_input,
                    st.session_state.session_id
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"处理出错了：{str(e)}\n\n请检查 API Key 是否正确配置。"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})