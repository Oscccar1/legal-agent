import os
import streamlit as st

# 先尝试从 Streamlit secrets 读取（云端），失败则从 .env 读取（本地）
try:
    api_key = st.secrets["DASHSCOPE_API_KEY"]
    os.environ["DASHSCOPE_API_KEY"] = api_key
except (FileNotFoundError, KeyError):
    from dotenv import load_dotenv
    load_dotenv()   # 读取本地的 .env 文件

# -*- coding: utf-8 -*-
import streamlit as st
from dotenv import load_dotenv
import os

# 优先使用 Streamlit secrets（云端），否则使用 .env（本地）
try:
    dashscope_api_key = st.secrets["DASHSCOPE_API_KEY"]
    os.environ["DASHSCOPE_API_KEY"] = dashscope_api_key
except:
    load_dotenv()

from utils.memory_manager import create_legal_agent, invoke_agent
from utils.rag_manager import add_to_knowledge_base, get_retrieval_tool, load_or_create_vectorstore

st.set_page_config(page_title="法律助手｜专业法律咨询", page_icon="⚖️", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant",
                                  "content": "您好，我是您的专属法律助手。请告诉我您需要什么帮助？\n\n⚠️ **免责声明**：仅供法律信息参考，不构成正式法律意见。"}]
if "session_id" not in st.session_state:
    st.session_state.session_id = "legal_user_default"
if "agent" not in st.session_state:
    st.session_state.agent = None

# 侧边栏
with st.sidebar:
    st.title("⚖️ 法律助手")
    new_session_id = st.text_input("会话 ID（切换 ID 将开启新会话）", value=st.session_state.session_id)
    if new_session_id != st.session_state.session_id:
        st.session_state.session_id = new_session_id
        st.session_state.agent = None
        st.session_state.messages = [
            {"role": "assistant", "content": f"已切换到会话「{new_session_id}」。有什么可以帮您？"}]
        st.rerun()

    st.markdown("---")
    st.subheader("📚 法律知识库")
    uploaded_files = st.file_uploader("上传 PDF/TXT 文件", type=["pdf", "txt"], accept_multiple_files=True)
    if uploaded_files and st.button("添加到知识库"):
        with st.spinner("处理中..."):
            cnt, msg = add_to_knowledge_base(uploaded_files)
            st.success(msg if cnt > 0 else "失败，请检查文件格式")
            st.session_state.agent = None
            st.rerun()

    vs = load_or_create_vectorstore()
    if vs and vs.index.ntotal > 1:
        st.success(f"✅ 知识库含 {vs.index.ntotal - 1} 个文档块")
    else:
        st.warning("⚠️ 知识库为空，请上传法律文档。")

# 聊天区域
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("请输入您的法律问题"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if st.session_state.agent is None:
        retrieval_tool = get_retrieval_tool()
        st.session_state.agent = create_legal_agent(st.session_state.session_id, [retrieval_tool])

    with st.chat_message("assistant"):
        with st.spinner("分析中..."):
            try:
                response = invoke_agent(st.session_state.agent, user_input, st.session_state.session_id)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                err = f"错误：{e}，请检查 API Key"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})