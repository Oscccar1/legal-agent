# -*- coding: utf-8 -*-
"""RAG 知识库管理：支持文件上传、向量化检索"""

import os
import tempfile
import pickle
from typing import List, Tuple, Callable
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.documents import Document

# 知识库缓存目录
KB_CACHE_DIR = "./knowledge_base_cache"
os.makedirs(KB_CACHE_DIR, exist_ok=True)

# 全局向量存储
_vectorstore = None
_embeddings = DashScopeEmbeddings(model="text-embedding-v2")


def load_or_create_vectorstore():
    """加载或创建向量存储"""
    global _vectorstore
    index_path = os.path.join(KB_CACHE_DIR, "faiss_index")
    if os.path.exists(index_path):
        try:
            _vectorstore = FAISS.load_local(index_path, _embeddings, allow_dangerous_deserialization=True)
            return _vectorstore
        except:
            pass
    # 创建空向量库
    _vectorstore = FAISS.from_texts(["初始化占位"], _embeddings)
    return _vectorstore


def save_vectorstore():
    """保存向量存储到本地"""
    if _vectorstore:
        index_path = os.path.join(KB_CACHE_DIR, "faiss_index")
        _vectorstore.save_local(index_path)


def process_uploaded_file(uploaded_file) -> List[Document]:
    """
    处理上传的文件（支持 PDF、TXT）
    返回 Document 列表
    """
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # 根据文件类型加载
    if uploaded_file.name.endswith('.pdf'):
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    elif uploaded_file.name.endswith('.txt'):
        loader = TextLoader(tmp_path, encoding='utf-8')
        documents = loader.load()
    else:
        documents = []

    # 清理临时文件
    os.unlink(tmp_path)
    return documents


def add_to_knowledge_base(uploaded_files) -> Tuple[int, str]:
    """
    将上传的文件添加到知识库
    返回: (添加的文档块数量, 状态消息)
    """
    global _vectorstore
    all_documents = []

    for uploaded_file in uploaded_files:
        docs = process_uploaded_file(uploaded_file)
        all_documents.extend(docs)

    if not all_documents:
        return 0, "未找到可处理的文档（支持 PDF 和 TXT 格式）"

    # 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)

    # 加载或创建向量库
    _vectorstore = load_or_create_vectorstore()

    # 添加新文档块
    _vectorstore.add_documents(chunks)
    save_vectorstore()

    return len(chunks), f"成功添加 {len(chunks)} 个文档块（来自 {len(uploaded_files)} 个文件）"


def get_retrieval_tool(session_id: str = None) -> Callable:
    """
    创建 RAG 检索工具（供 Agent 调用）
    """

    @tool
    def retrieve_legal_info(query: str) -> str:
        """
        从法律知识库中检索相关信息。当你需要回答涉及具体法律条文、合同条款、司法案例或专业领域问题时，可以使用此工具。
        参数: query - 用户问题中提取的关键检索词
        """
        vectorstore = load_or_create_vectorstore()
        try:
            docs = vectorstore.similarity_search(query, k=4)
            if not docs or (len(docs) == 1 and docs[0].page_content == "初始化占位"):
                return "知识库中暂无相关信息。请上传法律相关文档（PDF/TXT）后重试。"

            results = []
            for i, doc in enumerate(docs):
                results.append(f"【资料{i + 1}】\n{doc.page_content}\n")
            return "根据知识库检索到以下相关信息：\n\n" + "\n".join(results)
        except Exception as e:
            return f"检索失败: {str(e)}"

    return retrieve_legal_info