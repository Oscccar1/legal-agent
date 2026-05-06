import os, tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool

KB_CACHE_DIR = "./knowledge_base_cache"
os.makedirs(KB_CACHE_DIR, exist_ok=True)
_embeddings = DashScopeEmbeddings(model="text-embedding-v2")
_vectorstore = None

def load_or_create_vectorstore():
    global _vectorstore
    index_path = os.path.join(KB_CACHE_DIR, "faiss_index")
    if os.path.exists(index_path):
        try:
            _vectorstore = FAISS.load_local(index_path, _embeddings, allow_dangerous_deserialization=True)
            return _vectorstore
        except:
            pass
    _vectorstore = FAISS.from_texts(["初始化占位"], _embeddings)
    return _vectorstore

def save_vectorstore():
    if _vectorstore:
        index_path = os.path.join(KB_CACHE_DIR, "faiss_index")
        _vectorstore.save_local(index_path)

def process_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    if uploaded_file.name.endswith('.pdf'):
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
    else:
        loader = TextLoader(tmp_path, encoding='utf-8')
        docs = loader.load()
    os.unlink(tmp_path)
    return docs

def add_to_knowledge_base(uploaded_files):
    global _vectorstore
    all_docs = []
    for f in uploaded_files:
        all_docs.extend(process_uploaded_file(f))
    if not all_docs:
        return 0, "没有可处理的内容"
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    _vectorstore = load_or_create_vectorstore()
    _vectorstore.add_documents(chunks)
    save_vectorstore()
    return len(chunks), f"已添加 {len(chunks)} 个文档块"

def get_retrieval_tool():
    @tool
    def retrieve_legal_info(query: str) -> str:
        """从法律知识库检索相关信息。参数 query: 检索关键词。"""
        vs = load_or_create_vectorstore()
        docs = vs.similarity_search(query, k=4)
        if not docs or (len(docs)==1 and docs[0].page_content=="初始化占位"):
            return "知识库暂无相关内容，请先上传法律文档。"
        return "\n\n".join([f"【资料{i+1}】\n{d.page_content}" for i,d in enumerate(docs)])
    return retrieve_legal_info