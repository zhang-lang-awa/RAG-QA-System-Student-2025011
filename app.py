import streamlit as st
import os
import tempfile
from document_loader import load_document
from text_splitter import split_text
from vector_store import VectorStoreManager
from rag_chain import RAGQAChain

def init_session_state():
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStoreManager()
    
    if 'rag_chain' not in st.session_state:
        retriever = st.session_state.vector_store.get_retriever(k=3)
        st.session_state.rag_chain = RAGQAChain(retriever)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

def main():
    st.set_page_config(page_title="RAG智能问答系统", page_icon="📚", layout="wide")
    
    init_session_state()
    
    st.title("📚 基于本地知识库的RAG智能问答系统")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("文档上传")
        uploaded_files = st.file_uploader(
            "上传PDF、DOCX或TXT文件",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        
        if st.button("📥 构建知识库"):
            if uploaded_files:
                with st.spinner("正在处理文档..."):
                    all_texts = []
                    all_metadatas = []
                    
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                            temp_file.write(uploaded_file.read())
                            temp_file_path = temp_file.name
                        
                        try:
                            text = load_document(temp_file_path)
                            if text:
                                chunks = split_text(text, chunk_size=1000, chunk_overlap=200)
                                all_texts.extend(chunks)
                                all_metadatas.extend([{"source": uploaded_file.name} for _ in chunks])
                                st.session_state.uploaded_files.append(uploaded_file.name)
                        finally:
                            os.unlink(temp_file_path)
                    
                    if all_texts:
                        st.session_state.vector_store.add_documents(all_texts, all_metadatas)
                        retriever = st.session_state.vector_store.get_retriever(k=3)
                        st.session_state.rag_chain = RAGQAChain(retriever)
                        st.success(f"✅ 成功处理 {len(uploaded_files)} 个文档，生成 {len(all_texts)} 个文本块")
                    else:
                        st.error("❌ 未能提取任何文本内容")
            else:
                st.warning("⚠️ 请先上传文件")
        
        if st.button("🗑️ 清空知识库"):
            st.session_state.vector_store.clear_store()
            st.session_state.chat_history = []
            st.session_state.uploaded_files = []
            retriever = st.session_state.vector_store.get_retriever(k=3)
            st.session_state.rag_chain = RAGQAChain(retriever)
            st.success("✅ 知识库已清空")
        
        st.header("📊 知识库状态")
        total_chunks = st.session_state.vector_store.get_total_chunks()
        st.metric(label="文本块数量", value=total_chunks)
        
        if st.session_state.uploaded_files:
            st.subheader("已上传文档")
            for file_name in st.session_state.uploaded_files:
                st.write(f"- {file_name}")
    
    with col2:
        st.header("💬 问答交互")
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        user_question = st.text_input("请输入您的问题：", key="question_input")
        
        if st.button("🚀 提问"):
            if user_question.strip():
                with st.spinner("正在检索并生成回答..."):
                    answer = st.session_state.rag_chain.ask(user_question)
                    
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_question
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })
                    
                    st.rerun()
            else:
                st.warning("⚠️ 请输入问题")
        
        if st.button("🔄 清空对话"):
            st.session_state.chat_history = []
            st.session_state.rag_chain.clear_history()
            st.rerun()

if __name__ == "__main__":
    main()