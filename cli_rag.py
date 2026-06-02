from document_loader import load_documents_from_folder
from text_splitter import split_documents
from vector_store import VectorStoreManager
from rag_chain import RAGQAChain

def main():
    print("=== RAG智能问答系统 - 命令行版本 ===\n")
    
    vs_manager = VectorStoreManager()
    
    docs_folder = "documents"
    print(f"正在加载文档目录: {docs_folder}")
    documents, file_names = load_documents_from_folder(docs_folder)
    
    if documents:
        print(f"成功加载 {len(documents)} 个文档")
        for name in file_names:
            print(f"- {name}")
        
        print("\n正在进行文本分块...")
        chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
        print(f"生成 {len(chunks)} 个文本块")
        
        print("\n正在向量化并存储到Chroma...")
        vs_manager.add_documents(chunks)
        print("完成!")
    else:
        print("未找到文档，使用已有知识库")
    
    retriever = vs_manager.get_retriever(k=3)
    rag_chain = RAGQAChain(retriever)
    
    print("\n=== 开始问答 ===")
    print("输入 'quit' 或 'exit' 退出")
    
    while True:
        question = input("\n请输入问题: ")
        
        if question.lower() in ['quit', 'exit']:
            print("再见!")
            break
        
        if not question.strip():
            print("请输入有效的问题")
            continue
        
        print("正在思考...")
        answer = rag_chain.ask(question)
        print(f"\n回答: {answer}")

if __name__ == "__main__":
    main()