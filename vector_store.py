import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

class VectorStoreManager:
    def __init__(self, persist_directory="./chroma_db", model_name="nomic-embed-text"):
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.vector_store = None
        self._init_vector_store()
    
    def _init_vector_store(self):
        if os.path.exists(self.persist_directory):
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
    
    def add_documents(self, texts, metadatas=None):
        if metadatas is None:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
        
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)
        self.vector_store.persist()
    
    def get_retriever(self, k=3):
        return self.vector_store.as_retriever(search_kwargs={"k": k})
    
    def similarity_search(self, query, k=3):
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def get_total_chunks(self):
        return self.vector_store._collection.count()
    
    def clear_store(self):
        self.vector_store.reset_collection()
        self.vector_store.persist()

if __name__ == "__main__":
    vs_manager = VectorStoreManager()
    
    sample_texts = [
        "自然语言处理（NLP）是人工智能领域的一个重要分支，它致力于使计算机能够理解、解释和生成人类语言。",
        "Transformer架构是NLP领域的革命性突破，由Google在2017年提出。",
        "BERT是一种基于Transformer的预训练语言模型，在多个NLP任务上取得了优异成绩。",
        "词嵌入是将词语转换为向量表示的技术，常见的方法有Word2Vec、GloVe等。",
        "文本分类是NLP的一个重要任务，用于将文本划分到预先定义的类别中。"
    ]
    
    vs_manager.add_documents(sample_texts)
    print(f"Total chunks in store: {vs_manager.get_total_chunks()}")
    
    query = "Transformer架构是什么？"
    results = vs_manager.similarity_search(query, k=3)
    print(f"\nSearch results for: {query}")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(result.page_content)