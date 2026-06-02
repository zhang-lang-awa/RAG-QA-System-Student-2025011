from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

class RAGQAChain:
    def __init__(self, retriever, llm_model="deepseek-r1:7b"):
        self.llm = ChatOllama(model=llm_model, temperature=0.1)
        self.retriever = retriever
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.system_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
你是一个基于本地知识库的智能问答助手。请根据提供的参考文档来回答用户的问题。

参考文档内容：
{context}

用户问题：
{question}

请严格按照以下规则回答：
1. 仔细阅读参考文档，仅使用文档中包含的信息进行回答。
2. 如果文档中包含多个相关信息，请综合所有相关内容进行回答。
3. 如果文档中没有找到与问题相关的信息，请明确回答"文档中未找到相关答案"。
4. 回答要简洁明了，不要添加文档中没有的信息。
5. 如果问题是关于文档内容的，确保回答准确反映文档中的描述。

请开始回答：
"""
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.system_prompt},
            verbose=False
        )
    
    def ask(self, question):
        try:
            result = self.chain({"question": question})
            return result["answer"]
        except Exception as e:
            return f"回答时出现错误: {str(e)}"
    
    def clear_history(self):
        self.memory.clear()

if __name__ == "__main__":
    from vector_store import VectorStoreManager
    
    vs_manager = VectorStoreManager()
    
    sample_texts = [
        "自然语言处理（NLP）是人工智能领域的一个重要分支，它致力于使计算机能够理解、解释和生成人类语言。NLP技术涵盖了多个子领域，包括语音识别、机器翻译、文本分类、情感分析等。",
        "Transformer架构是NLP领域的革命性突破，由Google在2017年提出。它采用自注意力机制，能够并行处理输入序列，大大提高了模型的效率和性能。",
        "BERT是一种基于Transformer的预训练语言模型，由Google在2018年发布。BERT在多个NLP任务上取得了state-of-the-art的成绩，包括问答、命名实体识别和文本分类等。",
        "词嵌入是将词语转换为向量表示的技术。常见的词嵌入方法包括Word2Vec、GloVe和FastText等。词嵌入能够捕捉词语之间的语义关系。",
        "文本分类是NLP的一个重要任务，用于将文本划分到预先定义的类别中。常见的应用包括垃圾邮件过滤、情感分析和主题分类等。",
        "情感分析是NLP的一个子领域，旨在识别和提取文本中的情感倾向。情感分析可以分为积极、消极和中性三类，广泛应用于社交媒体监控和客户反馈分析。",
        "机器翻译是利用计算机技术将一种语言的文本转换为另一种语言的过程。近年来，基于Transformer的神经机器翻译模型在翻译质量上取得了显著进步。",
        "命名实体识别（NER）是识别文本中命名实体的任务，包括人名、地名、组织机构名等。NER在信息抽取和知识图谱构建中起着重要作用。"
    ]
    
    vs_manager.add_documents(sample_texts)
    retriever = vs_manager.get_retriever(k=3)
    
    rag_chain = RAGQAChain(retriever)
    
    test_questions = [
        "什么是自然语言处理？",
        "Transformer架构是什么时候提出的？",
        "BERT模型有什么应用？",
        "词嵌入有哪些方法？",
        "文本分类有哪些应用？",
        "什么是情感分析？",
        "如何制作蛋糕？",
        "量子计算的原理是什么？"
    ]
    
    print("=== RAG问答测试 ===\n")
    for question in test_questions:
        print(f"问题: {question}")
        answer = rag_chain.ask(question)
        print(f"回答: {answer}\n")