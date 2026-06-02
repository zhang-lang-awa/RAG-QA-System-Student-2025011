from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    all_chunks = []
    for doc in documents:
        chunks = split_text(doc, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
    return all_chunks

if __name__ == "__main__":
    sample_text = """自然语言处理（NLP）是人工智能领域的一个重要分支，它致力于使计算机能够理解、解释和生成人类语言。NLP技术涵盖了多个子领域，包括语音识别、机器翻译、文本分类、情感分析等。
    
近年来，随着深度学习技术的发展，NLP取得了显著的进步。Transformer架构的出现更是革命性地改变了NLP领域，使得模型能够处理更长的文本序列，并在各种任务上取得了state-of-the-art的性能。

常见的NLP应用包括：智能客服机器人、语言翻译软件、文本摘要工具、情感分析系统等。这些应用正在改变人们的工作和生活方式。

在实际应用中，NLP技术面临着许多挑战，例如：处理歧义、理解上下文、处理多语言、应对数据稀疏等问题。研究人员正在不断努力解决这些问题，推动NLP技术向前发展。"""
    
    chunks = split_text(sample_text, chunk_size=200, chunk_overlap=50)
    print(f"Original text length: {len(sample_text)}")
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}: {len(chunk)} characters")
        print(chunk)