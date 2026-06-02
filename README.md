# RAG智能问答系统

基于本地知识库的检索增强生成（RAG）智能问答系统，利用Ollama本地大模型、LangChain框架和Streamlit开发工具构建。

## 项目简介

本项目实现了一个能够"学习"指定本地文档并回答相关问题的智能问答系统。系统支持PDF、DOCX、TXT等多种文档格式，通过文本分块、向量化存储和相似性检索，结合大语言模型实现精准的问答功能。

## 环境要求与安装步骤

### 环境要求
- Python 3.8+
- Windows 10/11
- Ollama（本地大模型服务）

### 安装步骤

1. **安装Ollama**
   ```bash
   # 从官网下载安装：https://ollama.com/
   # 安装完成后，下载模型
   ollama pull deepseek-r1:7b
   ollama pull nomic-embed-text
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **安装依赖包**
   ```bash
   pip install -r requirements.txt
   ```

## 使用说明

### 运行Web应用
```bash
streamlit run app.py
```

### 使用命令行版本
```bash
python cli_rag.py
```

### 功能说明

1. **文档上传**：支持上传PDF、DOCX、TXT格式的文档
2. **知识库构建**：点击"构建知识库"按钮，系统会自动解析文档、分块并更新向量库
3. **问答交互**：在输入框中输入问题，点击"提问"按钮获取答案
4. **对话历史**：显示最近的多轮问答记录
5. **知识库状态**：显示当前知识库中的文本块数量和已上传文档列表

## 关键技术点说明

### RAG流程
1. **文档加载**：使用PyPDF2和python-docx加载PDF和DOCX文档
2. **文本分块**：使用RecursiveCharacterTextSplitter进行分块（chunk_size=1000, chunk_overlap=200）
3. **向量化存储**：使用Ollama的nomic-embed-text模型将文本块向量化，存入Chroma向量数据库
4. **相似性检索**：根据用户查询检索最相关的3个文本块
5. **问答生成**：使用ConversationalRetrievalChain将检索结果和问题输入大模型生成回答

### 所用模型
- **大语言模型**：deepseek-r1:7b（可配置为其他Ollama模型）
- **嵌入模型**：nomic-embed-text

### 项目结构
```
RAG-QA-System/
├── app.py              # Streamlit Web应用主文件
├── cli_rag.py          # 命令行版本RAG问答脚本
├── document_loader.py  # 文档加载模块
├── text_splitter.py    # 文本分块模块
├── vector_store.py     # 向量存储管理模块
├── rag_chain.py        # RAG问答链模块
├── test_ollama.py      # Ollama连接测试脚本
├── requirements.txt    # 依赖包列表
├── .gitignore          # Git忽略配置
└── documents/          # 示例文档目录
    ├── nlp_introduction.txt
    ├── transformer.txt
    ├── bert.txt
    ├── word_embedding.txt
    └── text_classification.txt
```

## 问答示例

**问题1**: 什么是自然语言处理？
**回答**: 自然语言处理（NLP）是人工智能领域的一个重要分支，它致力于使计算机能够理解、解释和生成人类语言。

**问题2**: Transformer架构是什么时候提出的？
**回答**: Transformer架构由Google在2017年的论文《Attention is All You Need》中提出。

**问题3**: BERT有哪些应用场景？
**回答**: BERT的应用场景包括问答系统（QA）、命名实体识别（NER）、文本分类、情感分析和语义相似度计算。

**问题4**: 如何制作蛋糕？
**回答**: 文档中未找到相关答案

## 已知问题与改进方向

### 已知问题
- 首次加载模型时可能需要较长时间
- 某些PDF文档的文本提取可能不够准确

### 改进方向
- 支持更多文档格式（如图片OCR识别）
- 优化文本分块策略
- 添加文档管理功能（删除、更新文档）
- 支持多模型切换
- 添加答案来源引用

## 许可证

MIT License