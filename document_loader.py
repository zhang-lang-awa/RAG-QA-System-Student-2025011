import os
from PyPDF2 import PdfReader
from docx import Document

def load_pdf(file_path):
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text()
    except Exception as e:
        print(f"Error loading PDF {file_path}: {e}")
    return text

def load_docx(file_path):
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error loading DOCX {file_path}: {e}")
    return text

def load_document(file_path):
    if file_path.lower().endswith('.pdf'):
        return load_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        return load_docx(file_path)
    elif file_path.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    else:
        print(f"Unsupported file type: {file_path}")
        return ""

def load_documents_from_folder(folder_path):
    documents = []
    file_names = []
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist")
        return documents, file_names
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            text = load_document(file_path)
            if text:
                documents.append(text)
                file_names.append(filename)
                print(f"Loaded: {filename}")
    return documents, file_names

if __name__ == "__main__":
    docs, names = load_documents_from_folder("documents")
    print(f"\nLoaded {len(docs)} documents")
    for name in names:
        print(f"- {name}")