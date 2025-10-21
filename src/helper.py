from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document


def load_pdf_files(data):
    loader=DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents=loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document]=[]
    for doc in docs:
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": doc.metadata.get("source")}
        )
        )
    return minimal_docs
# splitting document into smaller parts
def text_split(minimal_docs):
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts=text_splitter.split_documents(minimal_docs)
    return texts
from langchain.embeddings import HuggingFaceEmbeddings

def download_embeddings():
    embeddings=HuggingFaceEmbeddings( model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
