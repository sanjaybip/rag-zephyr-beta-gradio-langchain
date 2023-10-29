import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader

model_name = "BAAI/bge-large-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

loader = PyPDFLoader("dino-types.pdf")

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

texts = text_splitter.split_documents(documents)

# print(texts[0])
vector_store = Chroma.from_documents(
    texts,
    embeddings,
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory="stores/dino_cosine",
)
