import requests
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
import pickle

llm = ChatOllama(base_url="http://localhost:11434", model="nemotron-3-nano:4b")
# embeddings = OllamaEmbeddings(base_url="http://localhost:11434",model="nomic-embed-text")
embeddings = OllamaEmbeddings(base_url="http://localhost:11434",model="mxbai-embed-large:335m")

payload = {
  "sources": [
    {
      "kind": "http",
      "url": "https://www.londonsrimurugan.org/pdf1/RAMAYANA.pdf"
    }
  ],
  "options": {
    "to_formats": ["md"]
  }
}

response = requests.post(
    "http://localhost:5001/v1/convert/source",
    json=payload
)

print(response)

result = response.json()

print(result)

headers_to_split_on = [
    ("#", "book"),
    ("##", "chapter"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

docs = splitter.split_text(result["document"]["md_content"])

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200
)

chunks = recursive_splitter.split_documents(docs)
ids = [str(i) for i in range(len(chunks))]
# vectorstore = Chroma.from_documents(chunks,embeddings,persist_directory="rag.db")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    ids=ids,
    persist_directory="rag.db"
)

texts = [doc.page_content for doc in chunks]

tokenized_corpus = [
    text.lower().split()
    for text in texts
]

bm25 = BM25Okapi(tokenized_corpus)

with open("bm25.pkl", "wb") as f:
    pickle.dump(
        {
            "bm25": bm25,
            "texts": texts
        },
        f
    )