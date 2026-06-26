from langchain_core.documents import Document
import requests
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import pickle

llm = ChatOllama(base_url="http://localhost:11434", model="nemotron-3-nano:4b")
# embeddings = OllamaEmbeddings(base_url="http://localhost:11434",model="nomic-embed-text")
embeddings = OllamaEmbeddings(base_url="http://localhost:11434",model="mxbai-embed-large:335m")

vectorstore = Chroma(
    persist_directory="rag.db",
    embedding_function=embeddings
)

with open("bm25.pkl", "rb") as f:
    data = pickle.load(f)

question = input("Ask me a question")

results = vectorstore.similarity_search(query=question,k=20)

bm25 = data["bm25"]
texts = data["texts"]
query_tokens = question.lower().split()

scores = bm25.get_scores(query_tokens)

top_indices = sorted(
    range(len(scores)),
    key=lambda i: scores[i],
    reverse=True
)[:20]

bm25_docs = [
    Document(page_content=texts[i])
    for i in top_indices
]

all_docs = {}

for doc in results:
    all_docs[doc.page_content] = doc

for doc in bm25_docs:
    all_docs[doc.page_content] = doc

combined_docs = list(all_docs.values())

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    cache_folder="./models"
)

pairs = [
    (question, doc.page_content)
    for doc in combined_docs
]

scores = reranker.predict(pairs)
ranked = sorted(
    zip(combined_docs, scores),
    key=lambda x: x[1],
    reverse=True
)

for doc, score in ranked[:10]:
    print(score)
    print(doc.metadata)
    print(doc.page_content[:200])
    print("-" * 80)

top_docs = [
    doc.page_content
    for doc, score in ranked[:5]
]

context = "\n\n".join(top_docs)

# llm_prompt=f"""
#             Based on input context, answer the following question.
#             Rules: Do not answer from your memory.
#             Only answer questions based on provided context data
#             If you do not find any answer just say i don't know. do not assume anything

#             question: {question}
#             context: {context}
#            """

llm_prompt = f"""
You are a question answering system.

Answer ONLY using information explicitly present in the context.

If the answer is not present word-for-word or clearly stated in the context,
respond exactly:

I DON'T KNOW

Do not use outside knowledge.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
"""

print(f"prompt {llm_prompt}")

result = llm.invoke(input=llm_prompt)
print(result)