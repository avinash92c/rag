from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf
import fitz
import requests
import re

llm = ChatOllama(base_url="http://localhost:11434", model="nemotron-3-nano:4b")
embeddings = OllamaEmbeddings(base_url="http://localhost:11434",model="nomic-embed-text")

pdf_url = "https://www.londonsrimurugan.org/pdf1/RAMAYANA.pdf"

response = requests.get(pdf_url)
doc = fitz.open(stream=response.content, filetype="pdf", filename="ramayana")

doc_arr = []
doc_idx = []

# for page in doc:
for idx, page in enumerate(doc):
    pagestr = ""

    if idx == 0:
        continue  # SKIP TITLE PAGE
    elif idx == 1:
        # Contents Page
        for text in page.get_text_blocks():
            # print(text)
            match = re.search(pattern="^[0-9].[0-9]+[a-z A-Z -]+",string=text[4])
            if match:
                # print(match.group())
                doc_idx.append(match.group().strip())
        # continue
    else:
        # print(page.get_text_blocks())
        for text in page.get_text_blocks():
            # print(text[4].strip().replace("\n",""))
            sentence = text[4].strip().replace("\n", " ")
            # sentence = sentence.replace(,"")
            sentence = re.sub(pattern="^[0-9]+$", repl="", string=sentence)
            # print(sentence)
            if len(sentence) > 0:
                pagestr += " "+sentence
            # print("-"*40)
        doc_arr.append(pagestr.strip())

    # print("-"*80)

new_doc = []
total_text = " ".join(doc_arr)
del doc_arr
for index in reversed(doc_idx):
    if index in total_text:
        print(index)
        pieces = total_text.split(sep=index)
        # new_doc.append(pieces[0])
        new_doc.append(index + " " + pieces[1])
        total_text = pieces[0]

print(len(new_doc))

del total_text
    
# for page in new_doc:
    # print(page)
    # print(len(page.split()))
    # print(page)
    # print("-"*40)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)

final_docs = []

for section in new_doc:
    final_docs.extend(splitter.split_text(section))


        # for index in doc_idx:
        #     if index in pagestr:
        #         doc_arr.extend(pagestr.split(sep=index))
        #     else:
        #         doc_arr.append(pagestr.strip())

#STORE TO CHROMA DB
vectorstore = Chroma.from_texts(final_docs,embeddings,persist_directory="rag.db")


question = input("Ask me a question")
result = vectorstore.similarity_search(query=question,k=5)

for r in result:
    print(r.page_content)
    print("-" * 80)


llm_prompt=f"""
            Based on input context, answer the following question.
            Rules: Do not answer from your memory.
            Only answer questions based on provided context data
            If you do not find any answer just say i don't know. do not assume anything

            question: {question}
            context: {result}
           """

result = llm.invoke(input=llm_prompt)
print(result)
