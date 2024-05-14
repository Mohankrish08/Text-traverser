# importing libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceHubEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS 
from langchain.chains import RetrievalQA 

# LLM model
def build_embedding(data):
    model_name="mistralai/Mistral-7B-Instruct-v0.2"
    embeddings=HuggingFaceEmbeddings()
    vector_store=FAISS.from_documents(data, embeddings)
    llm=HuggingFaceHub(repo_id = model_name, model_kwargs={"temperature":0.3, "max_new_tokens":2048, "max_length": 256, "num_beams":4}, huggingfacehub_api_token=<"Your token">)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k":5}))
    return qa_chain

# File
loader = WebBaseLoader("https://medium.com/@mohankrishce/decoding-cnns-a-mathematical-perspective-1429359dfb96")
texts = loader.load()


# build embeddings
output = build_embedding(texts)

# result
print(output("what is tensorflow")['result'])
