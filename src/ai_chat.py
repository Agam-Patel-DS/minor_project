from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader


load_dotenv()
import os

from google.colab import userdata
HF_TOKEN=userdata.get('HF_TOKEN')
GROQ_API_KEY=userdata.get('GROQ_API_KEY')

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

groq_api_key=GROQ_API_KEY

llm=ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
  """
  Asnwer the questions based on the provided context only.
  Please provide the most accurate response based on the question.
  <context>
  {context}
  <context>
  Question:{input}
  """
)
import time


def create_vector_embeddings(directory):
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    loader=PyPDFDirectoryLoader(directory) #Data Ingestion
    docs=loader.load() #Document Loading
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    final_documents=text_splitter.split_documents(docs[:50])
    vectors=FAISS.from_documents(final_documents, embeddings)
    return vectors


def response(user_prompt,dire):
    if user_prompt:
      vectors=create_vector_embeddings(dire)
      document_chain=create_stuff_documents_chain(llm, prompt)
      retriever=vectors.as_retriever()
      retriever_chain=create_retrieval_chain(retriever, document_chain)

      start=time.process_time()
      response=retriever_chain.invoke({"input":user_prompt})
      return response



