import os
import requests

import config
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings,)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma



#work_dir = os.getcwd()
ECO101 = './transcripts/ECO101/'
standard_query = "What does it mean to think about margins?"

def embedding_function(model = ''):
    if model == "OpenAI":
        embeddings_model = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    else:
        embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings_model

def embeddings_dir(textfiles_dir, model = ''):
    return textfiles_dir + model+'embeddings/'
    
def create_embeddings(textfiles_dir, model = ''):
    print('Loading ' + textfiles_dir)
    loader = DirectoryLoader(textfiles_dir, glob="**/*.docx", loader_cls=Docx2txtLoader)
    doc = loader.load()
    print('Creating embedding model '+ model)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len, is_separator_regex=False,)
    chunks  = text_splitter.split_documents(doc)
    print('Creating embeddings for ' + textfiles_dir)
    db = Chroma.from_documents(chunks, embedding_function(model), 
                               persist_directory=embeddings_dir(textfiles_dir, model))
    
def load_embeddings(textfiles_dir, model = ''):
    print('Loading embeddings from ' + textfiles_dir + " " + model)
    db = Chroma(persist_directory=embeddings_dir(textfiles_dir, model), 
                embedding_function=embedding_function(model))
    return db

def find_chunks(db, query):
    docs = db.similarity_search(query)
    l = min(3, len(docs))
    for i in range(l):
        print(docs[i].page_content)
    return docs

def do_things():
    create_embeddings(ECO101, model = 'OpenAI')
    db = load_embeddings(ECO101, model = 'OpenAI')
    find_chunks(db, 'why people make decisions')    

def do_things2(query = standard_query):
    db = load_embeddings(ECO101, model = 'OpenAI')
    chunks = find_chunks(db, query)
    k=2

#do_things()
#do_things2(standard_query)