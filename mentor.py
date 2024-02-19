#Activate using .\myenv\Scripts\activate in terminal.
#Deactivate using deactivate in terminal.

import os
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI

import streamlit as st

import config
import read_transcripts as rt

#work_dir = os.getcwd()
db = rt.load_embeddings(rt.ECO101, model = 'OpenAI')
llm = OpenAI(temperature=0.3, openai_api_key=config.OPENAI_API_KEY)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

def get_response(query):
    chunks = db.similarity_search(query)
    context = " If it is sufficient, answer with a single paragraph. If necessary, use more than one paragraph."
    return qa.run(query+context), chunks

def get_chunks(query):
    chunks = db.similarity_search(query)
    return chunks

def run_streamlit():
    st.title('Query ECO101 lecture transcripts.')
    col_pages, col_queries = st.columns([2, 5])
    with col_pages:
        page = st.radio("What do you want to do?", ["Ask a question", "Do something else"])

    if page == "Ask a question":

        st.write('Examples: What does it mean to thinkin about margins? What are the key factors to take into account when making decisions?')
        query = st.text_input("Your question:")
        if query == "":
            query = rt.standard_query
        response, chunks = get_response(query)
        st.write(response)
        st.write("Relevant lecture fragments:")
        for i, chunk in enumerate(chunks, start=1):
            content = chunk.page_content
            content = content.replace("\n\n", "\n")
            source = chunk.metadata['source']
            st.write(f"{i}{source}: \n {content}")
    if page == "Do something else":
        st.write("TBD - To Be Developed.")
        
run_streamlit()
#response, chunks = get_response("What is Whatcott about?")
#chunks = get_chunks("What is Whatcott about?")
#chunks = add_to_chunks(chunks)
#k=2    
