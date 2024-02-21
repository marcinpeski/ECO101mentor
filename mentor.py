#Activate using .\myenv\Scripts\activate in terminal.
#Deactivate using deactivate in terminal.

import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI


import streamlit as st

import config
import read_transcripts as rt

#work_dir = os.getcwd()
db = rt.db
llm = rt.llm
qa = rt.qa

def get_chunks(query):
    chunks = db.similarity_search(query)
    return chunks

def run_streamlit():
    st.title('Query ECO101 lecture transcripts.')
    page = st.radio("What do you want to do?", 
                    ["Ask a question", "Explain content relevance", "Do something else"])
    if page == "Ask a question":
        st.write('Examples: What does it mean to thinkin about margins? What are the key factors to take into account when making decisions?')
        query = st.text_input("Your question:")
        if query == "":
            query = rt.standard_query
        response, chunks = rt.get_response(db, qa, query)
        st.write(response)
    if page == "Explain content relevance":
        st.write('Ask a question. Mentor will find relevant transcript content and explain its relevance to your question.')
        st.write('Examples: What does it mean to thinkin about margins? What are the key factors to take into account when making decisions?')
        query = st.text_input("Your question:")
        if query == "":
            query = rt.standard_query
        response, chunks = rt.get_response(db, qa, query)
        st.write("Relevant lecture fragments:")
        for i, chunk in enumerate(chunks, start=1):
            content = chunk.page_content
            content = content.replace("\n\n", "\n")
            source = chunk.metadata['source']
            explanation = rt.explanation_chain.invoke({"content":content, "query":query})
            st.write(f"**{i} {source}**: {explanation} \n {content}")
    if page == "Do something else":
        st.write("TBD - To Be Developed.")
        
run_streamlit()
#response, chunks = get_response("What is Whatcott about?")
#chunks = get_chunks("What is Whatcott about?")
#chunks = add_to_chunks(chunks)
#k=2    
