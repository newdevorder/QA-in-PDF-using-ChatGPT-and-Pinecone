from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import pinecone 
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
from elevenlabs import generate, play


load_dotenv()


PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


def doc_preprocessing():
    loader = DirectoryLoader(
        'data/',
        glob='**/*.pdf',     # only the PDFs
        show_progress=True
    )
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split

@st.cache_resource
def embedding_db():
    # we use the openAI embedding model
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    docs_split = doc_preprocessing()
    doc_db = Pinecone.from_documents(
        docs_split, 
        embeddings, 
        index_name='newdevorder'
    )
    return doc_db

llm = ChatOpenAI()
doc_db = embedding_db()

def retrieval_answer(query):
    qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type='stuff',
    retriever=doc_db.as_retriever(),
    )
    query = query
    result = qa.run(query)
    return result

def main():
    # NEW CODE
    
    # Define custom CSS to center the image and adjust its size
    custom_css = """
    <style>
    .centeredImage {
    display: flex;
    justify-content: center;
    }

    .centeredImage img {
    max-width: 10%; /* Change this value to adjust the image size */
    }
    </style>
    """

    # Display the custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)
    
    # END OF NEW CODE

    st.title("The New Dev Order Chatbot")

    # NEW CODE
    # Wrap the image inside a div with the "centeredImage" class
    st.markdown('<div class="centeredImage"><img src="' + image_url + '"></div>', unsafe_allow_html=True)
    # END OF NEW CODE

    st.info("You can ask a question like...")
    st.info("- How do I generate solutions leveraging The New Dev Order?")
    st.info("- How might I leverage the do I make money as a Developer in The New Dev Order?")
    st.info("- How do I make money as a UX Designer in The New Dev Order?")
    st.info("- How do I join The New Dev Order?")    
    text_input = st.text_input("What would you like to know? Type your question below...") 
    if st.button("Ask Question"):
        if len(text_input)>0:
            st.info("Your Query: " + text_input)
            answer = retrieval_answer(text_input)
            st.success(answer)

if __name__ == "__main__":
    main()

    







