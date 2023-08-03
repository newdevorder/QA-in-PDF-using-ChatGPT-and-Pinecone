import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import pinecone 
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
from elevenlabs import generate, set_api_key
from elevenlabs.api import Voices


load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CREATIVITY = os.getenv('CREATIVITY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
STABILITY = os.getenv('STABILITY')
SIMILARITY_BOOST = os.getenv('SIMILARITY_BOOST')
FINE_TUNE_VOICES = os.getenv('FINE_TUNE_VOICES')
ENABLE_ELEVENTLABS = os.getenv('ENABLE_ELEVENTLABS')

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

# @st.cache_resource
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

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=CREATIVITY
)
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
    image_url = "https://raw.githubusercontent.com/newdevorder/QA-in-PDF-using-ChatGPT-and-Pinecone/main/image04wfe.jpeg"
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
    st.info("How do I generate solutions leveraging The New Dev Order?")
    st.info("Why is it important for Founders who want to build a solution to meet with a Bounty Manager first?")    
    st.info("How might I leverage the do I make money as a Developer in The New Dev Order?")
    st.info("How do I make money as a UX Designer in The New Dev Order?")
    st.info("Why is a Developer considered a Bounty Hunter in The New Dev Order?")
    st.info("What is the Bounty Design Document and why should I care as a Founder?")
    st.info("How do I join The New Dev Order?")
    text_input = st.text_input("What would you like to know? Type your question below...") 
    if st.button("Ask Question"):
        if len(text_input)>0:
            st.info("Your Query: " + text_input)
            answer = retrieval_answer(text_input)
            
            # Enable or Disable Eleven Labs
            if ENABLE_ELEVENTLABS:
                condensed_answer = answer[0:2]
                # New Code
                # Use ElevenLabs API to generate speech and play it
                if ELEVENLABS_API_KEY:
                    generate_and_play(audio_text=condensed_answer)
                else:
                    st.error("ElevenLabs API key not found. Please set the 'ELEVENLABS_API_KEY' environment variable.")

            st.success(answer)
            # End of New Code

# New Code
def generate_and_play(audio_text):
    set_api_key(ELEVENLABS_API_KEY)
    # Generate audio using ElevenLabs
    audio = generate(text=audio_text, voice=getVoice("Bella"), 
                     model="eleven_monolingual_v1")

    # Play the audio
    st.audio(audio, format="audio/wav", start_time=0, sample_rate=None)
# End of New Code

def getVoice(voice_name):
    # Get available voices from api.
    voices = Voices.from_api()
    found_voices = [voice for voice in voices if voice.name == voice_name]
    if len(found_voices) >= 1:
        found_voice=found_voices[0]
        if(FINE_TUNE_VOICES):
            found_voice.settings.stability = STABILITY
            found_voice.settings.similarity_boost = SIMILARITY_BOOST
        return found_voices[0]
    else:
        return voices[0]

if __name__ == "__main__":
    main()
