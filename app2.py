import io
import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain.chains.question_answering import load_qa_chain

from langchain.prompts import PromptTemplate



from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

import fitz  # PyMuPDF

#read and extract pdf text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_file = io.BytesIO(pdf.read())  # Create a file-like object from bytes
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# divide the text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 80)
    chunks = text_splitter.split_text(text)
    return chunks 

# Convert text chunks into vectors
def get_vector_store(text_chunks):
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings  = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    encoded_text_chunks = [chunk.encode('utf-8', 'ignore').decode('utf-8') for chunk in text_chunks]

    # Load the Chroma database from disk
    chroma_db = Chroma(persist_directory="data", 
                    embedding_function=embeddings,
                    collection_name="lc_chroma_demo")
    
    # Get the collection from the Chroma database
    collection = chroma_db.get()

    # If the collection is empty, create a new one
    if len(collection['ids']) == 0:
        # Create a new Chroma database from the documents
        chroma_db = Chroma.from_documents(
            documents=encoded_text_chunks, 
            embedding=embeddings, 
            persist_directory="data",
            collection_name="lc_chroma_demo"
        )

        # Save the Chroma database to disk
        chroma_db.persist()
        
        
# Get conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the questions as detailed as possible from provided context, make sure to provide all the details,
    if the answer is not in the provided context just say, "Answer is not available in the Given Context".
    Don't provide wrong answers.
    Context : \n {context}? \n
    Question : \n {question} \n

    Answer :

    """

    # Initialize model
    # To use langchains with LLaMa we have used LlamaCpp
    # https://python.langchain.com/docs/use_cases/code_understanding
    
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    model = LlamaCpp(
    model_path= "zephyr-7b-beta.Q4_0.gguf",
    n_ctx=1024,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
    )

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
	
    # create chain
    chain = load_qa_chain(model, chain_type="stuff", prompt = prompt)
    
    return chain


# w.r.t User Input
def user_input(user_question):
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings  = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    # Load the Chroma database from disk
    chroma_db = Chroma(persist_directory="data", 
                    embedding_function=embeddings,
                    collection_name="lc_chroma_demo")
    
    # Get the collection from the Chroma database
    # collection = chroma_db.get()

    docs = chroma_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    print(response)

    st.write("Reply : ", response["output_text"])


# Streamlit app
def main():
    st.set_page_config("Chat With Multiple PDF")
    st.header("Chat With Multiple PDF using Gemini")

    user_question = st.text_input("Ask Question from PDF Files Provided")

    if user_question:
        user_input(user_question)  # as soon as user inputs question, execute user_input function

    with st.sidebar:
        st.title("Menu :")
        pdf_docs = st.file_uploader("Upload your PDF files and Click on Submit and Process", accept_multiple_files=True)
        
        if pdf_docs is not None and st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done...")

if __name__ == "__main__":
    main()