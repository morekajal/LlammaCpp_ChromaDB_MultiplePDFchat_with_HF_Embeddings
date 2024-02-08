# LlammaCpp_ChromaDB_MultiplePDFchat_with_HF_Embeddings
1. chromadb_with_langchain.ipynb :
  - worked with Langchain's DocumentLoader,  RecursiveCharacterTextSplitter, SentenceTransformerEmbeddings and ChromaDBVectorStore
  - used Langchain's chains.question_answering to extract the answer for the query using openai_chat_api as llm model

2. chromadb_llamacpp_generation.ipynb : 
  - Instead of using OpenAI API Key, used Hugging face Embeddings (BGE Embeddings) and the Llama_cpp - zephyr-7b-beta.Q4_0.gguf quantized model for inference
  - Along with Local Chroma Vector Database to store the embeddings and
  - Retrieved the query embeddings from the same vectorDB.

3. app2.py:
   - Build a end-to-end streamlit application using langchain's RecursiveCharacterTextSplitter, Chroma vectorDB, HuggingFaceBgeEmbeddings
   - Used chains.question_answering to set the question-answering chain with 'stuff' chain type
     
