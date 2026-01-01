from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_classic.chains.question_answering import load_qa_chain


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF", page_icon="💬", layout="wide")
    
    # Custom CSS for better aesthetics
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .header-style {
        font-size: 40px;
        font-weight: bold;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="header-style">Ask your PDF 💬</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("Settings 🛠️")
        st.info("This app uses Ollama for local LLM processing.")
        st.markdown("---")
        model_name = st.selectbox("Select Model", ["llama3.1", "qwen2.5:1.5b", "llama2"], index=0)
        embed_model = st.selectbox("Select Embedding Model", ["nomic-embed-text"], index=0)
        st.markdown("---")

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings
      with st.spinner("Processing PDF and creating knowledge base..."):
          embeddings = OllamaEmbeddings(model=embed_model)
          knowledge_base = FAISS.from_texts(chunks, embeddings)
          st.success("Knowledge base ready!")
      
      # show user input
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        with st.spinner("Thinking..."):
            docs = knowledge_base.similarity_search(user_question)
            
            llm = Ollama(model=model_name)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.invoke({"input_documents": docs, "question": user_question})
               
            st.markdown("### Answer:")
            st.write(response["output_text"])
    

if __name__ == '__main__':
    main()
