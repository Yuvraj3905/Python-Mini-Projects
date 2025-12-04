import os
import json
import streamlit as st
from dotenv import load_dotenv
from typing import List
from supabase import create_client, Client
from pypdf import PdfReader

# LangChain Imports
import sentence_transformers 
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 1. Configuration & Setup
load_dotenv()
st.set_page_config(page_title="TalentScout AI", layout="wide")

# Initialize Supabase
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

if not url or not key:
    st.error("Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_KEY in your .env file.")
    st.stop()

supabase: Client = create_client(url, key)

# Initialize Models with Caching
@st.cache_resource
def load_models():
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Groq API Key not found. Please set GROQ_API_KEY in your .env file.")
        st.stop()

    llm_instance = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    embeddings_instance = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    return llm_instance, embeddings_instance

llm, embeddings = load_models()

# --- DATA MODELS ---
class CandidateProfile(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Email address")
    years_of_experience: int = Field(description="Total years of professional experience (integer)")
    skills: str = Field(description="Comma separated list of top technical skills")
    summary: str = Field(description="A brief summary of the resume")

# --- HELPER FUNCTIONS ---

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def parse_resume_to_json(text):
    """Uses LLM to extract structured data from unstructured text."""
    parser = JsonOutputParser(pydantic_object=CandidateProfile)
    prompt = PromptTemplate(
        template="Extract the following information from the resume text.\n{format_instructions}\n\nResume Text:\n{text}",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    try:
        return chain.invoke({"text": text[:4000]}) 
    except Exception as e:
        st.error(f"Error parsing resume: {e}")
        return None

def ingest_resume(file):
    """Pipeline: PDF -> Text -> JSON Extraction -> Embedding -> Database"""
    text = extract_text_from_pdf(file)
    
    # 1. Extract Metadata using LLM
    profile = parse_resume_to_json(text)
    if not profile:
        st.error(f"Failed to parse resume: {file.name}")
        return None

    # 2. Create Embedding
    vector = embeddings.embed_query(text)

    # 3. Insert into Supabase
    data = {
        "name": profile['name'],
        "email": profile['email'],
        "years_of_experience": profile['years_of_experience'],
        "skills": profile['skills'],
        "summary": profile['summary'], # UPDATED: Now saving the summary!
        "content": text, 
        "embedding": vector
    }
    
    try:
        response = supabase.table("candidates").insert(data).execute()
        return profile['name']
    except Exception as e:
        st.error(f"Database Error: {e}")
        return None

def fetch_all_candidates():
    """Fetches all candidates for display in the UI"""
    try:
        # UPDATED: Select summary as well
        response = supabase.table("candidates").select("name, email, years_of_experience, skills, summary").execute()
        return response.data
    except Exception as e:
        st.error(f"Error fetching candidates: {e}")
        return []

def rank_candidates(job_description):
    """Semantic Search against the Job Description"""
    query_vector = embeddings.embed_query(job_description)
    
    try:
        response = supabase.rpc(
            "match_candidates", 
            {
                "query_embedding": query_vector,
                "match_threshold": 0.5, 
                "match_count": 5
            }
        ).execute()
        return response.data
    except Exception as e:
        st.error(f"Search Error: {e}")
        return []

def generate_hr_report(candidate, job_description):
    """Generates an explanation of why this candidate fits"""
    prompt = f"""
    You are an Expert HR Recruiter.
    
    Job Description:
    {job_description}
    
    Candidate Resume Summary:
    {candidate['content'][:3000]}
    
    Task: Write a concise 3-bullet point assessment of why this candidate is a good or bad fit.
    """
    response = llm.invoke(prompt)
    return response.content

def chat_with_db(query):
    """
    Advanced RAG: Hybrid approach using SQL for listing/filtering and Vector for semantic search.
    """
    query_lower = query.lower()
    
    # 1. DIRECT DATABASE LOOKUP 
    if any(k in query_lower for k in ["list", "names", "show", "all candidates", "who are", "candidates we have"]):
        try:
            data = supabase.table("candidates").select("name, years_of_experience, skills, summary").limit(20).execute().data
            
            if not data:
                return "The database is currently empty. Please upload some resumes first."
            
            # Helper to safely get summary
            def get_summary(c):
                return c.get('summary', 'No summary available') or 'No summary available'
                
            context = "\n".join([f"- {c['name']} ({c['years_of_experience']} yrs): {c['skills']} | Summary: {get_summary(c)}" for c in data])
            prompt = f"User Query: {query}\n\nDatabase Content (All Candidates):\n{context}\n\nAnswer the user based on the database content:"
            return llm.invoke(prompt).content
        except Exception as e:
            return f"Database Error: {e}"

    # 2. SQL FILTERING 
    elif "year" in query_lower and any(char.isdigit() for char in query):
        try:
            data = supabase.table("candidates").select("name, years_of_experience, skills").execute().data
            
            context = "\n".join([f"Name: {c['name']}, Exp: {c['years_of_experience']} years, Skills: {c['skills']}" for c in data])
            prompt = f"User Query: {query}\n\nCandidate Data:\n{context}\n\nAnswer the user based on the data:"
            return llm.invoke(prompt).content
        except Exception as e:
            return f"Database Error: {e}"

    # 3. VECTOR SEARCH 
    else:
        query_vector = embeddings.embed_query(query)
        try:
            response = supabase.rpc("match_candidates", {"query_embedding": query_vector, "match_threshold": 0.4, "match_count": 5}).execute()
            data = response.data
            
            # FALLBACK
            if not data:
                 # UPDATED: Select summary safely
                 data = supabase.table("candidates").select("name, skills, summary").limit(5).execute().data
                 context = "No direct semantic matches found. Here are some random candidates from the DB:\n" + "\n".join([f"- {c['name']}: {c['skills']}" for c in data])
            else:
                 context = "\n".join([f"Name: {c['name']}, Summary: {c.get('summary', 'No summary')}..." for c in data])
            
            prompt = f"User Query: {query}\n\nRelevant Context:\n{context}\n\nAnswer the user:"
            return llm.invoke(prompt).content
        except Exception as e:
            return f"Vector Search Error: {e}"

# --- UI LAYOUT ---

st.title("💼 TalentScout AI: Production RAG System")
st.markdown("Automated Resume Screening with **Structured Extraction** & **Hybrid Search**.")

tabs = st.tabs(["📤 Upload Resumes", "📊 Rank & Screen", "💬 HR Assistant"])

# TAB 1: UPLOAD & VIEW
with tabs[0]:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Ingest Candidates")
        uploaded_files = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)
        
        if st.button("Process Resumes"):
            if not uploaded_files:
                st.warning("Please upload files first.")
            else:
                progress_bar = st.progress(0)
                for i, file in enumerate(uploaded_files):
                    name = ingest_resume(file)
                    if name:
                        st.success(f"Ingested: {name}")
                    progress_bar.progress((i + 1) / len(uploaded_files))
                st.success("Processing Complete!")
                st.rerun() 

    with col2:
        st.header("Current Talent Pool")
        existing_candidates = fetch_all_candidates()
        if existing_candidates:
            st.dataframe(existing_candidates)
        else:
            st.info("No candidates in database yet.")

# TAB 2: RANKING
with tabs[1]:
    st.header("Screen Candidates against JD")
    jd = st.text_area("Paste Job Description (JD):", height=200)
    
    if st.button("Rank Candidates"):
        if not jd:
            st.warning("Please provide a JD.")
        else:
            with st.spinner("Analyzing Database..."):
                ranked_candidates = rank_candidates(jd)
                
                if not ranked_candidates:
                    st.info("No matching candidates found.")
                
                for rank, candidate in enumerate(ranked_candidates):
                    with st.expander(f"#{rank+1}: {candidate['name']} (Match: {round(candidate['similarity']*100, 1)}%)"):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.write(f"**Experience:** {candidate['years_of_experience']} Years")
                            st.write(f"**Skills:** {candidate['skills']}")
                        with col2:
                            report = generate_hr_report(candidate, jd)
                            st.markdown(f"**AI Assessment:**\n{report}")

# TAB 3: HR ASSISTANT
with tabs[2]:
    st.header("Ask questions about your talent pool")
    st.info("Try asking: 'Who has more than 5 years of experience?' or 'Do we have any React developers?'")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask the AI Recruiter..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Analyzing talent pool..."):
            response = chat_with_db(prompt)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})