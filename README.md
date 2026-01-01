# Python Mini Projects 🐍

A collection of autonomous AI agents and data analysis tools built with Python, LangChain, Streamlit, and modern LLMs.

## 🚀 Projects Overview

| Project | Description | Tech Stack |
| :--- | :--- | :--- |
| **[ChatWithPDF](file:///home/13843K/Desktop/mygitprojects/Python-Mini-Projects/ChatWithPDF)** | Talk to your PDF files locally using Ollama. | Streamlit, LangChain, Ollama, FAISS |
| **[HR-Companion](file:///home/13843K/Desktop/mygitprojects/Python-Mini-Projects/HR-Companion)** | TalentScout AI: Automated resume screening and ranking. | Streamlit, Groq (Llama 3), Supabase, HuggingFace |
| **[Self Healing Code Agent](file:///home/13843K/Desktop/mygitprojects/Python-Mini-Projects/Self%20Healing%20Code%20Agent)** | Autonomous agent that plans, writes, tests, and self-corrects code. | LangGraph, Groq (Llama 3), PythonREPL |
| **[Personal Finance Analyzer](file:///home/13843K/Desktop/mygitprojects/Python-Mini-Projects/Personal%20Finance%20Analyzer)** | Clean and categorize bank statements to track spending. | Pandas, Python |

---

## 🛠️ Detailed Project Descriptions

### 1. ChatWithPDF 💬
Interact with your documents without uploading them to the cloud. This project uses **Ollama** to run LLMs and embeddings locally, ensuring maximum privacy.
- **Features:** Multiple model selection (`llama3.1`, `qwen2.5`), local vector storage with FAISS, and an interactive UI.
- **Folder:** [ChatWithPDF](file:///home/13843K/Desktop/mygitprojects/Python-Mini-Projects/ChatWithPDF)

### 2. HR-Companion (TalentScout AI) 💼
A production-grade RAG system for HR teams. It extracts structured data from PDF resumes and ranks candidates against job descriptions using semantic search.
- **Features:** Hybrid search (SQL + Vector), automated candidate ranking, and an AI HR assistant.
- **Folder:** [HR-Companion](file:///home/13843K/Desktop/mygitprojects/Python-Mini-Projects/HR-Companion)

### 3. Self-Healing Code Agent 🤖
An agentic workflow built with **LangGraph** that follows a cyclic **Plan → Code → Test → Fix** pattern. It executes code in a sandbox and uses error traces to self-correct until the task is solved.
- **Features:** Cyclic reasoning, autonomous debugging, and visual workflow tracking.
- **Folder:** [Self Healing Code Agent](file:///home/13843K/Desktop/mygitprojects/Python-Mini-Projects/Self%20Healing%20Code%20Agent)

### 4. Personal Finance Analyzer 📊
An automation tool to process messy bank statement CSVs. It cleans data, handles overlapping dates, and categorizes transactions automatically.
- **Features:** Duplicate removal, automatic spending categorization, and master report generation.
- **Folder:** [Personal Finance Analyzer](file:///home/13843K/Desktop/mygitprojects/Python-Mini-Projects/Personal%20Finance%20Analyzer)

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) (Required for ChatWithPDF)
- API Keys: [Groq](https://console.groq.com/) and [Supabase](https://supabase.com/) (Required for HR-Companion and Code Agent)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Yuvraj3905/Python-Mini-Projects.git
   cd Python-Mini-Projects
   ```
2. Each project has its own `requirements.txt` and `.env.example`. Navigate to the specific project folder and set up a virtual environment:
   ```bash
   cd [Project-Name]
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Follow the specific `README.md` within each project folder for detailed setup.

## 🤝 Contributing
Feel free to open issues or submit pull requests to add new mini-projects or improve existing ones!
