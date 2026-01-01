# Langchain Ask PDF (Ollama Edition)

This is a Python application that allows you to load a PDF and ask questions about it using natural language. The application uses **Ollama** for local LLM processing and embeddings, ensuring your data stays private and runs entirely on your machine.

## How it works

The application reads the PDF and splits the text into smaller chunks. It uses **Ollama embeddings** (`nomic-embed-text`) to create vector representations of the chunks. The application then finds the chunks that are semantically similar to your question and feeds those chunks to a local LLM (like `llama3.1`) to generate a response.

The application uses **Streamlit** for the GUI and **Langchain** for the LLM orchestration.

## Installation

1. **Clone the repository.**
2. **Setup Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Ollama Setup:**
   Ensure you have [Ollama](https://ollama.com/) installed and running. Pull the required models:
   ```bash
   ollama pull llama3.1
   ollama pull nomic-embed-text
   ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```


## Contributing

This repository is for educational purposes only and is not intended to receive further contributions.


