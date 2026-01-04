# 🏦 Smart Spend & Financial Health Analyzer

A Streamlit-powered dashboard that analyzes your bank statements (PDF) using AI (Ollama/Llama 3.1) to provide insights into your spending habits, recurring subscriptions, and financial health.

## ✨ Features

- **📄 PDF Parsing**: Automatically extracts transaction data from bank statement PDFs.
- **📊 Interactive Dashboard**: Visualizes balance trends and transaction amounts over time.
- **💰 Financial Metrics**: Calculates total income, expenses, net savings, and average daily spend.
- **🤖 AI-Powered Analysis**: Uses Ollama (Llama 3.1) to:
    - Categorize transactions.
    - Identify recurring subscriptions.
    - Flag luxury or impulse purchases (>1000 INR).
    - Provide actionable "Stop Spend" recommendations to save 15%.
- **📑 Data Transparency**: View the cleaned transaction data in a searchable table.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Data Processing**: [Pandas](https://pandas.pydata.org/)
- **PDF Extraction**: [pdfplumber](https://github.com/jsvine/pdfplumber)
- **Visualization**: [Plotly](https://plotly.com/)
- **AI Engine**: [Ollama](https://ollama.com/) (running `llama3.1`)
- **LLM Orchestration**: [LangChain](https://www.langchain.com/)

## 🚀 Getting Started

### Prerequisites

1.  **Python 3.8+**
2.  **Ollama**: Install from [ollama.com](https://ollama.com/) and download the Llama 3.1 model:
    ```bash
    ollama run llama3.1
    ```

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd "Financial Health Analyzer"
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  Run the Streamlit application:
    ```bash
    streamlit run analyzer.py
    ```

2.  Open your browser and navigate to the URL provided (usually `http://localhost:8501`).
3.  Upload your bank statement PDF.
4.  Explore the charts and click **"Run AI Deep Drive"** for personalized financial insights.

## ⚠️ Important Note

This application processes all data **locally**. Your bank statements are parsed on your machine, and the AI analysis is performed using a local Ollama instance, ensuring your financial privacy.
