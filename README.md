# Self-Healing Code Agent

A self-healing code generation system that leverages LangGraph to create an agentic loop between code generation and testing, ensuring the generated code is both syntactically correct and functionally accurate.

## 🚀 Features

- **Agentic Loop Architecture**: Implements a cyclic dependency between code generation and testing nodes using LangGraph
- **Self-Correcting**: Automatically detects and fixes code issues through iterative testing
- **Deterministic Validation**: Executes generated code in a sandbox environment to verify correctness
- **Streamlit Interface**: User-friendly web interface for interacting with the agent

## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Yuvraj3905/Self-Healing-Code-Agent
   cd Self-Healing-Code-Agent
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ How to Run

1. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

## 🧠 How It Works

The system implements an agentic loop that:

1. **Generates** code based on user requirements
2. **Tests** the generated code in a sandbox environment
3. **Analyzes** test results and any errors
4. **Self-corrects** by feeding errors back into the generation process
5. **Repeats** until the code passes all tests or reaches a maximum number of iterations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
