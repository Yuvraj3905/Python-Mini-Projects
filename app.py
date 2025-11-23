import os
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
import operator

# LangChain / LangGraph Imports
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_experimental.utilities import PythonREPL

# 1. Setup & Config
load_dotenv()
st.set_page_config(page_title="Self-Healing AI Coder", layout="wide")

# --- CUSTOM CSS FOR VISUALS ---
st.markdown("""
<style>
    .stChatMessage { border-radius: 10px; padding: 10px; }
    .stSuccess { background-color: #d4edda; }
    .stError { background-color: #f8d7da; }
    .stCode { font-family: 'Fira Code', monospace; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Self-Healing Code Agent")
st.markdown("""
**Resume Project: Agentic Workflow with LangGraph**
This agent follows a cyclic workflow: **Plan → Code → Test → Fix**. 
It executes the code in a sandbox and uses the error trace to self-correct.
""")

# 2. Resources
@st.cache_resource
def get_llm():
    # Ensure API key is present
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Please set GROQ_API_KEY in .env file")
        st.stop()
    
    # Initialize Groq (Llama 3)
    return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

llm = get_llm()
repl = PythonREPL()

# 3. State Definition
# This dictionary carries the data between the Coder and Tester nodes
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    code: str
    iterations: int
    error: str
    success: bool

# 4. Nodes (The "Brain" logic)

def generator_node(state: AgentState):
    """Generates or Fixes Code based on the request or previous error."""
    messages = state['messages']
    error = state.get('error', "")
    code = state.get('code', "")
    iterations = state.get('iterations', 0)
    
    # Dynamic Prompting based on state (Fixing vs Creating)
    if error:
        print(f"--- FIXING ERROR (Attempt {iterations}) ---")
        prompt = f"""
        The previous code you wrote failed with this error:
        {error}
        
        Here is the broken code:
        {code}
        
        Please FIX the code. 
        IMPORTANT: Return ONLY the valid Python code. Do not include markdown formatting (like ```python). 
        Do not include explanations. Just the code.
        """
    else:
        print(f"--- GENERATING NEW CODE (Attempt {iterations}) ---")
        prompt = f"""
        You are a Python Expert. Write a Python script to solve this task:
        {messages[0].content}
        
        IMPORTANT: Return ONLY the valid Python code. Do not include markdown formatting (like ```python). 
        Do not include explanations. Just the code.
        """
    
    # Call LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Clean markdown if LLM adds it (Robustness)
    clean_code = response.content.replace("```python", "").replace("```", "").strip()
    
    return {
        "code": clean_code, 
        "iterations": iterations + 1, 
        "messages": [AIMessage(content=f"Generated Code (Attempt {iterations+1})")]
    }

def tester_node(state: AgentState):
    """Runs the code and checks for errors."""
    code = state['code']
    print(f"--- TESTING CODE ---")
    
    try:
        # We use PythonREPL to actually execute the code safely
        output = repl.run(code)
        
        # LangChain's REPL sometimes returns the error as the output string
        # This checks if the execution resulted in a traceback or error message
        if "Error" in output or "Traceback" in output:
            return {
                "error": output, 
                "success": False, 
                "messages": [AIMessage(content=f"Test Failed: {output}")]
            }
        
        # If no error, we assume success
        return {
            "error": "", 
            "success": True, 
            "messages": [AIMessage(content=f"Test Passed! Output: {output}")]
        }
        
    except Exception as e:
        # Catch execution errors
        return {
            "error": str(e), 
            "success": False, 
            "messages": [AIMessage(content=f"Execution Error: {e}")]
        }

# 5. Graph Definition (The "Workflow")
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("coder", generator_node)
workflow.add_node("tester", tester_node)

# Set entry point
workflow.set_entry_point("coder")

# Add edges
workflow.add_edge("coder", "tester")

# Define conditional routing
def router(state: AgentState):
    """Decides where to go next based on the Tester's results."""
    if state['success']:
        return "end"
    if state['iterations'] > 3: # Circuit breaker to prevent infinite loops
        return "max_retries"
    return "retry"

# Add conditional edges based on router output
workflow.add_conditional_edges(
    "tester",
    router,
    {
        "end": END,
        "max_retries": END,
        "retry": "coder"
    }
)

# Compile the graph
app = workflow.compile()

# 6. UI Logic
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Task")
    user_task = st.text_area(
        "Describe what you want the code to do:", 
        "Write a python script that calculates the factorial of 5 and prints the result."
    )
    start_btn = st.button("Start Agent Workflow", type="primary")

with col2:
    st.subheader("Agent Activity")
    output_container = st.container()

if start_btn:
    with output_container:
        with st.status("🚀 Agent Working...", expanded=True) as status:
            
            initial_state = {
                "messages": [HumanMessage(content=user_task)],
                "code": "",
                "iterations": 0,
                "error": "",
                "success": False
            }
            
            # Stream events from the graph
            for event in app.stream(initial_state):
                for key, value in event.items():
                    
                    if key == "coder":
                        iteration = value['iterations']
                        st.info(f"✍️ **Coder (Attempt {iteration}):** Generating/Fixing code...")
                        with st.expander("View Generated Code"):
                            st.code(value['code'], language='python')
                            
                    elif key == "tester":
                        if value['success']:
                            st.success("✅ **Tester:** Code executed successfully!")
                            st.code(f"Output:\n{value.get('messages')[-1].content}", language='text')
                        else:
                            st.error(f"❌ **Tester:** Error detected.")
                            st.warning(f"Sending error back to Coder for fix...")
                            with st.expander("View Error Trace"):
                                st.code(value['error'], language='text')
            
            status.update(label="Workflow Complete", state="complete", expanded=False)
        
        st.divider()
        st.subheader("Final Result")
        st.success("Job Done!")