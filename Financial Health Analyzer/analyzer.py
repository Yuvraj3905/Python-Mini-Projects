import streamlit as st
import pandas as pd
import pdfplumber
import plotly.express as px
import re
from langchain_community.llms import Ollama

llm=Ollama(model="llama3.1") #initialize the LLM

def parse_pdf(file):
    all_data=[]
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            table=page.extract_table()
            if table:
                for row in table[1:]:
                    if len(row)>=5:
                        all_data.append(row)

    df=pd.DataFrame(all_data,columns=["Date","ID","Amount","Type","Balance", "Remarks"])
    df=df.dropna(subset=["Date","Amount"])
    df['Amount']=df['Amount'].str.replace(',','').astype(float)
    df['Date']=pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df=df.dropna(subset=['Date'])

    df['Remarks']=df['Remarks'].apply(lambda x: re.sub(r'\d{8,}','********',str(x)))
    return df


def get_ai_analysis(df):
    sample_data=df[['Remarks','Amount','Type']].tail(20).to_string()
    prompt=f"""
    Analyze these bank transactions and provide:
    1. Categorization for : {sample_data}
    2. Identify recurring subscriptions.
    3. Identify "Luxury/Impulse" purchases (>1000 INR).
    4. Provide a 'Stop Spend' list of 5 areas to save 15%

    Format as JSON with keys:
    'categories','subscriptions','luxury_purchases','stop_spend'
    """

    try:
        response=llm.invoke(prompt)
        return response
    except Exception as e:
        return f"AI Error: {str(e)}"

##---UI---##
st.title("Smart Spend & Financial Health Analyzer")
st.markdown("---")

uploaded_file=st.file_uploader("Upload your bank statement (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Processing Data Locally..."):
        raw_df=parse_pdf(uploaded_file)

        expenses_df=raw_df[raw_df['Type']=='DR'].copy()
        income_df=raw_df[raw_df['Type']=='CR'].copy()

        col1,col2,col3,col4=st.columns(4)
        total_in=income_df['Amount'].sum()
        total_out=expenses_df['Amount'].sum()
        net_savings=total_in-total_out

        col1.metric("Total Income",f"₹{total_in:,.2f}")
        col2.metric("Total Expenses",f"₹{total_out:,.2f}")
        col3.metric("Net Savings",f"₹{net_savings:,.2f}")
        col4.metric("Avg Daily Spend",f"₹{net_savings/30:,.2f}")

        st.subheader("Spending Trends")
        fig=px.line(raw_df.sort_values('Date'),x='Date',y='Balance', title="Balance Over Time")
        st.plotly_chart(fig,use_container_width=True)
        
        st.subheader("Amount Trends")
        fig=px.line(raw_df.sort_values('Date'),x='Date',y='Amount', title="Amount Over Time")
        st.plotly_chart(fig,use_container_width=True)

        st.markdown("---")
        st.header("AI Analysis")

        if st.button("Run AI Deep Drive"):
            with st.spinner("Running AI Deep Drive..."):
                analysis=get_ai_analysis(expenses_df)
                st.write(analysis)

        with st.expander("View Cleaned Transactions"):
            st.dataframe(raw_df, use_container_width=True)

        