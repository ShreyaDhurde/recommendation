import streamlit as st
import pandas as pd
import os
from langchain_experimental.agents import create_csv_agent
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the LLM (e.g., Gemini via Google Generative AI)
#llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
# Define the initial prompt template
initial_prompt_template = PromptTemplate(
    input_variables=["query"],
    template="""
        You are a recommendation engine expert. 

        Based on the following query, provide personalized product recommendations.
        use collaborative and content-based filtering techniques for recommendations .

        suggest the products that are
         present in similar other custormes list but not purchased yet
        {query}

        Return the top recommendations of product descriptions ("Description") 
        show recommendation in a table format.
        give the top 5 products that the CustomerID is most likey to buy. 
        
    """
)

def generate_recommendations(query, csv_agent):
    # Format the prompt
    prompt = initial_prompt_template.format(query=query)
    
    # Create a query for the CSV agent using the formatted prompt
    full_query = f"Based on the data and the prompt, provide recommendations:\n\n{prompt}"
    
    # Run the query through the agent
    response = csv_agent.invoke(full_query)
    return response["output"]

# Streamlit app setup
st.set_page_config(page_title="CSV Query Tool", layout="wide")
st.title("Product recommendation bot")

# File uploader for a single CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    # Ensure the 'temp' directory exists
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Save the uploaded file temporarily
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load the CSV into a DataFrame
    df = pd.read_csv(temp_file_path)
    
    # Display the DataFrame
    st.write("### Uploaded Data:")
    st.dataframe(df)

    # Create CSV Agent with the uploaded CSV file
    csv_agent = create_csv_agent(
        llm,
        temp_file_path,
        verbose=False,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True,
        max_iterations=60 
    )

    # Query input field
    query = st.text_input("Enter your query:")

    # Generate recommendations
    if st.button("Get Recommendations") and query:
        response = generate_recommendations(query, csv_agent)
        st.write("### Recommendations:")
        st.write(response)
    elif query:
        st.write("### Recommendations:")
        st.write("Please click the button to get recommendations.")