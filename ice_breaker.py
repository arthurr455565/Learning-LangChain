import os
from dotenv import load_dotenv
load_dotenv()

# Disable LangSmith tracking to avoid warnings
os.environ["LANGCHAIN_TRACING_V2"] = "false"
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain

if __name__ == "__main__":
    print("Hello LangChain")
    
    information = (
        "Elon Musk is a business magnate and investor. "
        "He is the founder, chairman, CEO, and CTO of SpaceX; "
        "angel investor, CEO, and product architect of Tesla, Inc.; "
        "owner, chairman, and CTO of X Corp.; "
        "founder of The Boring Company; "
        "co-founder of Neuralink and OpenAI; "
        "and president of the Musk Foundation."
    )
    
    summary_template = """
        Given the information: {information}
        Create:
        1. A short summary
        2. Two interesting facts about them
    """
    
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )
    
    # Updated model name - using Llama 3.3 70B as replacement
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    
    chain = summary_prompt_template | llm
    
    res = chain.invoke(input={"information": information})
    print(res)