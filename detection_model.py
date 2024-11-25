from langchain.chains import LLMChain
from langchain_together import Together
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

## Load Model
def load_model(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"):
    llm = Together(
        model=model_name,
        temperature=0.7,
        max_tokens=256,
        top_k=1,
        together_api_key="b4bb49cc0500d5e4bebbe719649c88f452d099ed4814313aa3156bda74fbb50e"  
    )
    return llm

# System prompt template
def system_prompt():
    system_template = (
        "You are a classifier which detects that the answer is hallucinated or not. "
        "If hallucinated, return the hallucinated part of the answer in the format: 'Hallucinated: <hallucinated part>'. "
        "If not hallucinated, return 'Not hallucinated'."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    return system_message_prompt

# Human prompt template
def human_prompt(text):
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    return human_message_prompt.format(text=text)

# Chat prompt template
def create_chat_prompt(text):
    system_query = system_prompt()
    human_query = human_prompt(text)
    chat_prompt = ChatPromptTemplate.from_messages([system_query, human_query])
    return chat_prompt

def detect_hallucination(question, answer):
    # Create the prompt
    prompting = (
        "Question: " + question + "\n"
        "Answer: " + answer + "\n"
        "Is the answer factual and correct? (yes or no): "
    )
    
    chat_prompt = create_chat_prompt(prompting)
    llm = load_model()
    chain = LLMChain(llm=llm, prompt=chat_prompt, verbose=True)
    response = chain.run(text=prompting)
    
    # Process the response to find the hallucinated part
    if "Not hallucinated" in response:
        return False, None
    elif "Hallucinated:" in response:
        # Split the response to extract the hallucinated part
        hallucinated_part = response.split("Hallucinated:")[1].strip()  # Extract the part after 'Hallucinated:'
        # Further clean up by removing any lingering text after the hallucinated part
        hallucinated_part = hallucinated_part.split('.')[0].strip()  # Stop at the first period or another delimiter
        return True, hallucinated_part
    else:
        return None, None  # Inconclusive response
