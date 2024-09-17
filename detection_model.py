import json
from langchain.chains import LLMChain
from langchain_together import Together
from sklearn.model_selection import train_test_split
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Load Model for Fine-Tuning
def load_model(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"):
    llm = Together(
        model=model_name,
        temperature=0.7,
        max_tokens=256,
        top_k=1,
        together_api_key="b4bb49cc0500d5e4bebbe719649c88f452d099ed4814313aa3156bda74fbb50e"  
    )
    return llm

# System prompt template for hallucination detection
def system_prompt():
    system_template = (
        "You are a classifier which detects whether the answer is hallucinated or not. "
        "If hallucinated, return the hallucinated part of the answer. "
        "If not hallucinated, return 'Not hallucinated'."
    )
    return SystemMessagePromptTemplate.from_template(system_template)

# Human prompt template
def human_prompt(question, answer):
    human_template = f"Question: {question}\nAnswer: {answer}\n"
    return HumanMessagePromptTemplate.from_template(human_template)

# Fine-tune model
def fine_tune_model():
    # Load the dataset
    with open('qa_data.json') as f:
        data = [json.loads(line) for line in f]

    # Split into training and testing datasets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Load the model
    llm = load_model()

    # Train the model using the fine-tuning method
    for entry in train_data:
        question = entry["question"]
        answer = entry["hallucinated_answer"]
        right_answer = entry["right_answer"]

        # Create the prompts for training
        prompt = human_prompt(question, answer)
        chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        chain.run(text=prompt)

    return llm  # Return the fine-tuned model

# Detection function using the fine-tuned model
def detect_hallucination(question, answer, model):
    prompt = (
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        "Is the answer factual and correct? (yes or no): "
    )

    system_template = system_prompt()
    chat_prompt = system_template.format_prompt(text=prompt)
    chain = LLMChain(llm=model, prompt=chat_prompt, verbose=True)
    response = chain.run(text=prompt)

    # Process the response to find the hallucinated part
    if "Not hallucinated" in response:
        return False, None
    elif "Hallucinated:" in response:
        hallucinated_part = response.split("Hallucinated:")[1].strip().split('.')[0].strip()
        return True, hallucinated_part
    else:
        return None, None

# Generic usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune and detect hallucinations.")
    parser.add_argument('--question', type=str, required=True, help='The question to be asked.')
    parser.add_argument('--answer', type=str, required=True, help='The answer to be evaluated.')
    
    args = parser.parse_args()

    model = fine_tune_model()
    hallucinated, hallucinated_part = detect_hallucination(args.question, args.answer, model)
    print(f"Hallucinated: {hallucinated}, Part: {hallucinated_part}")
