import openai
from langchain.llms import OpenAI

def load_api_key(filename):
    with open(filename, "r") as f:
        return f.read().strip()

filename = "key.txt"
api_key = load_api_key(filename)

llm = OpenAI(openai_api_key = api_key, temperature = 0.3)

print(llm.predict("What is a stiffness matrix?"))

