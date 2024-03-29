import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import time

def load_api_key(filename):
    with open(filename, "r") as f:
        return f.read().strip()

# load api key
filename = "key.txt"
api_key = load_api_key(filename)

# loading a LLM
llm = OpenAI(openai_api_key = api_key, temperature = 0.3)

# defining a template
template_string = "What is a {mathematical_tool}?"
prompt = PromptTemplate.from_template(template_string)

# defining a list of prompts
tools = ["matrix", "vector", "tensor", "function", "derivative", "integral", "limit", "series", "equation", "inequality"] 

# creating a chain
chain = LLMChain(llm=llm, prompt = prompt)
for tool in tools:
    output = chain.run(tool)
    print(output)
    time.sleep(2)

