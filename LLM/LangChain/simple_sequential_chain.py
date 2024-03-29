import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
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
template_string_1 = "What is a {mathematical_tool}?"
prompt_1 = PromptTemplate.from_template(template_string_1)

template_string_2 = "What is a {programming_tool}?"
prompt_2 = PromptTemplate.from_template(template_string_2)

# defining a list of prompts
#tools = ["matrix", "vector", "tensor", "function", "derivative", "integral", "limit", "series", "equation", "inequality"] 

# creating a chain
chain1 = LLMChain(llm=llm, prompt = prompt_1)
chain2 = LLMChain(llm=llm, prompt = prompt_2)

# Create a chain that uses two LLM chains in a sequential manner
chain = SimpleSequentialChain(chains=[chain1, chain2], verbose = True)#, input_variables=["mathematical_tool", "programming_tool"])

chain.run("matrix")
