import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.agents import AgentType, initialize_agent, load_tools
import time

def load_api_key(filename):
    with open(filename, "r") as f:
        return f.read().strip()

# load api key
filename = "key.txt"
api_key = load_api_key(filename)

# loading a LLM
llm = OpenAI(openai_api_key = api_key, temperature = 0.3)

# tools
tools = load_tools(["wikipedia", "llm-math"], llm = llm)
# agent
agent = initialize_agent(tools, llm, agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True)

agent.run("At what age did Issac Newton invented the calculus?")
