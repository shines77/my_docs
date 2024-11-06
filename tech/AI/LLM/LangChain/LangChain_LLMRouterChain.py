import os
import warnings
# pip install -U python-dotenv
from dotenv import load_dotenv, find_dotenv

from operator import itemgetter
from typing import Literal
from typing_extensions import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE as RounterTemplate

_ = load_dotenv(find_dotenv())  # read local .env file
print('os.environ.URL = ', os.environ.get('URL'))

llm = ChatOpenAI(model="gpt-4o-mini")

prompt_1 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert on animals."),
        ("human", "{query}"),
    ]
)
prompt_2 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert on vegetables."),
        ("human", "{query}"),
    ]
)

chain_1 = prompt_1 | llm | StrOutputParser()
chain_2 = prompt_2 | llm | StrOutputParser()

route_system = "Route the user's query to either the animal or vegetable expert."
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", route_system),
        ("human", "{query}"),
    ]
)

class RouteQuery(TypedDict):
    """Route query to destination."""
    destination: Literal["animal", "vegetable"]

route_chain = (
    route_prompt
    | llm.with_structured_output(RouteQuery)
    | itemgetter("destination")
)

chain = {
    "destination": route_chain,  # "animal" or "vegetable"
    "query": lambda x: x["query"],  # pass through input query
} | RunnableLambda(
    # if animal, chain_1. otherwise, chain_2.
    lambda x: chain_1 if x["destination"] == "animal" else chain_2,
)

chain.invoke({"query": "what color are carrots"})
