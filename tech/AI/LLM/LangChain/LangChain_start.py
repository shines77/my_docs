import os
import warnings
# pip install -U python-dotenv
from dotenv import load_dotenv, find_dotenv

from langchain_openai import ChatOpenAI

_ = load_dotenv(find_dotenv())  # read local .env file
print('os.environ.URL = ', os.environ.get('URL'))

llm = ChatOpenAI()
llm.invoke("Hello, world!")
