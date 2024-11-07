import os
import warnings
# pip install -U python-dotenv
from dotenv import load_dotenv, find_dotenv

from langchain_openai import ChatOpenAI

# read local .env file
# 默认的文件路径是 '.env', 可在 find_dotenv(env_filepath) 中指定文件和路径.
_ = load_dotenv(find_dotenv())
print('os.environ.OPENAI_URL = ', os.environ.get('OPENAI_URL'))

llm = ChatOpenAI()
llm.invoke("Hello, world!")
