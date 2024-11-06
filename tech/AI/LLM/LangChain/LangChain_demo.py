# from langchain.chat_models import AzureChatOpenAI
from langchain.chat_models import init_chat_model
import os
import warnings
# pip install -U python-dotenv
from dotenv import load_dotenv, find_dotenv

warnings.filterwarnings('ignore')

# 设置 OpenAI API 密钥
# os.environ["OPENAI_API_KEY"] = 'Your Key'

# read local .env file
_ = load_dotenv(find_dotenv())
print('os.environ.URL = ', os.environ.get('URL'))

# 构建两个场景的模板
flower_care_template = """
你是一个经验丰富的园丁，擅长解答关于养花育花的问题。
下面是需要你来回答的问题:
{input}
"""

flower_deco_template = """
你是一位网红插花大师，擅长解答关于鲜花装饰的问题。
下面是需要你来回答的问题:
{input}
"""

# 构建提示信息
prompt_infos = [
    {
        "key": "flower_care",
        "description": "适合回答关于鲜花护理的问题",
        "template": flower_care_template,
    },
    {
        "key": "flower_decoration",
        "description": "适合回答关于鲜花装饰的问题",
        "template": flower_deco_template,
    }
]

# 初始化语言模型
# from langchain.llms import OpenAI
# llm = OpenAI()
# llm = AzureChatOpenAI(deployment_name="GPT-4", temperature=0)
llm = init_chat_model("GPT-4", temperature=0)

# 构建目标链
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

chain_map = {}

for info in prompt_infos:
    prompt = PromptTemplate(
        template=info['template'],
        input_variables=["input"]
    )
    print("目标提示:\\n", prompt)

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True
    )
    chain_map[info["key"]] = chain

# 构建路由链
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE as RounterTemplate

destinations = [f"{p['key']}: {p['description']}" for p in prompt_infos]
router_template = RounterTemplate.format(destinations="\\n".join(destinations))
print("路由模板:\\n", router_template)

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
print("路由提示:\\n", router_prompt)

router_chain = LLMRouterChain.from_llm(
    llm,
    router_prompt,
    verbose=True
)

# 构建默认链
from langchain.chains import ConversationChain

default_chain = ConversationChain(
    llm=llm,
    output_key="text",
    verbose=True
)

# 构建多提示链
from langchain.chains.router import MultiPromptChain

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=chain_map,
    default_chain=default_chain,
    verbose=True
)

# 测试1
print(chain.run("如何为玫瑰浇水？"))
