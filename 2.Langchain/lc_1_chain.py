import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
# ============================================================================
# Chain: 프롬프트 → LLM → 파서를 파이프(|)로 연결하여 순차 실행하는 구조
# ============================================================================

# load env
load_dotenv()

# API KEY
api_key = os.getenv("OPENAI_API_KEY")

# model
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

# parser
output_parser = StrOutputParser()

# prompt
prompt = ChatPromptTemplate(
    input_variables=['input'],
    messages=[HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=['input'],
            template="You are an expert in astronomy. Answer the question. <Question>:{input}")
    )],
)


# chain - | 파이프 연산자 사용. 순서대로 연결되어 실행되어 체인처럼 사용됨.
chain = prompt | llm | output_parser

# execute
result = chain.invoke({"input": "지구의 자전 주기는?"})

# print result
print(result)
