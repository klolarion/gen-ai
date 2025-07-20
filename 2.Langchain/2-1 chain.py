import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate

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


# chain
chain = prompt | llm | output_parser

# execute
result = chain.invoke({"input": "지구의 자전 주기는?"})

# print result
print(result)
