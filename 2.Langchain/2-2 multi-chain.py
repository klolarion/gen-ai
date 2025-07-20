import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# load env
load_dotenv()

# API KEY
api_key = os.getenv("OPENAI_API_KEY")

# model
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

# parser
output_parser = StrOutputParser()

# prompt
prompt1 = ChatPromptTemplate.from_template("translates {korean_word} to English.")
prompt2 = ChatPromptTemplate.from_template("explain {english_word} using oxford dictionary to me in Korean.")

# chain
chain1 = prompt1 | llm | output_parser
chain2 = ({"english_word": chain1} | prompt2 | llm | output_parser)

# execute
result = chain2.invoke({"korean_word": "미래"})

# print result
print(result)