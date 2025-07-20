import os
from langchain_openai import OpenAI
from dotenv import load_dotenv

# load env
load_dotenv()

# API KEY
api_key = os.getenv("OPENAI_API_KEY")

# model
llm = OpenAI(model="gpt-4o-mini", api_key=api_key)

# execute
result = llm.invoke("지구의 자전 주기는?")

# print result
print(result)


