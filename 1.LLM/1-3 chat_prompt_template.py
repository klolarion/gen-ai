import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI


# load env
load_dotenv()

# API KEY
api_key = os.getenv("OPENAI_API_KEY")

# model
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

# parser
output_parser = StrOutputParser()


# Message 유형

# SystemMessage: 모델에게 역할 또는 응답 방식 등을 지시함 -> 프롬프트 시작 부분
# HumanMessage: 사용자의 질문 또는 명령을 전달함 -> 사용자 입력
# AIMessage: 이전 대화에서 AI가 답변했던 내용을 나타냄 -> 대화 맥락 유지
# FunctionMessage: 함수 호출의 결과 데이터를 나타냄 -> 함수 호출 응답 시
# ToolMessage: 	외부 도구의 결과 출력을 제공함 -> 검색, DB, 계산기 등

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "이 시스템은 천문학 질문에 답변할 수 있습니다"),
    ("user", "{user_input}")
])

# 포맷 확인
# messages = chat_prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")
# print(messages)

chain = chat_prompt | llm | output_parser

result = chain.invoke({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"})
print(result)


# MessagePromptTemplate

message_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("이 시스템은 천문학 질문에 답변할 수 있습니다"),
    HumanMessagePromptTemplate.from_template("{user_input}"),
])

# 포맷 확인
# messages = chat_prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")
# print(messages)

chain = message_prompt | llm | output_parser
result = chain.invoke({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"})
print(result)