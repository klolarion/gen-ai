import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI, ChatOpenAI

# load env
load_dotenv()

# API KEY
api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI()


# LLM
# 기능: 텍스트 문자열을 입력으로 받아 처리한 후, 텍스트 문자열을 반환한다.
# 예를 들어, 문서 요약, 콘텐츠 생성, 질문에 대한 답변 생성 등 복잡한 자연어 처리 작업을 수행할 수 있다.
result = llm.invoke("한국의 대표적인 관광지 3군데를 추천해주세요.")
print(result)



# Chat Model
# 기능: Chat Model 클래스는 메시지의 리스트를 입력으로 받고, 하나의 메시지를 반환한다.
# 대화형 상황에 최적화되어 있으며, 사용자와의 연속적인 대화를 처리하는 데 사용된다.
# Chat Model은 이전 대화를 기억하고 대화의 맥락을 유지하면서 적절한 응답을 생성하는 데 중점을 둔다.
# 챗봇, 가상 비서, 고객 지원 시스템 등에 활용된다.

chat = ChatOpenAI()

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "이 시스템은 여행 전문가입니다."),
    ("user", "{user_input}"),
])

chain = chat_prompt | chat
chain.invoke({"user_input": "안녕하세요? 한국의 대표적인 관광지 3군데를 추천해주세요."})