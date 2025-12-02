import os
import requests
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults

# load env
load_dotenv()

# API KEY
api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
naver_client_id = os.getenv("NAVER_CLIENT_ID")
naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")



# ============================================================================
# Tool Calling: @tool 데코레이터로 함수를 Tool로 변환하고, bind_tools()로 LLM에 바인딩
# LLM이 필요시 자동으로 Tool을 호출하여 외부 기능(검색, 계산 등)을 사용
# 
# description을 상세히 작성해야 LLM이 이 함수를 언제 사용할지 알 수 있다.
# ============================================================================
# 도구 함수 정의
# ============================================================================

@tool
def tavily_search(query: str) -> str:
    # 아래 부분이 description으로 LLM에 전달된다.
    """
    Search the web for information using Tavily search engine.
    Returns search results as a formatted string.
    Use web search in English.    
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    tavily_client = TavilySearchResults(api_key=tavily_api_key, max_results=3)
    results = tavily_client.invoke(query)
    # 리스트를 문자열로 변환
    if isinstance(results, list):
        return "\n\n".join([str(result) for result in results])
    return str(results)

@tool
def naver_search(query: str) -> str:
    """
    Search the web for information using Naver search API.
    Returns blog search results as JSON string.
    """
    url = "https://openapi.naver.com/v1/search/blog.json"
    naver_client_id = os.getenv("NAVER_CLIENT_ID")
    naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")
    headers = {
        "X-Naver-Client-Id": naver_client_id,
        "X-Naver-Client-Secret": naver_client_secret,
    }
    params = {
        "query": query,
        "display": 10,
        "start": 1,
    }
    response = requests.get(url, headers=headers, params=params)
    return response.text



# 러너블람다 사용하기
from langchain_core.runnables import RunnableLambda

runnable_lambda = RunnableLambda(lambda x: x + " World")
result = runnable_lambda.invoke("Hello")
print(result)


# ============================================================================
# 메인 실행 코드: Tool Calling 예제
# ============================================================================

chat_llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

query = "맥켈란 12년의 오늘 최저, 최고 가격을 알려줘."
today_date = datetime.now().strftime("%Y-%m-%d")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can search the web for information."),
    ("system", "Today's date is {today_date}"),
    ("user", "{user_input}"),
])

# 도구 객체들을 딕셔너리에 저장 (도구 이름으로 매핑)
tools_dict = {
    'tavily_search': tavily_search,
    'naver_search': naver_search,
}

llm_with_tools = chat_llm.bind_tools([tavily_search, naver_search])

# 도구 호출 전에 먼저 LLM 응답 확인
messages = prompt.format_messages(user_input=query, today_date=today_date)
llm_response = llm_with_tools.invoke(messages)

# 예쁘게 출력
print("\n" + "="*60)
print("🔍 쿼리:", query)
print("📅 날짜:", today_date)
print("="*60)

# 도구 호출이 있는 경우
if hasattr(llm_response, 'tool_calls') and llm_response.tool_calls:
    print("🔧 도구 호출 감지!")
    for i, tool_call in enumerate(llm_response.tool_calls, 1):
        print(f"\n  {i}. 도구: {tool_call['name']}")
        print(f"     검색어: {tool_call['args'].get('query', tool_call['args'])}")
    
    # 도구 실행
    print("\n🔍 웹 검색 실행 중...")
    tool_results = []
    
    for tool_call in llm_response.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']  # 도구 호출 인자 (딕셔너리)
        
        # 도구 객체의 invoke() 메서드 사용
        if tool_name in tools_dict:
            tool_obj = tools_dict[tool_name]
            # 도구 객체에 인자 전달 (딕셔너리 형태)
            search_result = tool_obj.invoke(tool_args)
            tool_results.append(search_result)
            print(f"  ✓ {tool_name} 검색 완료")
        else:
            print(f"  ✗ 알 수 없는 도구: {tool_name}")
            tool_results.append(f"도구 {tool_name}를 실행할 수 없습니다.")
    
    # 도구 결과를 포함하여 다시 LLM 호출
    # 도구 결과를 메시지에 추가
    tool_messages = []
    for tool_call, tool_result in zip(llm_response.tool_calls, tool_results):
        tool_messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call['id']
            )
        )
    
    # LLM에 원본 메시지 + 도구 결과 전달
    final_messages = messages + [llm_response] + tool_messages
    final_response = chat_llm.invoke(final_messages)
    
    print("\n💬 최종 응답:")
    print(final_response.content)
else:
    # 도구 호출 없이 바로 응답
    print("💬 응답:")
    print(llm_response.content if llm_response.content else "(응답 없음)")

print("="*60)
