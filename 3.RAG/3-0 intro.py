import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma # langchain 0.29 이후로 아래 라이브러리로 분리됨
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough


# load env
load_dotenv()

# API KEY
api_key = os.getenv("OPENAI_API_KEY")

# model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Output parser
output_parser = StrOutputParser()

# 데이터 로드(Load Data)
url = 'https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EC%A0%95%EC%B1%85%EA%B3%BC_%EC%A7%80%EC%B9%A8'
loader = WebBaseLoader(url)

docs = loader.load()

# print(len(docs))
# print(len(docs[0].page_content))
# print(docs[0].page_content[5000:6000])


# 텍스트 분할(Text Split)
# chunk_size : 덩어리 크기
# chunk_overlap : 덩어리 앞뒤로 중복될 크기
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# print(len(splits))
# print(splits[10])
#
# # page_content 속성
# print(splits[10].page_content)
#
# # metadata 속성
# print(splits[10].metadata)

# 인덱싱(Indexing)
# 분할된 텍스트를 검색 가능한 형태로 만드는 단계
vectorstore = Chroma().from_documents(documents=splits, embedding=OpenAIEmbeddings())

# 입력 질문과 유사한 문서 조각 찾기
docs = vectorstore.similarity_search("격하 과정에 대해서 설명해주세요.")

# print(len(docs))
# print(docs[0].page_content)

# 검색(Retrieval)
# Prompt(RAG) : context를 기반으로만 답하라고 명시 → hallucination 방지
template = '''Answer the question based only on the following context
{context}

Question : {question}
'''
prompt = ChatPromptTemplate.from_template(template)
# Retriever
retriever = vectorstore.as_retriever()

# Combine documents : 검색된 문서 조각들을 하나의 문자열로 병합하는 함수
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

# RAG Chain 연결
# 사용자 질문 → Retriever로 context 검색 → Prompt 구성 → LLM → 결과 파싱
rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}  # context와 question을 prompt에 주입
    | prompt           # 프롬프트 생성
    | llm              # LLM 호출
    | output_parser    # 출력 문자열 추출
)

# 생성(Generation)
result = rag_chain.invoke("격하 과정에 대해서 설명해주세요.")
print(result)

# 실제로는 사용자 질문을 분석하여 의도를 파악하는 과정이 먼저 필요함