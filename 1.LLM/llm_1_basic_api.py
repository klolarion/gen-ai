import os
from openai import OpenAI
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# ============================================================================
# OpenAI 클라이언트 초기화
# ============================================================================

# 방법 1: OpenAI API 사용 (기본)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 방법 2: Ollama 로컬 모델 사용 (주석 해제하여 사용)
# Ollama를 사용하려면:
# 1. Ollama 설치: https://ollama.ai
# 2. 서버 실행: ollama serve
# 3. 모델 다운로드: ollama pull qwen2.5:20b or gpt-oss:20b
#
# client = OpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key="ollama"  # 더미 값 (Ollama는 키 불필요하지만 클라이언트 요구사항)
# )

# ============================================================================
# 1. 기본 텍스트 생성 (Completion)
# ============================================================================

print("=" * 60)
print("1. 기본 텍스트 생성")
print("=" * 60)

response = client.chat.completions.create(
    model="gpt-4o-mini", 
    messages=[
        {"role": "user", "content": "지구의 자전 주기는?"}
    ]
)

print(f"질문: 지구의 자전 주기는?")
print(f"답변: {response.choices[0].message.content}")
print(f"사용된 토큰: {response.usage.total_tokens}")

# ============================================================================
# 2. 시스템 메시지와 함께 사용
# ============================================================================

print("\n" + "=" * 60)
print("2. 시스템 메시지로 역할 지정")
print("=" * 60)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "당신은 전문 천문학자입니다. 간결하고 정확하게 답변하세요."},
        {"role": "user", "content": "태양계에서 가장 큰 행성은?"}
    ]
)

print(f"질문: 태양계에서 가장 큰 행성은?")
print(f"답변: {response.choices[0].message.content}")

# ============================================================================
# 3. 대화 히스토리 유지
# ============================================================================

print("\n" + "=" * 60)
print("3. 대화 히스토리")
print("=" * 60)

messages = [
    {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다."},
    {"role": "user", "content": "내 이름은 홍길동이야"},
]

# 첫 번째 대화
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print(f"사용자: 내 이름은 홍길동이야")
print(f"AI: {response.choices[0].message.content}")

# 대화 히스토리에 추가
messages.append({"role": "assistant", "content": response.choices[0].message.content})
messages.append({"role": "user", "content": "내 이름이 뭐였지?"})

# 두 번째 대화
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print(f"사용자: 내 이름이 뭐였지?")
print(f"AI: {response.choices[0].message.content}")

# ============================================================================
# 4. 온도(Temperature) 파라미터
# ============================================================================

print("\n" + "=" * 60)
print("4. 온도 파라미터 (창의성 조절)")
print("=" * 60)

prompt = "AI의 미래에 대해 한 문장으로 말해"

# 낮은 온도 (결정론적, 일관성 있음) - 기술적인 답변
response_low = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1
)

# 높은 온도 (창의적, 다양함) - 창의적인 답변
response_high = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=1.5
)

print(f"질문: {prompt}")
print(f"\n[낮은 온도 (0.1) - 일관성]: {response_low.choices[0].message.content}")
print(f"\n[높은 온도 (1.5) - 창의성]: {response_high.choices[0].message.content}")

# ============================================================================
# 5. 스트리밍 응답
# ============================================================================

print("\n" + "=" * 60)
print("5. 스트리밍 응답 (실시간 출력)")
print("=" * 60)

print("질문: 파이썬이란?")
print("답변: ", end="", flush=True)

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "파이썬이란 무엇인지 한 문장으로 설명해"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)

print("\n" + "=" * 60)

