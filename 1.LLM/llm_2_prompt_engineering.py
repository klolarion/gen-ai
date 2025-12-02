import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================================
# 프롬프트 작성 원칙
# ============================================================================
# 1. 명확성과 구체성: 질문은 명확하고 구체적이어야 함
# 2. 배경 정보 포함: 문맥을 제공하여 환각 현상 방지
# 3. 간결함: 핵심 정보에 집중
# 4. 열린 질문: 자세한 답변을 유도
# 5. 명확한 목표: 얻고자 하는 결과 유형을 정의
# 6. 적절한 언어와 문체: 상황에 맞는 표현 선택
# ============================================================================

# ============================================================================
# 1. 기본 프롬프트 vs 개선된 프롬프트
# ============================================================================

print("=" * 60)
print("1. 프롬프트 품질 비교")
print("=" * 60)

# 나쁜 예
bad_prompt = "주식에 대해 알려줘"
response_bad = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": bad_prompt}]
)

print(f"[나쁜 프롬프트]: {bad_prompt}")
print(f"답변: {response_bad.choices[0].message.content}...")

# 좋은 예
good_prompt = "2025년 11월 현재, 삼성전자 주식에 영향을 줄 수 있는 반도체 시장의 주요 동향 3가지를 알려줘"
response_good = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": good_prompt}]
)

print(f"\n[좋은 프롬프트]: {good_prompt}")
print(f"답변: {response_good.choices[0].message.content}...")

# ============================================================================
# 2. 프롬프트 구성 요소
# ============================================================================

print("\n" + "=" * 60)
print("2. 프롬프트 구성 요소")
print("=" * 60)

# 지시(Instruction) + 맥락(Context) + 질문(Question)
structured_prompt = """
[지시]
아래 제공된 제품 리뷰를 요약해주세요.

[맥락]
리뷰는 스마트워치에 대한 것이며, 사용자 경험에 초점을 맞추고 있습니다.

[리뷰]
"이 스마트워치는 정말 사용하기 편리합니다. 배터리 수명이 3일 이상 지속되어
매일 충전할 필요가 없습니다. 운동 추적 기능도 정확하고, 수면 분석 기능이
특히 유용합니다. 다만 가격이 다소 비싼 편입니다."

[질문]
이 리뷰를 바탕으로 스마트워치의 주요 장단점을 2-3문장으로 요약해주세요.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": structured_prompt}]
)

print(f"답변:\n{response.choices[0].message.content}")

# ============================================================================
# 3. 역할 지정 (Role Prompting)
# ============================================================================

print("\n" + "=" * 60)
print("3. 역할 지정")
print("=" * 60)

question = "파이썬과 자바스크립트 중 어떤 것을 먼저 배워야 할까요?"

# 전문가 역할 지정
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "당신은 20년 경력의 시니어 소프트웨어 엔지니어이자 프로그래밍 교육 전문가입니다."},
        {"role": "user", "content": question}
    ]
)

print(f"질문: {question}")
print(f"답변:\n{response.choices[0].message.content}")

# ============================================================================
# 4. Few-Shot 프롬프팅
# ============================================================================

print("\n" + "=" * 60)
print("4. Few-Shot 프롬프팅 (예시 제공)")
print("=" * 60)

few_shot_prompt = """
다음 문장을 긍정/부정으로 분류해주세요.

예시:
문장: "이 영화 정말 재미있었어요!"
분류: 긍정

문장: "서비스가 형편없었습니다."
분류: 부정

문장: "배송이 빠르고 제품 품질도 좋아요."
분류: 긍정

이제 다음 문장을 분류해주세요:
문장: "가격 대비 성능이 아쉽네요."
분류:
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": few_shot_prompt}],
    temperature=0.1  # 일관성을 위해 낮은 온도
)

print(f"답변: {response.choices[0].message.content}")

# ============================================================================
# 5. 단계별 사고 유도 (Chain of Thought)
# ============================================================================

print("\n" + "=" * 60)
print("5. 단계별 사고 유도")
print("=" * 60)

cot_prompt = """
다음 문제를 단계별로 풀어주세요:

문제: 한 상점에서 사과 12개를 3,600원에 판매합니다. 
만약 20% 할인을 받는다면, 사과 1개의 가격은 얼마인가요?

단계별로 계산 과정을 보여주세요.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": cot_prompt}]
)

print(f"답변:\n{response.choices[0].message.content}")

# ============================================================================
# 6. 출력 형식 지정
# ============================================================================

print("\n" + "=" * 60)
print("6. 출력 형식 지정 (JSON)")
print("=" * 60)

format_prompt = """
다음 정보를 JSON 형식으로 추출해주세요:

"홍길동은 30세이며, 서울에 거주하고 있습니다. 
직업은 소프트웨어 엔지니어이고, Python과 JavaScript를 주로 사용합니다."

JSON 형식:
{
  "name": "이름",
  "age": 나이,
  "location": "거주지",
  "job": "직업",
  "skills": ["기술1", "기술2"]
}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": format_prompt}],
    response_format={"type": "json_object"}  # JSON 형식 강제
)

print(f"답변:\n{response.choices[0].message.content}")

print("\n" + "=" * 60)

