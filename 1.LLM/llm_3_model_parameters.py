import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# ============================================================================
# 토큰(Token)이란?
# ============================================================================
# 토큰은 LLM이 텍스트를 처리하는 기본 단위입니다. 단어와 비슷하지만 정확히는 다릅니다.
# 
# - 영어: 1 토큰 ≈ 0.75 단어 (예: "unhappiness" = 3 토큰 ["un", "happy", "ness"])
# - 한국어: 1 토큰 ≈ 0.5~1 단어 (형태소 단위로 분리되는 경우가 많음)
# - 예시: "안녕하세요" = 약 2-3 토큰, "Hello world" = 약 2 토큰
# 
# 토큰 수는 API 비용과 응답 길이를 결정하는 중요한 요소입니다.
# ============================================================================
# 주요 파라미터 설명
# ============================================================================
# temperature: 0.0-1.0, 낮을수록 결정론적, 높을수록 창의적
# max_tokens: 생성할 최대 토큰 수
# top_p: 0.0-1.0, 누적 확률 상위 토큰만 샘플링 (낮을수록 보수적, 높을수록 다양함)
# frequency_penalty: -2.0 ~ 2.0, 이미 사용된 토큰의 빈도에 따라 페널티 부여 (반복 억제)
# presence_penalty: -2.0 ~ 2.0, 이미 사용된 토큰의 존재 자체에 페널티 부여 (새로운 주제/단어 장려)
# n: 생성할 응답 개수 (여러 후보 생성 시 사용)
# ============================================================================

test_prompt = "AI에 대해 한 문장으로 설명해"

# ============================================================================
# 1. Temperature (온도)
# ============================================================================

print("=" * 60)
print("1. Temperature - 창의성 조절")
print("=" * 60)

temperatures = [0.0, 0.5, 1.0]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": test_prompt}],
        temperature=temp
    )
    print(f"\nTemperature={temp}:")
    print(f"{response.choices[0].message.content}")

# ============================================================================
# 2. Max Tokens (최대 토큰)
# ============================================================================

print("\n" + "=" * 60)
print("2. Max Tokens - 응답 길이 제한")
print("=" * 60)

prompt = "파이썬의 장점을 설명해"

# 짧은 응답
response_short = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=50
)

# 긴 응답
response_long = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=200
)

print(f"[Max Tokens=50]:\n{response_short.choices[0].message.content}")
print(f"\n[Max Tokens=200]:\n{response_long.choices[0].message.content}")

# ============================================================================
# 3. Top P (Nucleus Sampling)
# ============================================================================

print("\n" + "=" * 60)
print("3. Top P - 확률 기반 샘플링")
print("=" * 60)

# Temperature와 Top P는 함께 사용하지 않는 것이 권장됨
# 둘 중 하나만 조정하는 것이 좋음

response_low_p = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": test_prompt}],
    top_p=0.1,  # 상위 10% 토큰만 고려 (보수적)
    temperature=1  # temperature를 1로 고정
)

response_high_p = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": test_prompt}],
    top_p=0.9,  # 상위 90% 토큰 고려 (다양함)
    temperature=1
)

print(f"[Top P=0.1 (보수적)]:\n{response_low_p.choices[0].message.content}")
print(f"\n[Top P=0.9 (다양함)]:\n{response_high_p.choices[0].message.content}")

# ============================================================================
# 4. Frequency Penalty (빈도 페널티)
# ============================================================================

print("\n" + "=" * 60)
print("4. Frequency Penalty - 반복 억제")
print("=" * 60)

repetitive_prompt = "Python은"

# 페널티 없음
response_no_penalty = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": repetitive_prompt}],
    frequency_penalty=0.0,
    max_tokens=100
)

# 높은 페널티
response_with_penalty = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": repetitive_prompt}],
    frequency_penalty=1.5,
    max_tokens=100
)

print(f"[Frequency Penalty=0.0]:\n{response_no_penalty.choices[0].message.content}")
print(f"\n[Frequency Penalty=1.5]:\n{response_with_penalty.choices[0].message.content}")

# ============================================================================
# 5. Presence Penalty (존재 페널티)
# ============================================================================

print("\n" + "=" * 60)
print("5. Presence Penalty - 새로운 주제 장려")
print("=" * 60)

broad_prompt = "프로그래밍 언어에 대해 설명해줘"

# 페널티 없음 (한 주제에 집중)
response_no_presence = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": broad_prompt}],
    presence_penalty=0.0,
    max_tokens=150
)

# 높은 페널티 (다양한 주제)
response_with_presence = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": broad_prompt}],
    presence_penalty=1.5,
    max_tokens=150
)

print(f"[Presence Penalty=0.0]:\n{response_no_presence.choices[0].message.content}")
print(f"\n[Presence Penalty=1.5]:\n{response_with_presence.choices[0].message.content}")


print("\n" + "=" * 60)

