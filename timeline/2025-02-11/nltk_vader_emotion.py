import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

##----------------------------------------------------
## NLTK VADER 감정 분석기를 사용한 감정 분석 - 한국어는 compound score가 잘 안나옴
##----------------------------------------------------


import nltk
from deep_translator import GoogleTranslator
from nltk.sentiment import SentimentIntensityAnalyzer
import openai

# 1️⃣ NLTK VADER 감정 분석기 다운로드 (최초 1회 실행 필요)
nltk.download('vader_lexicon')

# 2️⃣ 감정 분석기 초기화
sia = SentimentIntensityAnalyzer()

# 4️⃣ 감정 분석 함수 (NLTK + OpenAI GPT 사용)
def analyze_sentiment(text: str) -> str:
    """한국어 텍스트를 영어로 번역한 후 감정 분석을 수행하는 함수"""
    try:
        # 1️⃣ 한국어 → 영어 번역
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)

        # 2️⃣ NLTK VADER 감정 분석 수행
        sentiment_scores = sia.polarity_scores(translated_text)
        compound_score = sentiment_scores["compound"]

        # 3️⃣ 감정 분석 결과 결정
        if compound_score > 0.05:
            sentiment_label = "긍정적 😀"
        elif compound_score < -0.05:
            sentiment_label = "부정적 😞"
        else:
            sentiment_label = "중립적 😐"

        return f"📊 감정 분석 결과: {sentiment_label} (점수: {compound_score})"
    
    except Exception as e:
        return f"❌ 감정 분석 오류: {str(e)}"

# 5️⃣ 테스트 실행
if __name__ == "__main__":
    sample_texts = [
        "오늘 하루 정말 최악이었어. 너무 힘들다.",
        "so sad and lonely",
        "이 영화는 내 인생 최고의 영화야! 너무 감동적이야.",
        "its interesting",
        "그냥 평범한 하루였어."
    ]
    
    for text in sample_texts:
        print(f"📝 입력 텍스트: {text}")
        print(analyze_sentiment(text))
        print("-" * 50)
