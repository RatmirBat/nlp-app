import os
import re
import streamlit as st
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory
from huggingface_hub import InferenceClient

# =====================================================
# CONFIG
# =====================================================

DetectorFactory.seed = 0
load_dotenv()

st.set_page_config(
    page_title="NLP Web App",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 NLP Web App")
st.caption("Анализ тональности • Генерация • Суммаризация • Перефразирование • Исправление")

HF_TOKEN = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")

if not HF_TOKEN:
    st.error("❌ Не найден HF_TOKEN в .env файле")
    st.stop()


@st.cache_resource
def get_client():
    return InferenceClient(token=HF_TOKEN)


client = get_client()

# =====================================================
# MODELS
# =====================================================

CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

SENTIMENT_MODELS = {
    "ru": "blanchefort/rubert-base-cased-sentiment",
    "en": "cardiffnlp/twitter-roberta-base-sentiment-latest",
}

# =====================================================
# HELPERS
# =====================================================

def detect_lang(text: str) -> str:
    text = text.strip()

    if re.search(r"[а-яА-ЯёЁ]", text):
        return "ru"

    try:
        lang = detect(text)
        return "ru" if lang.startswith("ru") else "en"
    except Exception:
        return "en"


def clean_text(text: str) -> str:
    text = text.strip()

    prefixes = [
        "Answer:",
        "Summary:",
        "In Russian:",
        "На русском:",
        "Ответ:",
        "Резюме:",
        "Краткое содержание:",
        "Исправленный текст:",
        "Перефразированный текст:",
    ]

    for prefix in prefixes:
        if text.startswith(prefix):
            text = text.replace(prefix, "", 1).strip()

    text = re.sub(r"[\u4e00-\u9fff\u3040-\u30ff\u3400-\u4dbf]+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def normalize_sentiment(label: str) -> str:
    label = label.upper()

    mapping = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive",
        "NEGATIVE": "negative",
        "NEUTRAL": "neutral",
        "POSITIVE": "positive",
    }

    return mapping.get(label, "neutral")


def pretty_sentiment(label: str, lang: str) -> str:
    if lang == "ru":
        return {
            "positive": "Позитив 😊",
            "neutral": "Нейтрально 😐",
            "negative": "Негатив 😡",
        }.get(label, label)

    return {
        "positive": "Positive 😊",
        "neutral": "Neutral 😐",
        "negative": "Negative 😡",
    }.get(label, label)

# =====================================================
# NLP FUNCTIONS
# =====================================================

def analyze_sentiment(text: str):
    lang = detect_lang(text)
    model = SENTIMENT_MODELS["ru"] if lang == "ru" else SENTIMENT_MODELS["en"]

    try:
        result = client.text_classification(text, model=model)[0]
        label = normalize_sentiment(result["label"])
        score = round(float(result["score"]), 3)
        return label, score, lang

    except Exception:
        return "neutral", 0.0, lang


def generate_text(prompt: str):
    lang = detect_lang(prompt)

    if lang == "ru":
        system_prompt = """
Ты русскоязычный помощник.
Продолжи текст грамотно, связно и естественно.
Пиши только на русском языке.
Не используй иностранные символы.
"""
        user_prompt = f"Продолжи текст:\n\n{prompt}"
    else:
        system_prompt = """
You are an English writing assistant.
Continue the text clearly and naturally.
Use only English.
"""
        user_prompt = f"Continue the text:\n\n{prompt}"

    try:
        response = client.chat_completion(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=220,
            temperature=0.5,
            top_p=0.9,
        )

        return clean_text(response.choices[0].message.content)

    except Exception as e:
        return f"Ошибка генерации текста: {str(e)}"


def summarize_text(text: str):
    lang = detect_lang(text)
    text = text[:9000]

    if lang == "ru":
        prompt = f"""
Сделай краткое содержание текста.

Правила:
- Только русский язык
- 4-6 предложений
- Только ключевые идеи
- Без лишних слов

Текст:
{text}
"""
    else:
        prompt = f"""
Summarize the text.

Rules:
- 4-6 sentences
- Only key ideas
- Clear and concise

Text:
{text}
"""

    try:
        response = client.chat_completion(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=280,
            temperature=0.2,
            top_p=0.85,
        )

        return clean_text(response.choices[0].message.content)

    except Exception as e:
        return f"Ошибка суммаризации: {str(e)}"


def paraphrase_text(text: str):
    lang = detect_lang(text)
    text = text[:6000]

    if lang == "ru":
        prompt = f"""
Перефразируй текст на русском языке.

Правила:
- Сохрани исходный смысл.
- Измени формулировки.
- Сделай текст более грамотным и естественным.
- Не добавляй новые факты.
- Не используй иностранные символы.

Текст:
{text}
"""
    else:
        prompt = f"""
Paraphrase the text in English.

Rules:
- Preserve the original meaning.
- Change the wording.
- Make the text clear and natural.
- Do not add new facts.

Text:
{text}
"""

    try:
        response = client.chat_completion(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=450,
            temperature=0.35,
            top_p=0.85,
        )

        return clean_text(response.choices[0].message.content)

    except Exception as e:
        return f"Ошибка перефразирования: {str(e)}"


def correct_text(text: str):
    lang = detect_lang(text)
    text = text[:6000]

    if lang == "ru":
        prompt = f"""
Исправь ошибки в тексте на русском языке.

Правила:
- Исправь орфографию, пунктуацию и грамматику.
- Сохрани смысл текста.
- Не добавляй новые факты.
- Не сокращай текст.
- Верни только исправленный текст.

Текст:
{text}
"""
    else:
        prompt = f"""
Correct the text in English.

Rules:
- Fix spelling, punctuation and grammar.
- Preserve the original meaning.
- Do not add new facts.
- Do not shorten the text.
- Return only the corrected text.

Text:
{text}
"""

    try:
        response = client.chat_completion(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=450,
            temperature=0.1,
            top_p=0.8,
        )

        return clean_text(response.choices[0].message.content)

    except Exception as e:
        return f"Ошибка исправления текста: {str(e)}"

# =====================================================
# SIDEBAR
# =====================================================

task = st.sidebar.selectbox(
    "Выберите функцию",
    [
        "Анализ тональности",
        "Генерация текста",
        "Суммаризация",
        "Перефразирование",
        "Исправление текста",
    ]
)

# =====================================================
# UI
# =====================================================

if task == "Анализ тональности":

    st.subheader("😊 Анализ тональности")

    text = st.text_area("Введите текст", height=180)

    if st.button("Анализировать"):
        if text.strip():
            with st.spinner("⏳ Анализируем..."):
                label, score, lang = analyze_sentiment(text)

            st.success("Готово!")
            st.write("Язык:", "Русский" if lang == "ru" else "English")
            st.write("Тональность:", pretty_sentiment(label, lang))
            st.write("Уверенность:", score)
        else:
            st.warning("Введите текст.")


elif task == "Генерация текста":

    st.subheader("✍️ Генерация текста")

    prompt = st.text_area("Введите начало текста", height=180)

    if st.button("Сгенерировать"):
        if prompt.strip():
            with st.spinner("⏳ Генерация..."):
                result = generate_text(prompt)

            st.success("Готово!")
            st.write(result)
        else:
            st.warning("Введите текст.")


elif task == "Суммаризация":

    st.subheader("📄 Суммаризация текста")

    text = st.text_area("Вставьте длинный текст", height=260)

    if st.button("Сжать текст"):
        if text.strip():
            with st.spinner("⏳ Суммаризация..."):
                result = summarize_text(text)

            st.success("Готово!")
            st.write(result)
        else:
            st.warning("Введите текст.")


elif task == "Перефразирование":

    st.subheader("🔁 Перефразирование текста")

    text = st.text_area("Введите текст для перефразирования", height=240)

    if st.button("Перефразировать"):
        if text.strip():
            with st.spinner("⏳ Перефразируем..."):
                result = paraphrase_text(text)

            st.success("Готово!")
            st.write(result)
        else:
            st.warning("Введите текст.")


elif task == "Исправление текста":

    st.subheader("📝 Исправление текста")

    text = st.text_area("Введите текст с ошибками", height=240)

    if st.button("Исправить"):
        if text.strip():
            with st.spinner("⏳ Исправляем..."):
                result = correct_text(text)

            st.success("Готово!")
            st.write(result)
        else:
            st.warning("Введите текст.")