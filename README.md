# 🧠 NLP Web App

Веб-приложение для анализа и обработки текста с использованием моделей Hugging Face.

## 🔗 Демо

👉 https://ratmirnlpwebapp.streamlit.app/

## 🚀 Возможности

* 😊 Анализ тональности текста (RU / EN)
* ✍️ Генерация текста
* 📄 Суммаризация
* 🔁 Перефразирование
* 📝 Исправление текста

## 🛠 Используемые технологии

* Python
* Streamlit
* Hugging Face Inference API
* Langdetect

## ⚙️ Установка и запуск

1. Клонировать репозиторий:

```bash
git clone https://github.com/RatmirBat/nlp-app.git
cd nlp-app
```

2. Установить зависимости:

```bash
pip install -r requirements.txt
```

3. Создать файл `.env`:

```env
HF_TOKEN=your_huggingface_token
```

4. Запустить приложение:

```bash
streamlit run app.py
```

## 🔑 Токен Hugging Face

Получить токен можно здесь:
https://huggingface.co/settings/tokens

## ☁️ Деплой

Приложение развернуто через Streamlit Community Cloud.

Для деплоя используется:

* GitHub репозиторий
* Secrets (для хранения HF_TOKEN)

## 📌 Примечание

Файл `.env` не включён в репозиторий по соображениям безопасности.