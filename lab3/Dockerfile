# Базовый образ python 
FROM python:3.12-slim

EXPOSE 8501

# Устанавливаем рабочую директорию
WORKDIR /app

# Скопируем проект в рабочую директорию
COPY . .

# Устанавливаем зависимости
RUN pip install -r requirements.txt

# Запускаем pipelinedocker 
RUN chmod +x pipeline.sh && ./pipeline.sh

CMD streamlit run streamlit_app.py
