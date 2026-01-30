FROM python:3.10

WORKDIR /app

# Установить системные зависимости (libgl1)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

COPY app/ .

# Установить все зависимости
RUN pip install --no-cache-dir -r requirements.txt


# Запуск приложения
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]