FROM python:3.9

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port $PORT"]