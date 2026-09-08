FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install -r --no-cache-dir requirements.txt

COPY src/ ./src/
COPY static/ ./static/
COPY results/Model.pth ./results/Model.pth

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]