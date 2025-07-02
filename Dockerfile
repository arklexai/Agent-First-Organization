FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "create.py", "--config", "./examples/test/config.json", "--log-level", "INFO"]
