FROM python:3.9-slim

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
ENV AWS_REGION='eu-west-3'
COPY . .

CMD ["python", "app.py"]
