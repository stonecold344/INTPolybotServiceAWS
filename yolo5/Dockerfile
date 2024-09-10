FROM python:3.11-slim

# Set the working directory
WORKDIR /usr/src/app

# Install dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Clone YOLOv5 repository
RUN git clone https://github.com/ultralytics/yolov5.git
WORKDIR /usr/src/app/yolov5
RUN pip install -U -r requirements.txt

# Download YOLOv5 weights
RUN curl -L https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -o yolov5s.pt


ENV AWS_REGION='eu-west-3'
ENV DYNAMODB_TABLE='AWS-Project-Predictions-bennyi'
ENV TELEGRAM_TOKEN='6629220970:AAFUHIrGYAZh8RSvIAAl8HmE3q52JxwKm34'
# Copy your application code
COPY . .

# Run your application
CMD ["python3", "app.py"]
