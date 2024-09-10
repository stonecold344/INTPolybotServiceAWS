FROM python:3.11-slim

# Set the working directory
WORKDIR /usr/src/app

# Install git and curl
RUN apt-get update && \
    apt-get install -y git curl && \
    pip install --upgrade pip

# Clone YOLOv5 repository
RUN git clone https://github.com/ultralytics/yolov5.git
WORKDIR /usr/src/app/yolov5

# Install YOLOv5 dependencies
RUN pip install -r requirements.txt

# Download YOLOv5 weights
RUN curl -L https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -o yolov5s.pt

# Copy your application code
COPY . .

# Run your application
CMD ["python3", "app.py"]
