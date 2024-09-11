FROM python:3.11-slim

# Set the working directory
WORKDIR /usr/src/app

# Install git and curl
RUN apt-get update && \
    apt-get install -y git curl && \
    pip install --upgrade pip

# Clone YOLOv5 repository
RUN git clone https://github.com/ultralytics/yolov5.git /usr/src/app

# Install YOLOv5 dependencies
RUN pip install -r /usr/src/app/requirements.txt

# Download YOLOv5 weights
RUN curl -L https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -o /usr/src/app/yolov5s.pt

# Copy your application code
COPY . /usr/src/app

# Install additional Python packages
RUN pip install loguru boto3 python-dotenv requests pyyaml

# Set the working directory to your application
WORKDIR /usr/src/app

# Run your application
CMD ["python3", "app.py"]
