FROM python:3.11-slim

# Set the working directory
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git curl awscli libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to utilize Docker layer caching
RUN pip install --upgrade pip

# Clone YOLOv5 repository
RUN git clone https://github.com/ultralytics/yolov5.git /usr/src/app/yolov5

# Install YOLOv5 dependencies
RUN pip install -r /usr/src/app/yolov5/requirements.txt

# Download YOLOv5 weights
RUN curl -L https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -o /usr/src/app/yolov5/yolov5s.pt

# Copy your application code
COPY . /usr/src/app

# Install additional Python packages (your app-specific requirements)
RUN pip install loguru boto3 python-dotenv requests pyyaml flask

# Set the working directory to your application
WORKDIR /usr/src/app

# Run your application
CMD ["python3", "app.py"]
