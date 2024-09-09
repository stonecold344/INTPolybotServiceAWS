# Use the Python 3.9 slim image
FROM python:3.9-slim

# Set the working directory
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV AWS_REGION='eu-west-3'
ENV DYNAMODB_TABLE='AWS-Project-Predictions-bennyi'

# Copy the application code
COPY . .

# Command to run the application
CMD ["python", "app.py"]
