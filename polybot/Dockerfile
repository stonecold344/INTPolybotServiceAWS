# Use the Python 3.9 slim image
FROM python:3.9-slim

# Set the working directory
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    awscli \
    openssh-server \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Gunicorn
RUN pip install gunicorn

# Set environment variables
# These can be managed outside Dockerfile in production
ENV AWS_REGION='eu-west-3'
ENV DYNAMODB_TABLE='AWS-Project-Predictions-bennyi'
ENV SQS_QUEUE_URL='https://sqs.eu-west-3.amazonaws.com/019273956931/aws-sqs-image-processing-bennyi'

# Copy the application code
COPY . .

# Expose the port
EXPOSE 8443

# Command to run the application with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8443", "app:app"]
