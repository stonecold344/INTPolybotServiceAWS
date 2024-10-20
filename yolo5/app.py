import logging
import time
from pathlib import Path
import yaml
from loguru import logger
import os
import boto3
import requests
import json
from dotenv import load_dotenv
import sys
from urllib.parse import urlparse
from decimal import Decimal

sys.path.append('/usr/src/app/yolov5')
from detect import run

# Load environment variables
load_dotenv(dotenv_path='/usr/src/app/.env')
logger.info("Environment file loaded")
# Initialize S3, SQS, and DynamoDB clients
SQS_URL = os.getenv('SQS_URL')
AWS_REGION = os.getenv('AWS_REGION')
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
boto_session = boto3.session.Session(region_name=AWS_REGION)

def get_secret(secret_id):
    client = boto_session.client(service_name='secretsmanager')
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_id)
        secret = get_secret_value_response['SecretString']
        return json.loads(secret)
    except Exception as e:
        logging.error(f"Error retrieving secret: {e}")
        raise e

# Retrieve the Telegram token
SECRET_ID = "Telegram-Secret-Bennyi24"
secrets = get_secret(SECRET_ID)
TELEGRAM_TOKEN = secrets.get("Telegram-Secret-Bennyi")



logger.info(f'Telegram Bot information:\n{TELEGRAM_TOKEN}')
logger.info(f"S3 Bucket Name: {S3_BUCKET_NAME}")
logger.info(f"AWS Region: {AWS_REGION}")
logger.info(f"SQS URL: {SQS_URL}")
logger.info(f"DynamoDB Table: {DYNAMODB_TABLE}")

# Ensure all environment variables are loaded
if not all([SQS_URL, AWS_REGION, TELEGRAM_TOKEN, DYNAMODB_TABLE, S3_BUCKET_NAME]):
    logger.error("One or more environment variables are missing")
    raise ValueError("One or more environment variables are missing")

sqs_client = boto3.client('sqs', region_name=AWS_REGION)
queue_name = 'aws-sqs-image-processing-bennyi'
response = sqs_client.get_queue_url(QueueName=queue_name)
SQS_QUEUE_NAME = response['QueueUrl']
logger.info(f"SQS_QUEUE_URL: {SQS_QUEUE_NAME}")

s3_client = boto3.client('s3', region_name=AWS_REGION)
dynamodb_client = boto3.resource('dynamodb', region_name=AWS_REGION)
table = dynamodb_client.Table(DYNAMODB_TABLE)

# Load labels from YOLO configuration
with open("/usr/src/app/yolov5/data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

def get_img_name_from_url(image_url):
    """Extracts the image name from the URL."""
    path = urlparse(image_url).path
    return path.split('/')[-1]

def s3_object_exists(bucket, key):
    """Checks if an object exists in S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            logger.error(f"Error checking if object exists in S3: {e}")
            raise

def download_image_from_s3(img_name):
    """Downloads an image from the S3 bucket."""
    local_img_path = f"images/{img_name}"
    os.makedirs(os.path.dirname(local_img_path), exist_ok=True)
    s3_key = f'docker-project/{img_name}'

    if not s3_object_exists(S3_BUCKET_NAME, s3_key):
        logger.error(f"Image {img_name} not found in S3 bucket {S3_BUCKET_NAME} under key {s3_key}")
        raise FileNotFoundError(f"Image {img_name} not found in S3 bucket {S3_BUCKET_NAME} under key {s3_key}")

    try:
        logger.info(f"Downloading {img_name} from S3 bucket {S3_BUCKET_NAME} with key {s3_key}")
        s3_client.download_file(S3_BUCKET_NAME, s3_key, local_img_path)
        return local_img_path
    except Exception as e:
        logger.error(f"Error downloading image from S3: {e}")
        raise

def upload_image_to_s3(img_path, img_name):
    """Uploads an image to the S3 bucket."""
    s3_key = f'docker-project/{img_name}'
    try:
        s3_client.upload_file(img_path, S3_BUCKET_NAME, s3_key)
        logger.info(f"Uploaded {img_name} to S3 bucket {S3_BUCKET_NAME} under key {s3_key}")
    except Exception as e:
        logger.error(f"Error uploading image to S3: {e}")
        raise

def store_prediction_in_dynamodb(prediction_summary):
    """Stores the prediction summary in DynamoDB."""
    try:
        # Convert all float values to Decimal
        def convert_floats_to_decimal(data):
            for key, value in data.items():
                if isinstance(value, float):
                    data[key] = Decimal(str(value))
                elif isinstance(value, dict):
                    convert_floats_to_decimal(value)  # Recursively handle nested dictionaries
            return data

        prediction_summary = convert_floats_to_decimal(prediction_summary)
        table.put_item(Item=prediction_summary)
        logger.info(f"Stored prediction {prediction_summary['prediction_id']} in DynamoDB")
    except Exception as e:
        logger.error(f"Error storing prediction in DynamoDB: {e}")
        raise

def format_prediction_summary(labels):
    """Formats prediction results to show object counts."""
    # Initialize a dictionary to keep counts
    object_counts = {}

    # Count occurrences of each object
    for label in labels:
        object_name = label['class']
        if object_name in object_counts:
            object_counts[object_name] += 1
        else:
            object_counts[object_name] = 1

    # Construct the result message
    result_lines = [f"{object_name}:{count}" for object_name, count in object_counts.items()]
    return "\n".join(result_lines)

def notify_telegram(chat_id, message):
    """Sends a message directly to a Telegram chat."""
    telegram_api_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': chat_id, 'text': message}

    try:
        responses = requests.post(telegram_api_url, data=payload)
        responses.raise_for_status()
        logger.info(f"Sent message to Telegram chat {chat_id}: {message}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending message to Telegram: {e}")
        raise

def consume():
    while True:
        try:
            logger.info("Attempting to receive messages from SQS...")

            responses = sqs_client.receive_message(QueueUrl=SQS_QUEUE_NAME, MaxNumberOfMessages=1, WaitTimeSeconds=20)

            if 'Messages' in responses:
                message = json.loads(responses['Messages'][0]['Body'])
                receipt_handle = responses['Messages'][0]['ReceiptHandle']

                # Log the message to inspect its contents
                logger.info(f"Received SQS message: {message}")

                # Extract values safely
                prediction_id = responses['Messages'][0]['MessageId']
                image_url = message.get('image_url')
                chat_id = message.get('chat_id')

                if not image_url or not chat_id:
                    logger.error(f"Missing 'image_url' or 'chat_id' in message: {message}")
                    continue

                img_name = get_img_name_from_url(image_url)
                logger.info(f'Prediction {prediction_id} started for image {img_name}')

                try:
                    original_img_path = download_image_from_s3(img_name)
                    logger.info(f'Image {img_name} downloaded from S3 to {original_img_path}')

                    run(
                        weights='yolov5s.pt',
                        data='/usr/src/app/yolov5/data/coco128.yaml',
                        source=original_img_path,
                        project='static/data',
                        name=prediction_id,
                        save_txt=True,
                        exist_ok=True
                    )
                    logger.info(f'YOLOv5 completed processing for {original_img_path}')
                except Exception as e:
                    logger.error(f'Error during YOLOv5 inference: {e}')
                    continue

                logger.info(f'Prediction {prediction_id} completed')

                predicted_img_path = Path(f'static/data/{prediction_id}/{img_name}')
                pred_summary_path = Path(f'static/data/{prediction_id}/labels/{img_name.split(".")[0]}.txt')

                try:
                    upload_image_to_s3(predicted_img_path, f"predictions/{prediction_id}/{img_name}")

                    if pred_summary_path.exists():
                        with open(pred_summary_path) as f:
                            labels = f.read().splitlines()
                            labels = [line.split(' ') for line in labels]
                            labels = [{
                                'class': names[int(l[0])],
                                'cx': Decimal(l[1]),
                                'cy': Decimal(l[2]),
                                'width': Decimal(l[3]),
                                'height': Decimal(l[4]),
                            } for l in labels]

                        logger.info(f'Prediction summary for {prediction_id}: {labels}')

                        prediction_summary = {
                            'prediction_id': prediction_id,
                            'original_img_path': original_img_path,
                            'predicted_img_path': str(predicted_img_path),
                            'chat_id': chat_id,
                            'object_counts': format_prediction_summary(labels)
                        }

                        store_prediction_in_dynamodb(prediction_summary)
                        notify_telegram(chat_id, prediction_summary['object_counts'])
                    else:
                        logger.error(f"Prediction summary file not found for {img_name}.")

                except Exception as e:
                    logger.error(f"Error processing prediction for {prediction_id}: {e}")
                    continue

                finally:
                    # Delete the message from the queue
                    logger.info("Deleting message from SQS...")
                    sqs_client.delete_message(QueueUrl=SQS_QUEUE_NAME, ReceiptHandle=receipt_handle)
                    logger.info("Message deleted from SQS.")
            else:
                logger.info("No messages received. Retrying...")

        except Exception as e:
            logger.error(f"Error while consuming messages: {e}")
            time.sleep(1)  # Wait a moment before retrying

if __name__ == "__main__":
    logger.info("Starting the Yolo5 service...")
    consume()
