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

sys.path.append('/usr/src/app/yolov5')
from detect import run

load_dotenv(dotenv_path='/usr/src/app/.env')  # Update path if needed
logger.info("Env file loaded")

# Initialize S3, SQS, and DynamoDB clients
SQS_URL = os.getenv('SQS_URL')
AWS_REGION = os.getenv('AWS_REGION')
polybot_url = os.getenv('POLYBOT_URL')
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# Ensure all environment variables are loaded
if not all([SQS_URL, AWS_REGION, DYNAMODB_TABLE, S3_BUCKET_NAME]):
    logger.error("One or more environment variables are missing")
    raise ValueError("One or more environment variables are missing")

sqs_client = boto3.client('sqs', region_name=AWS_REGION)
s3_client = boto3.client('s3', region_name=AWS_REGION)
dynamodb_client = boto3.resource('dynamodb', region_name=AWS_REGION)
table = dynamodb_client.Table(DYNAMODB_TABLE)
s3_folder_path = 'aws-project'

with open("/usr/src/app/yolov5/data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

def get_img_name_from_url(image_url):
    """Extracts the image name from the URL."""
    path = urlparse(image_url).path
    return path.split('/')[-1]

def download_image_from_s3(img_name):
    """Downloads an image from the S3 bucket."""
    local_img_path = f"images/{img_name}"
    os.makedirs(os.path.dirname(local_img_path), exist_ok=True)
    try:
        s3_client.download_file(S3_BUCKET_NAME, img_name, local_img_path)
        return local_img_path
    except Exception as e:
        logger.error(f"Error downloading image from S3: {e}")
        raise

def upload_image_to_s3(img_path, img_name):
    """Uploads an image to the S3 bucket."""
    try:
        s3_client.upload_file(img_path, S3_BUCKET_NAME, img_name)
        logger.info(f"Uploaded {img_name} to S3 bucket {S3_BUCKET_NAME}")
    except Exception as e:
        logger.error(f"Error uploading image to S3: {e}")
        raise

def store_prediction_in_dynamodb(prediction_summary):
    """Stores the prediction summary in DynamoDB."""
    try:
        table.put_item(Item=prediction_summary)
        logger.info(f"Stored prediction {prediction_summary['prediction_id']} in DynamoDB")
    except Exception as e:
        logger.error(f"Error storing prediction in DynamoDB: {e}")
        raise

def notify_polybot(prediction_id):
    """Notifies Polybot of the prediction completion."""
    url = f"{polybot_url}/results?predictionId={prediction_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        logger.info(f"Notified Polybot of prediction {prediction_id}: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error notifying Polybot: {e}")
        raise

def consume():
    while True:
        try:
            response = sqs_client.receive_message(QueueUrl=SQS_URL, MaxNumberOfMessages=1, WaitTimeSeconds=5)

            if 'Messages' in response:
                message = json.loads(response['Messages'][0]['Body'])
                receipt_handle = response['Messages'][0]['ReceiptHandle']

                # Log the message to inspect its contents
                logger.info(f"Received SQS message: {message}")

                # Extract values safely
                prediction_id = response['Messages'][0]['MessageId']
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
                                'cx': float(l[1]),
                                'cy': float(l[2]),
                                'width': float(l[3]),
                                'height': float(l[4]),
                            } for l in labels]

                        logger.info(f'Prediction summary for {prediction_id}: {labels}')

                        prediction_summary = {
                            'prediction_id': prediction_id,
                            'original_img_path': original_img_path,
                            'predicted_img_path': str(predicted_img_path),
                            'labels': labels,
                            'chat_id': chat_id,
                            'time': time.time()
                        }

                        store_prediction_in_dynamodb(prediction_summary)
                        notify_polybot(prediction_id)
                    else:
                        logger.warning(f'No prediction summary file found for {prediction_id}')

                except Exception as e:
                    logger.error(f"Error processing prediction results: {e}")
                    continue

                sqs_client.delete_message(QueueUrl=SQS_URL, ReceiptHandle=receipt_handle)
            else:
                logger.info("No messages in SQS queue, waiting...")

        except Exception as e:
            logger.error(f"Error in SQS consume loop: {e}")
            time.sleep(10)  # Wait before retrying

if __name__ == "__main__":
    consume()
