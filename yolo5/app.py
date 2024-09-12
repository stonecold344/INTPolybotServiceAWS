import uuid
from flask import Flask, request, jsonify
import boto3
import requests
import json
import os
import logging
from loguru import logger
import time

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE')
AWS_REGION = os.getenv('AWS_REGION')
SQS_URL = os.getenv('SQS_URL')
POLYBOT_URL = os.getenv('POLYBOT_URL')

if not all([S3_BUCKET_NAME, DYNAMODB_TABLE, AWS_REGION, SQS_URL, POLYBOT_URL]):
    logger.error(f"Missing environment variables: S3_BUCKET_NAME={S3_BUCKET_NAME}, DYNAMODB_TABLE={DYNAMODB_TABLE}, AWS_REGION={AWS_REGION}, SQS_URL={SQS_URL}, POLYBOT_URL={POLYBOT_URL}")
    raise ValueError("One or more environment variables are missing")

# Initialize AWS clients
s3_client = boto3.client('s3', region_name=AWS_REGION)
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
table = dynamodb.Table(DYNAMODB_TABLE)
sqs_client = boto3.client('sqs', region_name=AWS_REGION)

@app.route('/process', methods=['POST'])
def process_image():
    try:
        req = request.get_json()
        if not req:
            logger.warning("Received empty request payload")
            return jsonify({'error': 'Empty request payload'}), 400

        image_url = req.get('image_url')
        chat_id = req.get('chat_id')

        if not image_url or not chat_id:
            logger.error("Missing image_url or chat_id")
            return jsonify({'error': 'image_url and chat_id are required'}), 400

        # Download image from S3
        object_key = image_url.split('/')[-1]
        local_file_path = f"/tmp/{object_key}"
        s3_client.download_file(S3_BUCKET_NAME, object_key, local_file_path)

        # Process image with YOLOv5 (placeholder for YOLOv5 processing code)
        # Here, you would call your YOLOv5 model to detect objects
        # For this example, we will mock the detection results
        labels = [{"class": "object_name", "count": 1}]  # Example label

        # Save results to DynamoDB
        prediction_id = str(uuid.uuid4())
        table.put_item(
            Item={
                'prediction_id': prediction_id,
                'chat_id': chat_id,
                'labels': labels
            }
        )

        # Notify Polybot about the results
        response = requests.post(f"{POLYBOT_URL}/results?predictionId={prediction_id}")

        if response.status_code != 200:
            logger.error(f"Error notifying Polybot: {response.text}")
            return jsonify({'error': 'Failed to notify Polybot'}), 500

        return jsonify({'prediction_id': prediction_id}), 200

    except Exception as e:
        logger.error(f"Error in /process endpoint: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
