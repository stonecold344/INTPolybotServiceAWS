import flask
from flask import request, jsonify
import os
import boto3
from bot import ObjectDetectionBot
import json
import requests
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from the .env file
load_dotenv(dotenv_path='/usr/src/app/.env')
logging.info("Env file loaded")

app = flask.Flask(__name__)

# Environment Variables
TELEGRAM_APP_URL = os.getenv('TELEGRAM_APP_URL')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE')
AWS_REGION = os.getenv('AWS_REGION')
SQS_URL = os.getenv('SQS_URL')

# Initialize boto3 session globally
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

def get_yolo5_url():
    ec2 = boto_session.client('ec2')
    try:
        response = ec2.describe_instances(Filters=[
            {'Name': 'tag:Name', 'Values': ['yolo5-instance-bennyi']},
            {'Name': 'instance-state-name', 'Values': ['running']}
        ])
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                yolo5_instance_ip = instance.get('PublicIpAddress')
                if yolo5_instance_ip:
                    return f'http://{yolo5_instance_ip}:8081'
    except Exception as e:
        logging.error(f"Error fetching YOLO5 instance IP: {e}")
    logging.error("Could not find YOLO5 instance IP")
    return None

# Retrieve the Telegram token
SECRET_ID = "Telegram-Secret-Bennyi24"
secrets = get_secret(SECRET_ID)
TELEGRAM_TOKEN = secrets.get("Telegram-Secret-Bennyi")

# Ensure all environment variables are loaded
if not all([TELEGRAM_TOKEN, TELEGRAM_APP_URL, S3_BUCKET_NAME, DYNAMODB_TABLE, AWS_REGION, SQS_URL]):
    logging.error("One or more environment variables are missing")
    raise ValueError("One or more environment variables are missing")

# Initialize DynamoDB
dynamodb = boto_session.resource('dynamodb')
table = dynamodb.Table(DYNAMODB_TABLE)

# Define bot object globally
YOLO5_URL = get_yolo5_url()
logging.info(f"YOLO5 service URL: {YOLO5_URL}")
bot = ObjectDetectionBot(TELEGRAM_TOKEN, TELEGRAM_APP_URL, S3_BUCKET_NAME, YOLO5_URL, AWS_REGION, SQS_URL, DYNAMODB_TABLE)

def set_webhook():
    try:
        # Get current webhook info
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getWebhookInfo"
        response = requests.get(url)
        webhook_info = response.json()

        # Check if webhook is already set to the correct URL
        current_url = webhook_info['result'].get('url', None)
        desired_url = f"{TELEGRAM_APP_URL}/{TELEGRAM_TOKEN}/"

        if current_url == desired_url:
            logging.info("Webhook is already set to the desired URL: %s", current_url)
            return

        # Set webhook if not already set or has a different URL
        set_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook"
        response = requests.post(set_url, data={"url": desired_url})
        result = response.json()
        if result.get('ok'):
            logging.info("Webhook set successfully")
        else:
            logging.error("Failed to set webhook: %s", result)

    except Exception as e:
        logging.error(f"Error occurred while setting webhook: {e}")

@app.route('/', methods=['GET'])
def index():
    return 'Ok'

@app.route(f'/{TELEGRAM_TOKEN}/', methods=['POST'])
def webhook():
    req = request.get_json()
    logging.info("Received request: %s", req)
    if req is None:
        return jsonify({'error': 'Empty request payload'}), 400
    bot.handle_message(req.get('message', {}))
    return 'Ok'

@app.route('/results', methods=['POST'])
def results():
    prediction_id = request.args.get('predictionId')
    if not prediction_id:
        return jsonify({'error': 'predictionId is required'}), 400
    try:
        response = table.get_item(Key={'prediction_id': prediction_id})
        if 'Item' not in response:
            return jsonify({'error': 'Prediction not found'}), 404
        prediction_summary = response['Item']
        chat_id = prediction_summary['chat_id']
        labels = prediction_summary['labels']
        text_results = '\n'.join([f"{label['class']} : {label['count']}" for label in labels])
        bot.send_text(chat_id, text_results)
        return 'Ok'
    except Exception as e:
        logging.error(f"Error fetching prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req = request.get_json()
        if req is None:
            return jsonify({'error': 'Empty request payload'}), 400

        image_url = req.get('image_url')
        if not image_url:
            return jsonify({'error': 'image_url is required'}), 400

        message_body = json.dumps({
            'image_url': image_url,
            'chat_id': req.get('chat_id')
        })
        response = boto_session.client('sqs').send_message(
            QueueUrl=SQS_URL,
            MessageBody=message_body
        )

        logging.info(f"Message sent to SQS with ID: {response.get('MessageId')}")
        return jsonify({'message': 'Prediction job queued successfully'}), 200
    except Exception as e:
        logging.error(f"Error in /predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/loadTest/', methods=['POST'])
def load_test():
    req = request.get_json()
    if req is None:
        return jsonify({'error': 'Empty request payload'}), 400
    bot.handle_message(req.get('message', {}))
    return 'Ok'

if __name__ == '__main__':
    set_webhook()
    app.run(host='0.0.0.0', port=8443, debug=True)
