import flask
from flask import request, jsonify
import os
import boto3
from bot import ObjectDetectionBot
import json
import requests
from dotenv import load_dotenv
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from the .env file
load_dotenv(dotenv_path='/usr/src/app/.env')  # Update path if needed
logging.info("Environment file loaded")

app = flask.Flask(__name__)

def get_secret(secret_id):
    """Retrieve a secret from AWS Secrets Manager."""
    session = boto3.session.Session()
    logging.info("Session credentials: %s", session.get_credentials())
    client = session.client(service_name='secretsmanager', region_name='eu-west-3')

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_id)
        secret = get_secret_value_response['SecretString']
        return json.loads(secret)
    except Exception as e:
        logging.error("Error retrieving secret: %s", e)
        raise

# Retrieve YOLO5 instance IP
yolo5_instance_ips = []
ec2 = boto3.client('ec2', region_name='eu-west-3')
response = ec2.describe_instances(Filters=[{'Name': 'tag:Name', 'Values': ['aws-yolo5-bennyi']}])

for reservation in response['Reservations']:
    for instance in reservation['Instances']:
        public_ip = instance.get('PublicIpAddress')
        if public_ip:
            yolo5_instance_ips.append(public_ip)
            logging.info("Found YOLO5 instance IP: %s", public_ip)

if yolo5_instance_ips:
    YOLO5_URL = f'http://{yolo5_instance_ips[0]}:8081'  # Use the first IP if there are multiple
    logging.info("YOLO5 service URL: %s", YOLO5_URL)
else:
    logging.error("Could not find YOLO5 instance IP")
    YOLO5_URL = None

# Retrieve the Telegram token
SECRET_ID = "telegram/token"
secrets = get_secret(SECRET_ID)
TELEGRAM_TOKEN = secrets.get("TELEGRAM_TOKEN")

# Use the token in your application
logging.info("Telegram Token: %s", TELEGRAM_TOKEN)

# Environment Variables
TELEGRAM_APP_URL = os.getenv('TELEGRAM_APP_URL')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE')
AWS_REGION = os.getenv('AWS_REGION')
SQS_URL = os.getenv('SQS_URL')
logging.info("Environment variables: TELEGRAM_TOKEN=%s, TELEGRAM_APP_URL=%s, S3_BUCKET_NAME=%s, YOLO5_URL=%s, DYNAMODB_TABLE=%s, AWS_REGION=%s, SQS_URL=%s",
             TELEGRAM_TOKEN, TELEGRAM_APP_URL, S3_BUCKET_NAME, YOLO5_URL, DYNAMODB_TABLE, AWS_REGION, SQS_URL)

# Ensure all environment variables are loaded
if not all([TELEGRAM_TOKEN, TELEGRAM_APP_URL, S3_BUCKET_NAME, YOLO5_URL, DYNAMODB_TABLE, AWS_REGION, SQS_URL]):
    logging.error("One or more environment variables are missing")
    raise ValueError("One or more environment variables are missing")

# Initialize DynamoDB
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
table = dynamodb.Table(DYNAMODB_TABLE)

# Define bot object globally
bot = ObjectDetectionBot(TELEGRAM_TOKEN, TELEGRAM_APP_URL, S3_BUCKET_NAME, YOLO5_URL, AWS_REGION, SQS_URL)

def set_webhook():
    """Set the webhook for the Telegram bot."""
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getWebhookInfo"
            response = requests.get(url)
            response.raise_for_status()
            webhook_info = response.json()
            logging.info("Webhook info: %s", webhook_info)

            current_url = webhook_info['result'].get('url', None)
            desired_url = f"{TELEGRAM_APP_URL}/{TELEGRAM_TOKEN}/"

            if current_url == desired_url:
                logging.info("Webhook is already set to the desired URL: %s", current_url)
                return
            else:
                logging.info("Setting webhook as it is not set or has a different URL. Current webhook URL: %s", current_url)

            set_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook"
            response = requests.post(set_url, data={"url": desired_url})
            response.raise_for_status()
            result = response.json()
            logging.info("Set webhook response: %s", result)

            if result.get('ok'):
                logging.info("Webhook set successfully")
                return
            else:
                logging.error("Failed to set webhook: %s", result)

        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429:
                retry_after = int(http_err.response.headers.get('Retry-After', 1))
                logging.info("Rate limit exceeded. Retrying after %d seconds...", retry_after)
                time.sleep(retry_after)
            else:
                logging.error("HTTP error occurred: %s", http_err)
                raise
        except Exception as e:
            logging.error("Error occurred while setting webhook: %s", e)
            raise

@app.route('/', methods=['GET'])
def index():
    return 'Ok'

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    req = request.get_json()
    logging.info("Received request: %s", req)
    if req is None:
        logging.warning("Received empty request payload")
        return jsonify({'error': 'Empty request payload'}), 400
    bot.handle_message(req.get('message', {}))
    return 'Ok'

@app.route('/results', methods=['POST'])
def results():
    prediction_id = request.args.get('predictionId')
    if not prediction_id:
        logging.error("Missing predictionId")
        return jsonify({'error': 'predictionId is required'}), 400

    try:
        response = table.get_item(Key={'prediction_id': prediction_id})
        if 'Item' not in response:
            logging.error("Prediction not found for ID: %s", prediction_id)
            return jsonify({'error': 'Prediction not found'}), 404
        prediction_summary = response['Item']
        chat_id = prediction_summary['chat_id']
        labels = prediction_summary['labels']
        text_results = '\n'.join([f"{label['class']} : {label['count']}" for label in labels])
        bot.send_text(chat_id, text_results)
        return 'Ok'
    except Exception as e:
        logging.error("Error fetching prediction: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req = request.get_json()
        if req is None:
            logging.warning("Received empty request payload")
            return jsonify({'error': 'Empty request payload'}), 400

        image_url = req.get('image_url')
        if not image_url:
            logging.error("Missing image_url")
            return jsonify({'error': 'image_url is required'}), 400

        if YOLO5_URL is None:
            logging.error("YOLO5_URL is not set")
            return jsonify({'error': 'YOLO5 service URL is not available'}), 500

        response = requests.post(YOLO5_URL, json={"image_url": image_url})
        if response.status_code != 200:
            logging.error("Error from YOLO5 service: %s", response.text)
            return jsonify({'error': 'Error from YOLO5 service'}), 500

        result = response.json()
        prediction_id = result.get('prediction_id')
        if not prediction_id:
            logging.error("Missing prediction_id in YOLO5 response")
            return jsonify({'error': 'Prediction ID not found in YOLO5 response'}), 500

        return jsonify({'prediction_id': prediction_id}), 200

    except Exception as e:
        logging.error("Error in /predict endpoint: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/loadTest/', methods=['POST'])
def load_test():
    req = request.get_json()
    if req is None:
        logging.warning("Received empty request payload")
        return jsonify({'error': 'Empty request payload'}), 400
    bot.handle_message(req.get('message', {}))
    return 'Ok'

if __name__ == '__main__':
    set_webhook()  # Set the webhook when the app starts
    app.run(host='0.0.0.0', port=8443, debug=True)
