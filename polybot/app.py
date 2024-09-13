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
load_dotenv(dotenv_path='/usr/src/app/.env')  # Update path if needed
logging.info("Env file loaded")

app = flask.Flask(__name__)

def get_secret(secret_id):
    session = boto3.session.Session()
    logging.info("Session credentials: %s", session.get_credentials())
    client = session.client(service_name='secretsmanager', region_name='eu-west-3')

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_id)
        secret = get_secret_value_response['SecretString']
        return json.loads(secret)
    except Exception as e:
        logging.error(f"Error retrieving secret: {e}")
        raise e

# Retrieve YOLO5 instance IP
yolo5_instance_ips = []
ec2 = boto3.client('ec2', region_name='eu-west-3')
response = ec2.describe_instances(Filters=[{'Name': 'tag:Name', 'Values': ['aws-yolo5-bennyi']}])

for reservation in response['Reservations']:
    for instance in reservation['Instances']:
        public_ip = instance.get('PublicIpAddress')
        if public_ip:
            yolo5_instance_ips.append(public_ip)
            logging.info(f"Found YOLO5 instance IP: {public_ip}")

if yolo5_instance_ips:
    YOLO5_URL = f'http://{yolo5_instance_ips[0]}:8081'  # Use the first IP if there are multiple
    logging.info(f"YOLO5 service URL: {YOLO5_URL}")
else:
    logging.error("Could not find YOLO5 instance IP")
    YOLO5_URL = None  # Handle the case where YOLO5_URL is not set

# Retrieve the Telegram token
SECRET_ID = "telegram/token"
secrets = get_secret(SECRET_ID)
TELEGRAM_TOKEN = secrets.get("TELEGRAM_TOKEN")

# Use the token in your application
logging.info(f"Telegram Token: {TELEGRAM_TOKEN}")

# Environment Variables
TELEGRAM_APP_URL = os.getenv('TELEGRAM_APP_URL')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE')
AWS_REGION = os.getenv('AWS_REGION')
SQS_URL = os.getenv('SQS_URL')
logging.info(f"Env variables: TELEGRAM_TOKEN={TELEGRAM_TOKEN}, TELEGRAM_APP_URL={TELEGRAM_APP_URL}, "
             f"S3_BUCKET_NAME={S3_BUCKET_NAME}, YOLO5_URL={YOLO5_URL}, DYNAMODB_TABLE={DYNAMODB_TABLE}, "
             f"AWS_REGION={AWS_REGION}, SQS_URL={SQS_URL}")

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
    try:
        # Get current webhook info
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getWebhookInfo"
        responses = requests.get(url)
        webhook_info = responses.json()
        logging.info("Webhook info: %s", webhook_info)

        # Check if webhook is already set to the correct URL
        current_url = webhook_info['result'].get('url', None)
        desired_url = f"{TELEGRAM_APP_URL}/{TELEGRAM_TOKEN}/"

        if current_url == desired_url:
            logging.info("Webhook is already set to the desired URL: %s", current_url)
            return
        else:
            logging.info("Setting webhook as it is not set or has a different URL. Current webhook URL: %s", current_url)

        # Set webhook if not already set or has a different URL
        set_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook"
        responses = requests.post(set_url, data={"url": desired_url})
        result = responses.json()
        logging.info("Set webhook response: %s", result)

        if result.get('ok'):
            logging.info("Webhook set successfully")
        else:
            logging.error("Failed to set webhook: %s", result)

    except Exception as e:
        logging.error(f"Error occurred while setting webhook: {e}")

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
        responses = table.get_item(Key={'prediction_id': prediction_id})
        if 'Item' not in responses:
            logging.error(f"Prediction not found for ID: {prediction_id}")
            return jsonify({'error': 'Prediction not found'}), 404
        prediction_summary = responses['Item']
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
            logging.warning("Received empty request payload")
            return jsonify({'error': 'Empty request payload'}), 400

        # Extract necessary fields from request
        image_url = req.get('image_url')
        if not image_url:
            logging.error("Missing image_url")
            return jsonify({'error': 'image_url is required'}), 400

        if YOLO5_URL is None:
            logging.error("YOLO5_URL is not set")
            return jsonify({'error': 'YOLO5 service URL is not available'}), 500

        # Call YOLO5 service for prediction
        responses = requests.post(YOLO5_URL, json={"image_url": image_url})
        if responses.status_code != 200:
            logging.error(f"Error from YOLO5 service: {responses.text}")
            return jsonify({'error': 'Error from YOLO5 service'}), 500

        # Process YOLO5 response
        result = responses.json()
        prediction_id = result.get('prediction_id')
        if not prediction_id:
            logging.error("Missing prediction_id in YOLO5 response")
            return jsonify({'error': 'Prediction ID not found in YOLO5 response'}), 500

        return jsonify({'prediction_id': prediction_id}), 200

    except Exception as e:
        logging.error(f"Error in /predict endpoint: {e}")
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
