import flask
from flask import request, jsonify
import os
import boto3
from bot import ObjectDetectionBot
import json
import requests
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(dotenv_path='/home/ubuntu/projects/AWSProject-bennyi/polybot/.env')
logging.info("Env file loaded")

app = flask.Flask(__name__)


def get_secret(secret_id):
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name='eu-west-3')

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_id)
        secret = get_secret_value_response['SecretString']
        return json.loads(secret)
    except Exception as e:
        logging.error(f"Error retrieving secret: {e}")
        raise e


# Retrieve the Telegram token
SECRET_ID = "telegram/token"
secrets = get_secret(SECRET_ID)
TELEGRAM_TOKEN = secrets.get('TELEGRAM_TOKEN')

# Environment Variables
TELEGRAM_APP_URL = os.getenv('TELEGRAM_APP_URL')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
YOLO5_URL = os.getenv('YOLO5_URL')
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE')
AWS_REGION = os.getenv('AWS_REGION')
SQS_URL = os.getenv('SQS_URL')

# Ensure all environment variables are loaded
if not all([TELEGRAM_TOKEN, TELEGRAM_APP_URL, S3_BUCKET_NAME, YOLO5_URL, DYNAMODB_TABLE, AWS_REGION, SQS_URL]):
    logging.error(
        f"Missing environment variables: {TELEGRAM_TOKEN}, {TELEGRAM_APP_URL}, {S3_BUCKET_NAME}, {YOLO5_URL}, {DYNAMODB_TABLE}, {AWS_REGION}, {SQS_URL}")
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
        response = requests.get(url)
        webhook_info = response.json()
        logging.info("Webhook info: %s", webhook_info)

        # Check if webhook is already set to the correct URL
        current_url = webhook_info['result'].get('url', None)
        desired_url = f"{TELEGRAM_APP_URL}/{TELEGRAM_TOKEN}/"

        if current_url == desired_url:
            logging.info("Webhook is already set to the desired URL: %s", current_url)
            return
        else:
            logging.info("Setting webhook as it is not set or has a different URL. Current webhook URL: %s",
                         current_url)



        # Set webhook if not already set or has a different URL
        set_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook"
        response = requests.post(set_url, data={"url": desired_url})
        result = response.json()
        logging.info("Set webhook response: %s", result)

        if result.get('ok'):
            logging.info("Webhook set successfully")
        else:
            logging.error("Failed to set webhook: %s", result)

    except Exception as e:
        logging.error(f"Error in setting webhook: {e}")
        raise e


# Function to get webhook info
def get_webhook_info():
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getWebhookInfo"
    response = requests.get(url)
    logging.info("Webhook info: %s", response.json())

@app.route('/', methods=['GET'])
def index():
    return 'Ok'


@app.route(f'/{TELEGRAM_TOKEN}/', methods=['POST'])
def webhook():
    req = request.get_json()
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
        return jsonify({'error': str(e)}), 500


@app.route('/loadTest/', methods=['POST'])
def load_test():
    req = request.get_json()
    bot.handle_message(req.get('message', {}))
    return 'Ok'


if __name__ == "__main__":
    try:
        set_webhook()
        get_webhook_info()
    except Exception as e:
        logging.error(f"Error setting or getting webhook: {e}")
    app.run(host='0.0.0.0', port=8443, debug=True)
