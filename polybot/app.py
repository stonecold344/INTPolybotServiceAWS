import flask
from flask import request, jsonify
import os
import boto3
from dotenv import load_dotenv
from bot import ObjectDetectionBot
import json
import requests

# Load environment variables from .env file
load_dotenv()

app = flask.Flask(__name__)

def get_secret(secret_id):
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name='eu-west-3')

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_id)
        secret = get_secret_value_response['SecretString']
        return json.loads(secret)
    except Exception as e:
        print(f"Error retrieving secret: {e}")
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
SQS_QUEUE_URL = os.getenv('SQS_URL')

# Ensure all environment variables are loaded
if not all([TELEGRAM_TOKEN, TELEGRAM_APP_URL, S3_BUCKET_NAME, YOLO5_URL, DYNAMODB_TABLE, AWS_REGION, SQS_QUEUE_URL]):
    print(TELEGRAM_TOKEN, TELEGRAM_APP_URL, S3_BUCKET_NAME, YOLO5_URL, DYNAMODB_TABLE, AWS_REGION, SQS_QUEUE_URL)
    raise ValueError("One or more environment variables are missing")

# Initialize DynamoDB
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
table = dynamodb.Table(DYNAMODB_TABLE)

# Define bot object globally
bot = ObjectDetectionBot(TELEGRAM_TOKEN, TELEGRAM_APP_URL, S3_BUCKET_NAME, YOLO5_URL, SQS_QUEUE_URL)

def set_webhook():
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook"
    response = requests.post(url, data={"url": f"{TELEGRAM_APP_URL}/{TELEGRAM_TOKEN}/"})
    print("Set webhook response:", response.json())

def get_webhook_info():
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getWebhookInfo"
    response = requests.get(url)
    print("Webhook info:", response.json())

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
    set_webhook()
    get_webhook_info()
    app.run(host='0.0.0.0', port=8443)
