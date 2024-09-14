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
load_dotenv(dotenv_path=os.getenv("ENV_PATH", "/usr/src/app/.env"))  # Ensure ENV_PATH is set if dynamic
logging.info("Env file loaded")

app = flask.Flask(__name__)

def get_secret(secret_id):
    """Retrieve secret from AWS Secrets Manager."""
    session = boto3.session.Session()
    logging.info(f"Session credentials: {session.get_credentials()}")
    client = session.client(service_name='secretsmanager', region_name='eu-west-3')

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_id)
        secret = get_secret_value_response['SecretString']
        return json.loads(secret)
    except Exception as e:
        logging.error(f"Error retrieving secret: {e}")
        raise e

def get_yolo5_instance_ip():
    """Get the public IP of the YOLO5 instance using EC2 describe_instances."""
    try:
        ec2 = boto3.client('ec2', region_name='eu-west-3')
        response = ec2.describe_instances(Filters=[{'Name': 'tag:Name', 'Values': ['aws-yolo5-bennyi']}])
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                public_ip = instance.get('PublicIpAddress')
                if public_ip:
                    logging.info(f"Found YOLO5 instance IP: {public_ip}")
                    return public_ip
        logging.error("Could not find YOLO5 instance IP")
        return None
    except Exception as e:
        logging.error(f"Error fetching YOLO5 instance IP: {e}")
        return None

# Retrieve YOLO5 instance IP
yolo5_instance_ip = get_yolo5_instance_ip()

if yolo5_instance_ip:
    YOLO5_URL = f'http://{yolo5_instance_ip}:8081'
else:
    YOLO5_URL = None  # YOLO5 URL will remain None if not found

# Retrieve the Telegram token
SECRET_ID = "telegram/token"
try:
    secrets = get_secret(SECRET_ID)
    TELEGRAM_TOKEN = secrets.get("TELEGRAM_TOKEN")
except Exception as e:
    logging.error("Error retrieving Telegram token")
    TELEGRAM_TOKEN = None

# Ensure all required environment variables are loaded
required_vars = [TELEGRAM_TOKEN, YOLO5_URL]
if not all(required_vars):
    logging.error("Missing critical configuration (Telegram token, YOLO5 URL)")
    raise ValueError("One or more required configurations are missing")

# Retrieve additional environment variables
TELEGRAM_APP_URL = os.getenv('TELEGRAM_APP_URL')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE')
AWS_REGION = os.getenv('AWS_REGION')
SQS_URL = os.getenv('SQS_URL')

logging.info(f"Loaded environment variables: TELEGRAM_APP_URL={TELEGRAM_APP_URL}, S3_BUCKET_NAME={S3_BUCKET_NAME}, "
             f"DYNAMODB_TABLE={DYNAMODB_TABLE}, AWS_REGION={AWS_REGION}, SQS_URL={SQS_URL}")

# Initialize DynamoDB and ObjectDetectionBot
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
table = dynamodb.Table(DYNAMODB_TABLE)
bot = ObjectDetectionBot(TELEGRAM_TOKEN, TELEGRAM_APP_URL, S3_BUCKET_NAME, YOLO5_URL, AWS_REGION, SQS_URL)

def set_webhook():
    """Set or update the Telegram bot webhook."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getWebhookInfo"
        webhook_info = requests.get(url).json()
        logging.info("Current webhook info: %s", webhook_info)

        desired_url = f"{TELEGRAM_APP_URL}/{TELEGRAM_TOKEN}/"
        current_url = webhook_info['result'].get('url', None)

        if current_url == desired_url:
            logging.info("Webhook already set to the desired URL: %s", current_url)
            return

        # Set new webhook
        set_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook"
        response = requests.post(set_url, data={"url": desired_url}).json()
        if response.get('ok'):
            logging.info("Webhook set successfully")
        else:
            logging.error(f"Failed to set webhook: {response}")
    except Exception as e:
        logging.error(f"Error setting webhook: {e}")

@app.route('/', methods=['GET'])
def index():
    return 'Ok'

@app.route(f'/{TELEGRAM_TOKEN}/', methods=['POST'])
def webhook():
    req = request.get_json()
    logging.info("Received webhook request: %s", req)
    if not req:
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
            return jsonify({'error': 'Prediction not found'}), 404
        prediction_summary = response['Item']
        labels = prediction_summary['labels']
        text_results = '\n'.join([f"{label['class']} : {label['count']}" for label in labels])
        bot.send_text(prediction_summary['chat_id'], text_results)
        return 'Ok'
    except Exception as e:
        logging.error(f"Error fetching prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req = request.get_json()
        if not req:
            return jsonify({'error': 'Empty request payload'}), 400
        image_url = req.get('image_url')
        if not image_url:
            return jsonify({'error': 'image_url is required'}), 400

        # Send prediction request to YOLO5
        response = requests.post(YOLO5_URL, json={"image_url": image_url})
        if response.status_code != 200:
            return jsonify({'error': 'Error from YOLO5 service'}), 500

        # Return YOLO5 result
        result = response.json()
        return jsonify(result), 200
    except Exception as e:
        logging.error(f"Error in predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    set_webhook()  # Set webhook on app start
    app.run(host='0.0.0.0', port=8443, ssl_context='adhoc', debug=True)
