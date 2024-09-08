import flask
from flask import request, jsonify
import os
import boto3
from dotenv import load_dotenv
from bot import ObjectDetectionBot

# Load environment variables from .env file
load_dotenv()

app = flask.Flask(__name__)

# Environment Variables
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_APP_URL = os.getenv('TELEGRAM_APP_URL')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
YOLO5_URL = os.getenv('YOLO5_URL')
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE')

# Initialize DynamoDB
dynamodb = boto3.resource('dynamodb', region_name=os.getenv('AWS_REGION'))
table = dynamodb.Table(DYNAMODB_TABLE)

# Define bot object globally
bot = ObjectDetectionBot(TELEGRAM_TOKEN, TELEGRAM_APP_URL, S3_BUCKET_NAME, YOLO5_URL)

@app.route('/', methods=['GET'])
def index():
    return 'Ok'

@app.route(f'/{TELEGRAM_TOKEN}/', methods=['POST'])
def webhook():
    req = request.get_json()
    bot.handle_message(req['message'])
    return 'Ok'

@app.route(f'/results', methods=['POST'])
def results():
    prediction_id = request.args.get('predictionId')
    if not prediction_id:
        return jsonify({'error': 'predictionId is required'}), 400

    try:
        response = table.get_item(Key={'prediction_id': prediction_id})
        if 'Item' not in response:
            return jsonify({'error': 'Prediction not found'}), 404
        prediction_summary = response['Item']
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    chat_id = prediction_summary['chat_id']
    labels = prediction_summary['labels']
    text_results = '\n'.join([f"{label['class']} : {label['count']}" for label in labels])
    bot.send_text(chat_id, text_results)

    return 'Ok'

@app.route(f'/loadTest/', methods=['POST'])
def load_test():
    req = request.get_json()
    bot.handle_message(req['message'])
    return 'Ok'

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8443)
