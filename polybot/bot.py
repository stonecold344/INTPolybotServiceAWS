import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
import json
import uuid
import requests
import boto3

class Bot:
    def __init__(self, token, telegram_chat_url, s3_bucket_name, yolo5_url, aws_region, sqs_url):
        self.telegram_bot_client = telebot.TeleBot(token)
        self.telegram_chat_url = telegram_chat_url
        self.s3_bucket_name = s3_bucket_name
        self.yolo5_url = yolo5_url
        self.aws_region = aws_region
        self.sqs_url = sqs_url
        self.setup_webhook(token)
        logger.info(f'Telegram Bot information:\n{self.telegram_bot_client.get_me()}')

    def setup_webhook(self, token):
        webhook_url = f'{self.telegram_chat_url}/{token}/'
        try:
            # Check current webhook info
            webhook_info = self.telegram_bot_client.get_webhook_info()
            if webhook_info.url == webhook_url:
                logger.info("Webhook is already set.")
                return

            # Remove existing webhook
            self.telegram_bot_client.remove_webhook()

            # Set new webhook
            retry_attempts = 5
            for attempt in range(retry_attempts):
                try:
                    self.telegram_bot_client.set_webhook(url=webhook_url, timeout=60)
                    logger.info("Webhook successfully set.")
                    return
                except telebot.apihelper.ApiTelegramException as e:
                    if e.error_code == 429:  # Rate limit exceeded
                        retry_after = int(e.result_json.get('parameters', {}).get('retry_after', 1))
                        logger.info(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                        time.sleep(retry_after)
                    else:
                        logger.error(f"Error setting webhook: {e}")
                        break
                time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logger.error(f"Error occurred while setting webhook: {e}")

    def send_text(self, chat_id, text):
        try:
            self.telegram_bot_client.send_message(chat_id, text)
            logger.info(f"Sent message to chat_id {chat_id}: {text}")
        except Exception as e:
            logger.error(f"Error sending message to chat_id {chat_id}: {e}")

    @staticmethod
    def is_current_msg_photo(msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        file_id = msg['photo'][-1]['file_id']
        file_info = self.telegram_bot_client.get_file(file_id)
        file_path = file_info.file_path
        file_url = f"https://api.telegram.org/file/bot{self.telegram_bot_client.token}/{file_path}"
        response = requests.get(file_url)
        if response.status_code == 200:
            photo_name = f"{uuid.uuid4().hex}.jpg"
            with open(photo_name, 'wb') as f:
                f.write(response.content)
            return photo_name
        else:
            raise Exception(f"Failed to download photo. Status code: {response.status_code}")

    def upload_to_s3(self, file_path):
        try:
            s3_client = boto3.client('s3', region_name=self.aws_region)
            photo_name = os.path.basename(file_path)
            s3_client.upload_file(file_path, self.s3_bucket_name, photo_name)
            logger.info(f"Uploaded photo to S3 with name: {photo_name}")
            return photo_name
        except Exception as e:
            logger.error(f"Error uploading photo to S3: {e}")
            raise e

    def send_message_to_sqs(self, message_body):
        try:
            sqs_client = boto3.client('sqs', region_name=self.aws_region)
            response = sqs_client.send_message(QueueUrl=self.sqs_url, MessageBody=message_body)
            logger.info(f"Message sent to SQS: {response.get('MessageId')}")
        except Exception as e:
            logger.error(f"Error sending message to SQS: {e}")

    def handle_message(self, msg):
        if 'chat' not in msg or 'id' not in msg['chat']:
            logger.error("Message format is incorrect, missing 'chat' or 'id'")
            return

        chat_id = msg['chat']['id']

        if 'text' in msg:
            text = msg['text']
            logger.info(f'Received text message: {text}')
            if text.startswith('/predict'):
                self.send_text(chat_id, 'Please send a photo to analyze.')
                logger.info(f'Waiting for photo from chat_id {chat_id}')
            else:
                self.send_text(chat_id, 'Unsupported command. Please use the /predict command with a photo.')

        elif self.is_current_msg_photo(msg):
            logger.info(f'Received photo message for chat_id {chat_id}')
            try:
                photo_path = self.download_user_photo(msg)
                logger.info(f'Photo downloaded to: {photo_path}')

                photo_name = self.upload_to_s3(photo_path)
                logger.info(f'Photo uploaded to S3 with name: {photo_name}')

                photo_url = f'https://{self.s3_bucket_name}.s3.{self.aws_region}.amazonaws.com/{photo_name}'
                logger.info(f'Photo URL: {photo_url}')

                # Send the S3 photo URL and chat ID to SQS
                self.queue_prediction_job(photo_url, chat_id)

            except Exception as e:
                logger.error(f"Error processing photo message: {e}")
                self.send_text(chat_id, f"An error occurred: {e}")
        else:
            self.send_text(chat_id, 'Unsupported command or message.')
            logger.info(f'Unsupported message type for chat_id {chat_id}.')

    def queue_prediction_job(self, photo_url, chat_id):
        message_body = json.dumps({'image_url': photo_url, 'chat_id': chat_id})
        try:
            self.send_message_to_sqs(message_body)
            logger.info(f"Prediction job queued with photo URL: {photo_url}")
        except Exception as e:
            logger.error(f"Error queuing prediction job: {e}")
