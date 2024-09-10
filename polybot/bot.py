import json

import telebot
from loguru import logger
import os
import boto3
from telebot.types import InputFile

class Bot:
    def __init__(self, token, telegram_chat_url):
        self.telegram_bot_client = telebot.TeleBot(token)
        self.telegram_bot_client.remove_webhook()
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60)
        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    @staticmethod
    def is_current_msg_photo(msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f"Message content of type 'photo' expected")
        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        file_path = os.path.join(folder_name, os.path.basename(file_info.file_path))
        with open(file_path, 'wb') as photo:
            photo.write(data)
        return file_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")
        self.telegram_bot_client.send_photo(chat_id, InputFile(img_path))

    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')
        if 'text' in msg:
            self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')

class ObjectDetectionBot(Bot):
    def __init__(self, token, telegram_chat_url, bucket_name, yolo5_url, aws_region, sqs_queue_url):
        super().__init__(token, telegram_chat_url)
        self.sqs_client = boto3.client('sqs', region_name=aws_region)
        self.s3_client = boto3.client('s3', region_name=aws_region)
        self.yolo5_url = yolo5_url
        self.bucket_name = bucket_name
        self.sqs_queue_url = sqs_queue_url

    def handle_message(self, msg):
        logger.info(f'Handling message: {msg}')
        if self.is_current_msg_photo(msg):
            try:
                photo_path = self.download_user_photo(msg)
                img_name = os.path.basename(photo_path)
                self.s3_client.upload_file(photo_path, self.bucket_name, img_name)
                logger.info(f"Uploaded {img_name} to S3 bucket {self.bucket_name}")

                # Send message to SQS queue
                message_body = {
                    "chat_id": msg['chat']['id'],
                    "img_name": img_name
                }
                self.sqs_client.send_message(QueueUrl=self.sqs_queue_url, MessageBody=json.dumps(message_body))

                self.send_text(msg['chat']['id'], "Your image is being processed. Please wait...")
            except Exception as e:
                logger.error(f"Error handling photo message: {e}")
                self.send_text(msg['chat']['id'], "An error occurred while processing your photo.")
