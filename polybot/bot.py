import telebot
from loguru import logger
import os
import time
import boto3
from telebot.types import InputFile

class Bot:

    def __init__(self, token, telegram_chat_url):
        self.telegram_bot_client = telebot.TeleBot(token)

        # Remove existing webhooks and set new one
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)
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
        """Downloads the photo sent to the bot."""
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f"Message content of type 'photo' expected")

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Save the photo
        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def send_photo(self, chat_id, img_path):
        """Send a photo to the Telegram chat."""
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(chat_id, InputFile(img_path))

    def handle_message(self, msg):
        """Bot main message handler."""
        logger.info(f'Incoming message: {msg}')
        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')


class ObjectDetectionBot(Bot):

    def __init__(self, token, telegram_chat_url, bucket_name, queue_url):
        super().__init__(token, telegram_chat_url)
        self.s3_client = boto3.client('s3')
        self.sqs_client = boto3.client('sqs')
        self.bucket_name = bucket_name
        self.queue_url = queue_url

    def handle_message(self, msg):
        """Handle photo messages for object detection."""
        logger.info(f'Incoming message: {msg}')

        if self.is_current_msg_photo(msg):
            # Download the user's photo
            photo_path = self.download_user_photo(msg)

            # Upload the photo to S3
            img_name = os.path.basename(photo_path)
            self.s3_client.upload_file(photo_path, self.bucket_name, img_name)
            logger.info(f"Uploaded {img_name} to S3 bucket {self.bucket_name}")

            # Send a message to SQS queue
            message_body = {
                "chat_id": msg['chat']['id'],
                "img_name": img_name
            }
            self.sqs_client.send_message(QueueUrl=self.queue_url, MessageBody=str(message_body))
            logger.info(f"Sent message to SQS queue {self.queue_url}")

            # Notify the user that the image is being processed
            self.send_text(msg['chat']['id'], "Your image is being processed. Please wait...")


