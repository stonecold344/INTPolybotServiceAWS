import json
import time
import uuid
import requests
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
        logger.info(f'Telegram Bot information:\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        try:
            self.telegram_bot_client.send_message(chat_id, text)
            logger.info(f'Sent text message to chat_id {chat_id}')
        except Exception as e:
            logger.error(f"Error sending text message: {e}")

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        try:
            self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)
            logger.info(f'Sent quoted text message to chat_id {chat_id}')
        except Exception as e:
            logger.error(f"Error sending quoted text message: {e}")

    @staticmethod
    def is_current_msg_photo(msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        if not self.is_current_msg_photo(msg):
            raise RuntimeError("Message content of type 'photo' expected")

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        file_path = os.path.join(folder_name, os.path.basename(file_info.file_path))
        with open(file_path, 'wb') as photo:
            photo.write(data)
        logger.info(f'Photo downloaded to: {file_path}')
        return file_path

    def send_photo(self, chat_id, img_path):
        try:
            with open(img_path, 'rb') as photo:
                self.telegram_bot_client.send_photo(chat_id, photo)
            logger.info(f'Sent photo to chat_id {chat_id}')
        except Exception as e:
            logger.error(f"Error sending photo: {e}")

    def handle_message(self, msg):
        if not self.is_current_msg_photo(msg):
            logger.warning(f"Received message without photo: {msg}")
            return

        file_path = self.download_user_photo(msg)
        object_key = self.upload_to_s3(file_path)
        chat_id = msg['chat']['id']
        self.send_text(chat_id, "Your photo is being processed. Please wait...")
        # Call the YOLO service to process the image
        self.trigger_yolo_service(object_key, chat_id)

    def upload_to_s3(self, file_path):
        file_name = os.path.basename(file_path)
        unique_id = uuid.uuid4()
        object_name = f'docker-project/photos_{unique_id}_{file_name}'

        s3_client = boto3.client('s3')
        try:
            logger.info(f'Uploading file to S3: {file_path}')
            s3_client.upload_file(file_path, os.getenv('S3_BUCKET_NAME'), object_name)

            # Check if file is uploaded with retries
            max_attempts = 10
            for attempt in range(max_attempts):
                response = s3_client.list_objects_v2(Bucket=os.getenv('S3_BUCKET_NAME'), Prefix=object_name)
                if 'Contents' in response:
                    logger.info("File is available on S3")
                    return object_name
                else:
                    logger.info(
                        f"File is not available on S3 yet, retrying in 5 seconds... (Attempt {attempt + 1}/{max_attempts})")
                    time.sleep(5)
            else:
                raise TimeoutError("File upload timeout. Could not find file after 10 attempts.")

        except Exception as e:
            logger.error(f"Error uploading to S3: {e}")
            raise

    def trigger_yolo_service(self, object_key, chat_id):
        payload = {
            'image_url': f"https://{os.getenv('S3_BUCKET_NAME')}.s3.amazonaws.com/{object_key}",
            'chat_id': chat_id
        }
        try:
            response = requests.post(os.getenv('YOLO5_URL'), json=payload)
            if response.status_code != 200:
                logger.error(f"Error from YOLO5 service: {response.text}")
                self.send_text(chat_id, "There was an error processing your photo.")
        except Exception as e:
            logger.error(f"Error triggering YOLO service: {e}")
            self.send_text(chat_id, "There was an error processing your photo.")
