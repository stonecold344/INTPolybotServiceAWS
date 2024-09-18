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
        webhook_url = f'{self.telegram_chat_url}/{token}/'  # Make sure HTTPS is used for production
        try:
            webhook_info = self.telegram_bot_client.get_webhook_info()
            if webhook_info.url == webhook_url:
                logger.info("Webhook is already set.")
                return

            self.telegram_bot_client.remove_webhook()

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
            logger.error(f"Error setting up webhook: {e}")

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
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        try:
            self.telegram_bot_client.send_photo(chat_id, InputFile(img_path))
            logger.info(f'Sent photo to chat_id {chat_id}')
        except Exception as e:
            logger.error(f"Error sending photo: {e}")


class ObjectDetectionBot(Bot):
    def __init__(self, token, telegram_chat_url, s3_bucket_name, yolo5_url, aws_region, sqs_url):
        super().__init__(token, telegram_chat_url, s3_bucket_name, yolo5_url, aws_region, sqs_url)
        self.s3_client = boto3.client('s3', region_name=self.aws_region)
        self.sqs_client = boto3.client('sqs', region_name=self.aws_region)
        self.pending_prediction = {}
        self.images_queue = {}
        logger.info("ObjectDetectionBot initialized.")

    def upload_to_s3(self, file_path):
        file_name = os.path.basename(file_path)
        unique_id = uuid.uuid4()
        object_name = f'docker-project/photos_{unique_id}_{file_name}'

        retry_attempts = 3
        for attempts in range(retry_attempts):
            try:
                logger.info(f'Uploading file to S3: {file_path}')
                self.s3_client.upload_file(file_path, self.s3_bucket_name, object_name)

                max_attempts = 10
                for attempt in range(max_attempts):
                    response = self.s3_client.list_objects_v2(Bucket=self.s3_bucket_name, Prefix=object_name)
                    if 'Contents' in response:
                        logger.info("File is available on S3")
                        return object_name
                    else:
                        logger.info(
                            f"File is not available on S3 yet, retrying in 5 seconds... (Attempt {attempt + 1}/{max_attempts})")
                        time.sleep(5)
                else:
                    raise TimeoutError("File upload timeout. Could not find file after maximum attempts.")

            except Exception as e:
                logger.error(f"Error uploading to S3: {e}")
                if attempts < retry_attempts - 1:
                    time.sleep(2 ** attempts)  # Exponential backoff
                else:
                    raise

    def send_message_to_sqs(self, message_body):
        retry_attempts = 5
        for attempt in range(retry_attempts):
            try:
                response = self.sqs_client.send_message(
                    QueueUrl=self.sqs_url,
                    MessageBody=message_body
                )
                logger.info(f"Message sent to SQS: {response.get('MessageId')}")
                return
            except Exception as e:
                if attempt < retry_attempts - 1:
                    logger.error(f"Error sending message to SQS, retrying... (Attempt {attempt + 1}/{retry_attempts})")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Error sending message to SQS: {e}")
                    raise

    def handle_message(self, msg):
        logger.info(f'Handling message: {msg}')

        if 'chat' not in msg or 'id' not in msg['chat']:
            logger.error("Message format is incorrect, missing 'chat' or 'id'")
            return

        chat_id = msg['chat']['id']
        logger.info(
            f'Current pending_prediction state for chat_id {chat_id}: {self.pending_prediction.get(chat_id, False)}')

        if 'text' in msg:
            self.handle_text_message(chat_id, msg['text'])
        elif self.is_current_msg_photo(msg):
            self.handle_photo_message(chat_id, msg)
        else:
            self.send_text(chat_id, 'Unsupported command or message.')
            logger.info(f'Unsupported message type for chat_id {chat_id}.')

    def handle_text_message(self, chat_id, text):
        logger.info(f'Received text message: {text}')
        if text.startswith('/predict'):
            self.pending_prediction[chat_id] = True
            self.images_queue[chat_id] = []
            self.send_text(chat_id, 'Please send the photos you want to analyze.')
            logger.info(f'Set pending_prediction for chat_id {chat_id} to True')
        elif text.startswith('/done'):
            if self.pending_prediction.get(chat_id, False):
                self.process_images(chat_id)
            else:
                self.send_text(chat_id, 'No prediction is pending. Please use the /predict command first.')
        else:
            self.send_text(chat_id, 'Unsupported command. Please use the /predict command followed by the /done command when finished.')

    def handle_photo_message(self, chat_id, msg):
        logger.info(f'Received photo message for chat_id {chat_id}')
        if not self.pending_prediction.get(chat_id, False):
            self.send_text(chat_id, 'Please use the /predict command before sending photos.')
            logger.info(f'Ignored photo message for chat_id {chat_id} as no prediction is pending.')
            return

        try:
            photo_path = self.download_user_photo(msg)
            logger.info(f'Photo downloaded to: {photo_path}')

            # Ensure photos are not processed prematurely
            if chat_id not in self.images_queue:
                self.images_queue[chat_id] = []

            self.images_queue[chat_id].append(photo_path)

        except (RuntimeError, TimeoutError) as e:
            logger.error(f"Error occurred: {e}")
            self.send_text(chat_id, f"An error occurred: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            self.send_text(chat_id, f"An unexpected error occurred: {e}")

    def process_images(self, chat_id):
        logger.info(f'Processing images for chat_id {chat_id}')
        # Process all images in the queue
        if chat_id in self.images_queue and self.images_queue[chat_id]:
            for photo_path in self.images_queue[chat_id]:
                try:
                    s3_object = self.upload_to_s3(photo_path)
                    logger.info(f'Image uploaded to S3: {s3_object}')

                    message = json.dumps({
                        'chat_id': chat_id,
                        'image_path': s3_object
                    })
                    self.send_message_to_sqs(message)

                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                    self.send_text(chat_id, f"Error processing image: {e}")
            self.send_text(chat_id, "Images are being processed.")
            self.pending_prediction[chat_id] = False
            self.images_queue[chat_id] = []
        else:
            self.send_text(chat_id, 'No images to process.')
