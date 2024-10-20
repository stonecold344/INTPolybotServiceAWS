import telebot
from loguru import logger
import os
import time
import json
import uuid
import boto3
from telebot.types import InputFile
from botocore.exceptions import ClientError

class Bot:
    def __init__(self, token, telegram_chat_url, s3_bucket_name, yolo5_url, aws_region, sqs_url, dynamodb_table):
        self.telegram_bot_client = telebot.TeleBot(token)
        self.telegram_chat_url = telegram_chat_url
        self.s3_bucket_name = s3_bucket_name
        self.yolo5_url = yolo5_url
        self.aws_region = aws_region
        self.sqs_url = sqs_url
        self.dynamodb_table = dynamodb_table
        self.setup_webhook(token)
        logger.info(f'Telegram Bot information:\n{self.telegram_bot_client.get_me()}')
        logger.info(f"Telegram Chat URL: {self.telegram_chat_url}")
        logger.info(f"S3 Bucket Name: {self.s3_bucket_name}")
        logger.info(f"YOLO5 URL: {self.yolo5_url}")
        logger.info(f"AWS Region: {self.aws_region}")
        logger.info(f"SQS URL: {self.sqs_url}")
        logger.info(f"DynamoDB Table: {self.dynamodb_table}")


        logger.info("Starting to initialize DynamoDB client...")
        dynamodb = boto3.resource('dynamodb', region_name=self.aws_region)
        self.table = dynamodb.Table('ChatPredictionState-bennyi')
        logger.info(f"Using DynamoDB table: {self.table}")

    def get_pending_status(self, chat_id):
        logger.info(chat_id)
        try:
            response = self.table.get_item(Key={'chat_id': chat_id})
            logger.info(response)
            if 'Item' in response:
                logger.info(response['Item'])
                return response['Item'].get('pending_prediction', False)
            else:
                return False
        except Exception as e:
            logger.error(f"Error retrieving data: {e}")
            return False

    def set_pending_status(self, chat_id, status):
        try:
            self.table.put_item(
                Item={
                    'chat_id': chat_id,
                    'pending_prediction': status,
                    'timestamp': int(time.time())
                }
            )
        except ClientError as e:
            logger.error(f"Error setting data: {e.response['Error']['Message']}")

    def setup_webhook(self, token):
        webhook_url = f'{self.telegram_chat_url}/{token}/'
        logger.info(f"Webhook URL: {webhook_url}")
        try:
            webhook_info = self.telegram_bot_client.get_webhook_info()
            if webhook_info.url == webhook_url:
                logger.info("Webhook is already set.")
                logger.info(f"Webhook: {webhook_url}")
                return

            self.telegram_bot_client.remove_webhook()
            for attempt in range(1):
                try:
                    self.telegram_bot_client.set_webhook(url=webhook_url, timeout=60)
                    logger.info("Webhook successfully set.")
                    logger.info(f"Webhook: {webhook_url}")
                    return
                except telebot.apihelper.ApiTelegramException as e:
                    if e.error_code == 429:
                        retry_after = int(e.result_json.get('parameters', {}).get('retry_after', 1))
                        time.sleep(retry_after)
                    else:
                        logger.info(f"Webhook URL: {webhook_url}")
                        logger.error(f"Error setting webhook: {e}")
                        break
                time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"Error setting up webhook: {e}")

    def send_text(self, chat_id, text):
        try:
            self.telegram_bot_client.send_message(chat_id, text)
        except Exception as e:
            logger.error(f"Error sending text message: {e}")

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        try:
            self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)
        except Exception as e:
            logger.error(f"Error sending quoted text message: {e}")

    @staticmethod
    def is_current_msg_photo(msg):
        return 'photo' in msg

    def download_user_photo(self, photo_id):
        try:
            file_info = self.telegram_bot_client.get_file(photo_id)
            data = self.telegram_bot_client.download_file(file_info.file_path)
            folder_name = file_info.file_path.split('/')[0]
            os.makedirs(folder_name, exist_ok=True)
            file_path = os.path.join(folder_name, os.path.basename(file_info.file_path))
            with open(file_path, 'wb') as photo:
                photo.write(data)
            logger.info(f'Photo downloaded to: {file_path}')
            return file_path
        except Exception as e:
            logger.error(f"Error downloading photo: {e}")
            return None

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            logger.error(f"Image path {img_path} doesn't exist")
            return

        try:
            self.telegram_bot_client.send_photo(chat_id, InputFile(img_path))
        except Exception as e:
            logger.error(f"Error sending photo: {e}")


class ObjectDetectionBot(Bot):
    def __init__(self, token, telegram_chat_url, s3_bucket_name, yolo5_url, aws_region, sqs_url, dynamodb_table):
        super().__init__(token, telegram_chat_url, s3_bucket_name, yolo5_url, aws_region, sqs_url, dynamodb_table)

        logger.info("Starting to initialize S3 client...")
        self.s3_client = boto3.client('s3', region_name=self.aws_region)
        logger.info("S3 client initialized.")

        logger.info("Starting to initialize SQS client...")
        self.sqs_client = boto3.client('sqs', region_name=self.aws_region)
        logger.info("SQS client initialized.")


    def upload_to_s3(self, file_path):
        file_name = os.path.basename(file_path)
        unique_id = uuid.uuid4()
        object_name = f'docker-project/photos_{unique_id}_{file_name}'

        for attempt in range(2):
            try:
                self.s3_client.upload_file(file_path, self.s3_bucket_name, object_name)
                # Poll S3 to check if object is available
                for retry in range(1):
                    response = self.s3_client.list_objects_v2(Bucket=self.s3_bucket_name, Prefix=object_name)
                    if 'Contents' in response:
                        return object_name
                    time.sleep(5)
                raise TimeoutError("File upload timeout.")
            except ClientError as e:
                logger.error(f"ClientError: {e}")
                raise
            except Exception as e:
                logger.error(f"Error uploading to S3: {e}")
                if attempt < 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def send_message_to_sqs(self, message_body):
        for attempt in range(5):
            try:
                self.sqs_client.send_message(
                    QueueUrl=self.sqs_url,
                    MessageBody=message_body
                )
                return
            except ClientError as e:
                logger.error(f"ClientError: {e}")
                raise
            except Exception as e:
                logger.error(f"Error sending message to SQS: {e}")
                if attempt < 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def handle_message(self, msg):
        if 'chat' not in msg or 'id' not in msg['chat']:
            return

        chat_id = msg['chat']['id']
        pending_status = self.get_pending_status(chat_id)

        if 'text' in msg:
            self.handle_text_message(chat_id, msg['text'])
        elif self.is_current_msg_photo(msg):
            self.handle_photo_message(chat_id, msg)
        else:
            self.send_text(chat_id, 'Unsupported command or message.')

    def handle_text_message(self, chat_id, text):
        if text.startswith('/predict'):
            if self.get_pending_status(chat_id):
                self.send_text(chat_id, 'You already have a pending prediction.')
            else:
                self.set_pending_status(chat_id, True)
                self.send_text(chat_id, 'Please send the photos you want to analyze.')
        else:
            self.send_text(chat_id, 'Unsupported command. Use /predict.')

    def handle_photo_message(self, chat_id, msg):
        if not self.get_pending_status(chat_id):
            self.send_text(chat_id, "Unexpected photo. Please use the /predict command first.")
            return

        photos = msg['photo']
        for photo in photos:
            photo_id = photo['file_id']
            file_path = self.download_user_photo(photo_id)
            if not file_path:
                self.send_text(chat_id, "Failed to process the photo.")
                return

            s3_object_name = self.upload_to_s3(file_path)
            message_body = json.dumps({
                'chat_id': chat_id,
                'photo_id': photo_id,
                'image_url': s3_object_name
            })
            self.send_message_to_sqs(message_body)

        # After processing, set pending status to False
        self.set_pending_status(chat_id, False)
        self.send_text(chat_id, "Photos received! Processing started.")
