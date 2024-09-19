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
    def __init__(self, token, telegram_chat_url, s3_bucket_name, yolo5_url, aws_region, sqs_url):
        self.telegram_bot_client = telebot.TeleBot(token)
        self.telegram_chat_url = telegram_chat_url
        self.s3_bucket_name = s3_bucket_name
        self.yolo5_url = yolo5_url
        self.aws_region = aws_region
        self.sqs_url = sqs_url
        self.setup_webhook(token)
        logger.info(f'Telegram Bot information:\n{self.telegram_bot_client.get_me()}')

        # Setup DynamoDB
        dynamodb = boto3.resource('dynamodb', region_name=aws_region)
        self.table = dynamodb.Table('ChatPredictionState-bennyi')  # Set table as instance attribute

    # Function to get the pending prediction status for a chat
    def get_pending_status(self, chat_id):
        try:
            response = self.table.get_item(Key={'chat_id': chat_id})
            if 'Item' in response and 'pending_prediction' in response['Item']:
                return response['Item']['pending_prediction']
            else:
                return False
        except Exception as e:
            logger.error(f"Error retrieving data: {e}")
            return False

    # Function to set pending prediction status
    def set_pending_status(self, chat_id, status):
        try:
            self.table.put_item(
                Item={
                    'chat_id': chat_id,
                    'pending_prediction': status,
                    'timestamp': int(time.time())  # Add a timestamp for reference
                }
            )
            logger.info(f"Set pending status for chat_id {chat_id} to {status}")
        except ClientError as e:
            logger.error(f"Error setting data: {e.response['Error']['Message']}")

    def setup_webhook(self, token):
        webhook_url = f'{self.telegram_chat_url}/{token}/'
        try:
            webhook_info = self.telegram_bot_client.get_webhook_info()
            if webhook_info.url == webhook_url:
                logger.info("Webhook is already set.")
                return

            self.telegram_bot_client.remove_webhook()
            for attempt in range(5):
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
                time.sleep(2 ** attempt)
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

    def download_user_photo(self, photo_id):
        file_info = self.telegram_bot_client.get_file(photo_id)
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]
        os.makedirs(folder_name, exist_ok=True)
        file_path = os.path.join(folder_name, os.path.basename(file_info.file_path))
        with open(file_path, 'wb') as photo:
            photo.write(data)
        logger.info(f'Photo downloaded to: {file_path}')
        return file_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            logger.error(f"Image path {img_path} doesn't exist")
            return

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
        logger.info("ObjectDetectionBot initialized.")

    def upload_to_s3(self, file_path):
        file_name = os.path.basename(file_path)
        unique_id = uuid.uuid4()
        object_name = f'docker-project/photos_{unique_id}_{file_name}'

        for attempt in range(3):
            try:
                logger.info(f'Uploading file to S3: {file_path}')
                self.s3_client.upload_file(file_path, self.s3_bucket_name, object_name)

                for retry in range(10):
                    response = self.s3_client.list_objects_v2(Bucket=self.s3_bucket_name, Prefix=object_name)
                    if 'Contents' in response:
                        logger.info("File is available on S3")
                        return object_name
                    logger.info(f"File not available on S3 yet, retrying in 5 seconds... (Attempt {retry + 1}/10)")
                    time.sleep(5)
                raise TimeoutError("File upload timeout. Could not find file after maximum attempts.")

            except ClientError as e:
                logger.error(f"ClientError: {e}")
                raise
            except Exception as e:
                logger.error(f"Error uploading to S3: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def send_message_to_sqs(self, message_body):
        for attempt in range(5):
            try:
                response = self.sqs_client.send_message(
                    QueueUrl=self.sqs_url,
                    MessageBody=message_body
                )
                logger.info(f"Message sent to SQS: {response.get('MessageId')}")
                return
            except ClientError as e:
                logger.error(f"ClientError: {e}")
                raise
            except Exception as e:
                logger.error(f"Error sending message to SQS, retrying... (Attempt {attempt + 1}/5)")
                if attempt < 4:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def handle_message(self, msg):
        logger.info(f'Handling message: {msg}')

        if 'chat' not in msg or 'id' not in msg['chat']:
            logger.error("Message format is incorrect, missing 'chat' or 'id'")
            return

        chat_id = msg['chat']['id']
        pending_status = self.get_pending_status(chat_id)  # Get status from DynamoDB
        logger.info(f'Current pending_prediction state for chat_id {chat_id}: {pending_status}')

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
            self.set_pending_status(chat_id, True)  # Set status in DynamoDB
            self.send_text(chat_id, 'Please send the photos you want to analyze.')
        else:
            self.send_text(chat_id, 'Unsupported command. Please use the /predict command.')

    def handle_photo_message(self, chat_id, msg):
        logger.info(f'Received photo message for chat_id {chat_id}')

        # Check pending prediction state in DynamoDB
        if not self.get_pending_status(chat_id):
            self.send_text(chat_id, "Unexpected photo. Please use the /predict command first.")
            return

        # Download the photo
        photos = msg['photo']
        photo_id = photos[-1]['file_id']
        file_path = self.download_user_photo(photo_id)

        # Upload the photo to S3 and send a job to the SQS queue
        s3_object_name = self.upload_to_s3(file_path)
        message_body = json.dumps({
            'chat_id': chat_id,
            'photo_id': photo_id,
            'file_path': s3_object_name
        })
        self.send_message_to_sqs(message_body)
        self.set_pending_status(chat_id, False)  # Reset status in DynamoDB

        self.send_text(chat_id, "Photo received! Processing will start shortly.")
