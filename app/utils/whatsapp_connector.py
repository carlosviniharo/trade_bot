from typing import Text, Dict, Any, Optional, List
import requests
import json

from app.core.logging import AppLogger

# Initialize logging
logger = AppLogger.get_logger()

class WhatsAppOutput:
    """Output channel for WhatsApp Cloud API.

    This class allows sending messages to the WhatsApp Cloud API using the
    official Graph API provided by Facebook. It supports sending text messages,
    batch messages, and custom JSON payloads.

    Attributes:
        access_token (Text): The access token for authenticating with the WhatsApp API.
        phone_number_id (Text): The ID of the phone number associated with the WhatsApp account.
        api_url (Text): The API endpoint for sending messages.
    """

    def __init__(self, access_token: Text, phone_number_id: Text) -> None:
        """
        Args:
            access_token (Text): The access token for the WhatsApp Cloud API.
            phone_number_id (Text): The ID of the phone number used for sending messages.
        """
        self.access_token = access_token
        self.phone_number_id = phone_number_id
        self.api_url = f"https://graph.facebook.com/v13.0/{phone_number_id}/messages"

    async def send_text_message(self, recipient_id: Text, message: Text, **kwargs: Any) -> None:
        """Sends a text message to a recipient.

        Args:
            recipient_id (Text): The WhatsApp ID of the message recipient.
            message (Text): The text message to send.
            **kwargs (Any): Additional arguments (optional).

        Returns:
            None
        """
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        data = {
            "messaging_product": "whatsapp",
            "to": recipient_id,
            "type": "text",
            "text": {
                "body": message
            }
        }
        response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
        if response.status_code != 200:
            logger.error(f"Failed to send message: {response.text}")
            raise Exception(f"WhatsApp API error: {response.status_code} - {response.text}")

    async def send_batch_messages(self, recipient_messages: List[Dict[Text, Any]]) -> None:
        """Sends a batch of messages to multiple recipients.

        Args:
            recipient_messages (List[Dict[Text, Any]]): A list of dictionaries containing
                recipient IDs and their corresponding messages.

        Returns:
            None
        """
        for entry in recipient_messages:
            recipient_id = entry["recipient_id"]
            message = entry["message"]
            await self.send_text_message(recipient_id, message)

    async def send_custom_json(self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any) -> None:
        """Sends a custom JSON message to a recipient.

        Args:
            recipient_id (Text): The WhatsApp ID of the message recipient.
            json_message (Dict[Text, Any]): The custom JSON payload to send.
            **kwargs (Any): Additional arguments (optional).

        Returns:
            None
        """
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

        json_message.setdefault("messaging_product", "whatsapp")
        json_message.setdefault("recipient_type", "individual")
        json_message.setdefault("to", recipient_id)

        response = requests.post(self.api_url, headers=headers, data=json.dumps(json_message))
        logger.debug(f"WhatsApp API response: {response.status_code}, {response.text}")

# TODO: Implement the WhatsAppInput class for incoming messages from the Whatsapp Cloud API

# class WhatsAppInput(InputChannel):
#     """Custom input channel for WhatsApp Cloud API.
#
#     This class defines a custom input channel for receiving messages from the
#     WhatsApp Cloud API. It handles verification and message receipt via
#     webhooks.
#
#     Attributes:
#         access_token (Text): The access token for authenticating with the WhatsApp API.
#         phone_number_id (Text): The ID of the phone number associated with the WhatsApp account.
#     """
#
#     def __init__(self, access_token: Text, phone_number_id: Text) -> None:
#         """
#         Args:
#             access_token (Text): The access token for the WhatsApp Cloud API.
#             phone_number_id (Text): The ID of the phone number used for sending messages.
#         """
#         self.access_token = access_token
#         self.phone_number_id = phone_number_id
#
#     @classmethod
#     def name(cls) -> Text:
#         """Returns the name of the input channel.
#
#         Returns:
#             Text: The name of the input channel.
#         """
#         return "whatsapp_cloud"
#
#     @classmethod
#     def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
#         """Creates an instance of the input channel from credentials.
#
#         Args:
#             credentials (Optional[Dict[Text, Any]]): The credentials for the WhatsApp API.
#
#         Returns:
#             InputChannel: An instance of the WhatsAppInput channel.
#         """
#         if not credentials:
#             cls.raise_missing_credentials_exception()
#
#         return cls(
#             access_token=credentials.get("access_token"),
#             phone_number_id=credentials.get("phone_number_id")
#         )
#
#     def blueprint(self, on_new_message):
#         """Defines the Sanic blueprint for handling WhatsApp webhook requests.
#
#         Args:
#             on_new_message: The callback function for handling new messages.
#
#         Returns:
#             Blueprint: A Sanic blueprint for the WhatsApp webhook.
#         """
#         whatsapp_webhook = Blueprint("whatsapp_webhook", __name__)
#
#         @whatsapp_webhook.route("/", methods=["GET"])
#         async def verify(request: Request) -> Any:
#             """Handles the verification of the webhook.
#
#             Args:
#                 request (Request): The HTTP request.
#
#             Returns:
#                 Any: A response object with the verification result.
#             """
#             mode = request.args.get("hub.mode")
#             token = request.args.get("hub.verify_token")
#             challenge = request.args.get("hub.challenge")
#             verify_token = "0c4b2d1ef9a8cade887f5acd2915c761b601dc87ec5093677566bcf5cdceea79"
#
#             if mode == "subscribe" and token == verify_token:
#                 return text(challenge)
#             else:
#                 return text("Verification token mismatch", status=403)
#
#         @whatsapp_webhook.route("/", methods=["POST"])
#         async def receive(request: Request) -> Any:
#             """Handles incoming messages from the WhatsApp Cloud API.
#
#             Args:
#                 request (Request): The HTTP request containing the message payload.
#
#             Returns:
#                 Any: A JSON response indicating the status of the received message.
#             """
#             payload = request.json
#             logger.debug(f"Received WhatsApp payload: {payload}")
#
#             if "messages" in payload["entry"][0]["changes"][0]["value"]:
#                 for message in payload["entry"][0]["changes"][0]["value"]["messages"]:
#                     sender_id = message["from"]
#
#                     text = ""
#                     # Determine message type
#                     message_type = message.get("type")
#
#                     # Handle text messages
#                     if message_type == "text":
#                         text = message.get("text", {}).get("body", "")
#
#                     # Handle interactive messages (e.g., button replies)
#                     elif message_type == "interactive":
#                         # Process button replies
#                         interactive = message.get("interactive", {})
#                         if interactive.get("button_reply", {}):
#                             button_reply = interactive.get("button_reply", {})
#                             button_id = button_reply.get("id")
#                             button_title = button_reply.get("title")
#                             # Optionally, map button IDs to specific actions or texts
#                             text = f"{button_title}"
#                         elif interactive.get("list_reply", {}):
#                             list_reply = interactive.get("list_reply", {})
#                             list_id = list_reply.get("id")
#                             list_title = list_reply.get("title")
#                             text = f"{list_title[1:]}"
#
#                     # Add handling for other message types as needed
#                     # For example: images, documents, etc.
#                     else:
#                         logger.debug(f"Unhandled message type: {message_type}")
#
#                     await on_new_message(
#                         UserMessage(text, WhatsAppOutput(self.access_token, self.phone_number_id), sender_id)
#                     )
#
#             return response.json({"status": "received"})
#
#         return whatsapp_webhook