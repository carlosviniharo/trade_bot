#!/usr/bin/env python3

"""
Telegram Connector Module
--------------------------

This module provides asynchronous integration with the Telegram Bot API
for sending and receiving messages. Designed for use in FastAPI-based
AI agents, including LLM and chatbot applications.

Classes:
    TelegramOutput: Handles outbound message delivery to Telegram.
    TelegramInput: Manages inbound Telegram webhook payloads and message routing.

"""

from fastapi import HTTPException
import httpx
from typing import Text, Dict, Any, Callable, List, Optional

# TODO: Include webhook support for TelegramInput and TelegramOutput, so there is not need to
#  include the chat ID in the constructors.
class TelegramOutput:
    """
    Asynchronous output channel for Telegram Bot API.

    Attributes:
        bot_token (Text): Bot access token for authentication.
        chat_id (Text): Target Telegram chat ID to send messages to.
        api_url (Text): Preformatted URL for Telegram Bot API.
        client (httpx.AsyncClient): Asynchronous HTTP client for requests.
    """

    def __init__(self, bot_token: Text, chat_id: Text) -> None:
        """
        Initialize TelegramOutput with bot token and chat ID.

        Args:
            bot_token (Text): Bot access token for authentication.
            chat_id (Text): Target Telegram chat ID to send messages to.
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
        self.client = httpx.AsyncClient()

    async def send_text_message(self, message: Text, **kwargs: Any) -> httpx.Response:
        """
        Sends a plain text message to the configured Telegram chat.

        Args:
            message (Text): The message content to be sent.
            **kwargs (Any): Optional additional Telegram API parameters.

        Returns:
            httpx.Response: Response object for logging or inspection.

        Raises:
            HTTPException: If the Telegram API request fails.
        """
        payload = {"chat_id": self.chat_id, "text": message, **kwargs}
        try:
            response = await self.client.post(f"{self.api_url}/sendMessage", json=payload)
            response.raise_for_status()
            return response
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Telegram API request error: {str(e)}")

    async def send_custom_json(self, json_message: Dict[Text, Any]) -> None:
        """
        Sends a raw JSON payload to Telegram API.

        Args:
            json_message (Dict[Text, Any]): Raw Telegram API-compliant payload.

        Returns:
            None

        Raises:
            HTTPException: On any request failure.
        """
        try:
            response = await self.client.post(f"{self.api_url}/sendMessage", json=json_message)
            response.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Telegram API request error: {str(e)}")

    async def send_batch_messages(self, recipient_messages: List[Dict[Text, Any]]) -> None:
        """
        Sends a batch of messages to multiple recipients.

        Args:
            recipient_messages (List[Dict[Text, Any]]): A list of dicts with
                "chat_id" and "message" keys.

        Returns:
            None
        """
        for entry in recipient_messages:
            chat_id = entry["chat_id"]
            message = entry["message"]
            # Create a temporary instance with the specific chat_id
            temp_output = TelegramOutput(self.bot_token, chat_id)
            await temp_output.send_text_message(message)
            await temp_output.close()

    async def close(self) -> None:
        """
        Closes the underlying HTTP client.

        Returns:
            None
        """
        await self.client.aclose()


class TelegramInput:
    """
    Asynchronous input handler for Telegram webhook updates.

    Routes incoming messages to the specified agent handler.

    Attributes:
        output (TelegramOutput): Telegram message sender.
        agent_handler (Callable): Callback that takes sender ID and text message,
            returns a text response.
    """

    def __init__(self, bot_token: Text, agent_handler: Callable[[Text, Text], Text]) -> None:
        """
        Initialize TelegramInput with bot token and agent handler.

        Args:
            bot_token (Text): Bot access token for authentication.
            agent_handler (Callable[[Text, Text], Text]): Callback function that takes
                sender ID and text message, returns a text response.
        """
        self.output = TelegramOutput(bot_token, "")  # Chat ID can be set dynamically
        self.agent_handler = agent_handler

    async def process_update(self, update: Dict[str, Any]) -> Dict[str, str]:
        """
        Processes a single Telegram update payload.

        Args:
            update (Dict[str, Any]): Telegram webhook JSON payload.

        Returns:
            Dict[str, str]: Acknowledgement status.
        """
        message = update.get("message")
        if message:
            sender_id = str(message["chat"]["id"])
            user_input = message.get("text", "")
            response = self.agent_handler(sender_id, user_input)
            # Set the chat_id dynamically for this response
            self.output.chat_id = sender_id
            await self.output.send_text_message(response)
        return {"status": "ok"}
