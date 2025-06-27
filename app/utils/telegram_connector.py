#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
import httpx
from typing import Text, Dict, Any, Callable, List


class TelegramOutput:
    """Handles message sending via Telegram Bot API."""

    def __init__(self, bot_token: Text, chat_id: Text) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
        self.client = httpx.AsyncClient()

    async def send_text_message(self, message: Text, **kwargs: Any) -> httpx.Response:
        payload = {"chat_id": self.chat_id, "text": message, **kwargs}
        try:
            response = await self.client.post(f"{self.api_url}/sendMessage", json=payload)
            response.raise_for_status()
            return response
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Telegram API request error: {str(e)}")


    async def send_custom_json(self, json_message: Dict[Text, Any]) -> None:
        try:
            response = await self.client.post(f"{self.api_url}/sendMessage", json=json_message)
            response.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Telegram API request error: {str(e)}")

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

    async def close(self):
        await self.client.aclose()


class TelegramInput:
    """Asynchronous input channel for receiving Telegram webhook events and routing to LLM."""

    def __init__(self, bot_token: Text, agent_handler: Callable[[Text, Text], str]) -> None:
        self.output = TelegramOutput(bot_token)
        self.agent_handler = agent_handler

    async def process_update(self, update: Dict[str, Any]) -> Dict[str, Any]:
        message = update.get("message")
        if message:
            sender_id = str(message["chat"]["id"])
            user_input = message.get("text", "")
            response = self.agent_handler(sender_id, user_input)
            await self.output.send_text_message(sender_id, response)
        return {"status": "ok"}