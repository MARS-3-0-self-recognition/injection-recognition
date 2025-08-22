from __future__ import annotations

from inspect_ai.model._providers.hf import HuggingFaceAPI
from inspect_ai.model._chat_message import ChatMessage
from inspect_ai.tool import ToolInfo


class HFPrefillAPI(HuggingFaceAPI):
    def hf_chat(self, messages: list[ChatMessage], tools: list[ToolInfo]) -> str:
        chat = super().hf_chat(messages, tools)
        if messages and getattr(messages[-1], "role", None) == "assistant":
            last = messages[-1].content
            if isinstance(last, str) and last:
                to_remove = chat.split(last)[-1]
                chat = chat.strip(to_remove)
        return chat


