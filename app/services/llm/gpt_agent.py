
import json
import logging
import traceback
from datetime import datetime
import time
from typing import Literal

from openai import AsyncOpenAI, OpenAIError  # pip install openai>=1.0
from app.core.config import get_settings
from app.utils.post_processing import PostProcessing
settings = get_settings()
logger = logging.getLogger("llm")

post_processor = PostProcessing()


class LLMAgent:
    """
    LLMAgent formats input prompts and invokes ChatGPT via OpenAI's async API client.

    Attributes:
        model_name (str): OpenAI model ID configured in environment settings.
        client (AsyncOpenAI): Async client for OpenAI API calls.
    """

    def __init__(self, region: str = "us-west-2", model: str = None):
        """
        Initialize the LLMAgent with credentials, model, and async OpenAI client.

        Args:
            region (str): Unused here; kept for signature compatibility.
            model (str): OpenAI model name. Defaults to settings.OPENAI_MODEL_NAME or 'gpt-4o-mini'.
        """
        # Preserve the attribute names used elsewhere in your code
        self.model_name =  model or getattr(settings, "OPENAI_MODEL", "gpt-4o-mini")
        self.region = region  # kept for compatibility with existing call sites
        self.client = AsyncOpenAI(api_key=getattr(settings, "OPENAI_API_KEY", None))
    

    def _build_payload(
        self,
        messages,
        max_gen_len: int,
        temperature: float,
    ) -> dict:
        """
        Build the request payload according to chat-style schema expected by OpenAI.
        - Accepts either list[{"role","content"}] or a raw string (wrapped as a user message).
        - Adds a strict JSON system instruction identical to your Bedrock version when applicable.
        """
        chat_messages = []

        # Keep the same "strict JSON" system rail to minimize downstream changes
        chat_messages.append(
            {
                "role": "system",
                "content": (
                    "You are a strict JSON generator. "
                    "Return ONLY valid JSON. Do NOT include any reasoning, explanations, markdown, or commentary. "
                    "Do NOT use <reasoning> or code blocks."
                ),
            }
        )

        if isinstance(messages, list):
            for msg in messages:
                role = (msg.get("role", "user") or "user").lower()
                content = msg.get("content", "")
                if role not in ("system", "user", "assistant"):
                    role = "user"
                chat_messages.append({"role": role, "content": content})
        else:
            chat_messages.append({"role": "user", "content": str(messages)})

        # OpenAI Chat Completions params
        native_request = {
            "model": self.model_name,
            "messages": chat_messages,
            "temperature": temperature,
            "top_p": 1.0,
            "max_tokens": max_gen_len,
            "stream": False,
        }
        return native_request

    @staticmethod
    def _parse_response(response_obj) -> str:
        """
        Parse OpenAI Responses API response.

        Collects all assistant message text parts (newline-joined), preserving your
        old behavior of stripping anything before a hidden </reasoning> tag.
        Works with both SDK model objects and raw dicts. Returns "" if nothing found.
        """
        def _strip_reasoning(s: str) -> str:
            return s.split("</reasoning>")[-1] if s else s

        # --- Preferred path: SDK object access ---
        try:
            output = getattr(response_obj, "output", None) or []
            texts = []
            for item in output:
                # We only care about assistant "message" items
                item_type = getattr(item, "type", None) or getattr(item, "object", None)
                if item_type == "message":
                    parts = getattr(item, "content", None) or []
                    for part in parts:
                        # Collect only textual content parts
                        text = getattr(part, "text", None)
                        if text:
                            texts.append(_strip_reasoning(text))
            if texts:
                return "\n".join(t.strip() for t in texts if t).strip()
        except Exception:
            pass

        # --- Fallback: raw dict-style access ---
        try:
            if isinstance(response_obj, dict):
                output = response_obj.get("output", []) or []
                texts = []
                for item in output:
                    if item.get("type") == "message":
                        for part in (item.get("content") or []):
                            text = part.get("text")
                            if text:
                                texts.append(_strip_reasoning(text))
                if texts:
                    return "\n".join(t.strip() for t in texts if t).strip()
        except Exception:
            pass

        # --- Final fallbacks: some SDKs expose an aggregated text field ---
        try:
            agg = getattr(response_obj, "output_text", None)
            if isinstance(agg, str) and agg:
                return _strip_reasoning(agg).strip()
        except Exception:
            pass
        try:
            if isinstance(response_obj, dict):
                agg = response_obj.get("output_text")
                if isinstance(agg, str) and agg:
                    return _strip_reasoning(agg).strip()
        except Exception:
            pass

        return ""

    # Safely extract token details from a response object or dict
    @staticmethod
    def extract_token_details(response):
        # Try to handle both attribute-style and dict-style access
        usage = getattr(response, "usage", None) or getattr(response, "get", lambda x, d=None: None)("usage", None)
        if usage is None:
            return {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}

        # Handle both dict and attribute structures gracefully
        def safe_get(obj, key, default=0):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        input_tokens = safe_get(usage, "input_tokens", 0)
        input_details = safe_get(usage, "input_tokens_details", {}) or {}
        cached_tokens = safe_get(input_details, "cached_tokens", 0)
        output_tokens = safe_get(usage, "output_tokens", 0)

        return {
            "input_tokens": input_tokens,
            "cached_tokens": cached_tokens,
            "output_tokens": output_tokens,
        }

    async def infer(
        self,
        key,
        messages,
        request_id,
        max_gen_len: int = 8192,
        temperature: float = 0.3,
        reasoning_effort: Literal['minimal', 'low'] = 'minimal'
    ) -> tuple[str, dict[str, int]]:
        """
        Sends a prompt to the ChatGPT model and retrieves the generated response.
        Signature preserved for compatibility with callers.
        """
        request_payload = self._build_payload(messages, max_gen_len, temperature)
        try:
            logger.info(f"Received request for {key} (model={self.model_name})")

            # Call OpenAI Chat Completions API asynchronously
            response = await self.client.responses.create(
                model=request_payload["model"],
                 reasoning={
                    "effort": reasoning_effort,
                    "summary": "auto"
                },
                prompt_cache_key="migration_llm_call",
                input=request_payload["messages"],
                parallel_tool_calls=False, 
                top_p=request_payload["top_p"],
                stream=request_payload["stream"]
            )
            response_out = self._parse_response(response)
            tokens_detail = self.extract_token_details(response)
            logger.info(f"Completed request for {key}")
            return response_out, tokens_detail

        except OpenAIError as e:
            logger.error(
                f"OpenAIError while invoking '{self.model_name}'. Reason: {e}",
                exc_info=True,
            )
            exception = {
                "exception_type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            return "", {}

    async def infer_custom_forms(
            self,
            key,
            messages,
            request_id,
            max_gen_len: int = 16384,
            temperature: float = 0.1,
        ) -> tuple[str, dict[str, int]]:
            """
            Sends a prompt to the ChatGPT model and retrieves the generated response.
            Signature preserved for compatibility with callers.
            """
            request_payload = self._build_payload(messages, max_gen_len, temperature)
            try:
                logger.info(f"Received request for {key} (model={self.model_name})")
                start_time = time.time()
                # Call OpenAI Chat Completions API asynchronously
                response = await self.client.responses.create(
                    model=request_payload["model"],
                    #  reasoning={
                    #     "effort": "low",
                    #     "summary": "auto"
                    # },
                    input=request_payload["messages"],
                    parallel_tool_calls=False, 
                    top_p=request_payload["top_p"],
                )
                end_time = time.time()
                logger.info(f"Request for {key} completed in {end_time - start_time:.2f} seconds")
                response_out = self._parse_response(response)
                tokens_detail = self.extract_token_details(response)

                logger.info(f"Completed request for {key}")
                # print("Response from infer_custom_forms:", response_out)
                return response_out

            except OpenAIError as e:
                logger.error(
                    f"OpenAIError while invoking '{self.model_name}'. Reason: {e}",
                    exc_info=True,
                )
                exception = {
                    "exception_type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
                return "", {}