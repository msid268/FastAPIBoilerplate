from typing import Optional, Dict, Any
import logging
import json
import asyncio
import random

from aiobotocore.session import get_session
from botocore.exceptions import (
    ClientError,
    ReadTimeoutError,
    EndpointConnectionError,
    ConnectionClosedError,
)

from app.core.config import settings
from app.core.decorators import log_action

logger = logging.getLogger(__name__)

# Single shared session for the entire process
_SESSION = get_session()


class BedrockService:
    """Minimal Bedrock text-generation service (non-streaming)."""

    def __init__(self) -> None:
        self.model_id: str = settings.BEDROCK_MODEL_ID
        self.max_tokens: int = settings.BEDROCK_MAX_TOKENS
        self.region: str = settings.AWS_REGION

        # Retry configuration
        self.retry_attempts: int = getattr(settings, "BEDROCK_RETRY_ATTEMPTS", 3)
        self.retry_base_delay: float = getattr(settings, "BEDROCK_RETRY_BASE_DELAY", 0.5)

    async def _get_client(self):
        """Create a new Bedrock runtime client using the shared session."""
        return _SESSION.create_client(
            "bedrock-runtime",
            region_name=self.region,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )

    async def _with_retries(self, fn, *args, **kwargs) -> Any:
        """
        Retry wrapper with exponential backoff for transient errors.
        Designed for single-shot (non-streaming) calls.
        """
        attempt = 0
        while True:
            try:
                return await fn(*args, **kwargs)

            except asyncio.CancelledError:
                # Don't swallow cancellation
                raise

            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                transient = {
                    "ThrottlingException",
                    "TooManyRequestsException",
                    "ProvisionedThroughputExceededException",
                }
                if code not in transient or attempt >= self.retry_attempts - 1:
                    logger.error("Non-retriable or max-retries ClientError: %s", code)
                    raise

            except (ReadTimeoutError, EndpointConnectionError, ConnectionClosedError) as e:
                if attempt >= self.retry_attempts - 1:
                    logger.error("Network error after retries: %s", type(e).__name__)
                    raise

            except Exception:
                # Unknown / non-transient error – do not retry
                raise

            delay = min(60.0, self.retry_base_delay * (2 ** attempt))
            jitter = random.uniform(0, delay * 0.1)
            sleep_for = delay + jitter
            attempt += 1

            logger.warning(
                "Retrying Bedrock request (attempt %d/%d) in %.2fs",
                attempt,
                self.retry_attempts,
                sleep_for,
            )
            await asyncio.sleep(sleep_for)

    @log_action(action_type="llm_call", log_result=True)
    async def chat_completion(
        self,
        prompt: str,
        *,
        key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Create a text completion using AWS Bedrock.

        This uses a Claude-style schema that expects a `prompt` field, e.g.:

        {
            "prompt": "your text here",
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 1.0
        }

        Adjust keys if your specific model expects a slightly different shape.
        """
        log_key = key or "bedrock_request"
        logger.info("Starting Bedrock request for %s", log_key)

        model_id = model or self.model_id
        if not model_id:
            raise RuntimeError("BEDROCK_MODEL_ID is not configured")

        request_body: Dict[str, Any] = {
            "prompt": prompt,
            # "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        async with await self._get_client() as client:

            async def _invoke():
                return await client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(request_body),
                )

            try:
                response = await self._with_retries(_invoke)

                raw_body = await response["body"].read()
                response_body = json.loads(raw_body)
                print(response_body)
                # Common Claude-on-Bedrock patterns:
                # e.g. {"completion": "...", "stop_reason": "...", "usage": {...}}
                output_text = (
                    response_body.get("generation")
                    or response_body.get("outputText")
                    or ""
                )

                usage = response_body.get("usage", {})
                input_tokens = response_body.get("prompt_token_count", 0)
                output_tokens = response_body.get("generation_token_count", 0)

                token_details = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }

                result = {
                    "success": True,
                    "content": output_text,
                    "finish_reason": response_body.get("stop_reason"),
                    "model": model_id,
                    "token_details": token_details,
                    "provider": "bedrock",
                }

                logger.info("Completed Bedrock request for %s", log_key)
                return result

            except Exception as e:
                logger.error(
                    "Bedrock API call failed for %s: %s", log_key, str(e), exc_info=True
                )
                raise

    @log_action(action_type="llm_call", log_result=True)
    async def generate_text(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Simple alias for chat_completion for convenience.
        """
        return await self.chat_completion(prompt, **kwargs)
