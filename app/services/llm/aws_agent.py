# app/services/llm/aws_agent.py

from __future__ import annotations
import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Literal, Tuple
from functools import wraps

import aioboto3
from botocore.exceptions import ClientError, ReadTimeoutError, EndpointConnectionError, ConnectionClosedError
from botocore.config import Config

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger("llm")

# Configuration
MAX_CONCURRENCY = getattr(settings, "AWS_MAX_CONCURRENCY", 32)
RETRY_ATTEMPTS = getattr(settings, "AWS_RETRY_ATTEMPTS", 3)
RETRY_BASE_DELAY = getattr(settings, "AWS_RETRY_BASE_DELAY", 0.5)

CLIENT_CONFIG = Config(
    read_timeout=120,
    connect_timeout=10,
    max_pool_connections=max(32, MAX_CONCURRENCY * 2),
    retries={"max_attempts": RETRY_ATTEMPTS, "mode": "standard"},
    tcp_keepalive=True,
)


class AWSInference:
    """
    Singleton AWS Bedrock inference client with async initialization.
    
    Usage:
        # Get the singleton instance
        client = await AWSInference.get_instance()
        
        # Use it
        result = await client.infer(...)
        
        # Cleanup (in app shutdown)
        await AWSInference.shutdown()
    """
    
    # Class-level singleton state
    _instance: Optional["AWSInference"] = None
    _instance_lock: asyncio.Lock = None  # Created lazily
    _initialization_lock: asyncio.Lock = None  # Separate lock for client init
    
    # Class-level concurrency control
    _global_sem: asyncio.Semaphore = None  # Created lazily
    
    def __init__(self):
        """
        Private constructor. Do not call directly.
        Use AWSInference.get_instance() instead.
        """
        if AWSInference._instance is not None:
            raise RuntimeError(
                "AWSInference is a singleton. Use await AWSInference.get_instance() instead."
            )
        
        # Instance state
        self._client = None
        self._client_ctx = None
        self._is_initialized = False
        
        # Session (one per singleton)
        self.session = aioboto3.Session(
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
            region_name=settings.AWS_REGION,
        )
        
        logger.info("AWSInference singleton instance created")
    
    @classmethod
    def _ensure_locks(cls):
        """Ensure class-level locks exist (thread-safe for sync initialization)."""
        if cls._instance_lock is None:
            cls._instance_lock = asyncio.Lock()
        if cls._initialization_lock is None:
            cls._initialization_lock = asyncio.Lock()
        if cls._global_sem is None:
            cls._global_sem = asyncio.Semaphore(MAX_CONCURRENCY)
    
    @classmethod
    async def get_instance(cls) -> "AWSInference":
        """
        Get or create the singleton instance.
        
        This is the ONLY way to obtain an AWSInference instance.
        Thread-safe and async-safe.
        
        Returns:
            The singleton AWSInference instance (initialized and ready to use)
        """
        cls._ensure_locks()
        
        if cls._instance is None:
            async with cls._instance_lock:
                if cls._instance is None:
                    # Create instance
                    instance = object.__new__(cls)  # Bypass __init__ check
                    instance.__dict__.update({
                        '_client': None,
                        '_client_ctx': None,
                        '_is_initialized': False,
                        'session': aioboto3.Session(
                            aws_access_key_id=settings.AWS_ACCESS_KEY,
                            aws_secret_access_key=settings.AWS_SECRET_KEY,
                            region_name=settings.AWS_REGION,
                        )
                    })
                    cls._instance = instance
                    logger.info("AWSInference singleton created")
        
        # Ensure client is initialized
        await cls._instance._ensure_initialized()
        return cls._instance
    
    async def _ensure_initialized(self):
        """
        Ensure the AWS Bedrock client is initialized.
        Safe to call multiple times (idempotent).
        """
        if self._is_initialized:
            return
        
        async with AWSInference._initialization_lock:
            if self._is_initialized:
                return
            
            try:
                logger.info("Initializing AWS Bedrock client...")
                self._client_ctx = self.session.client(
                    "bedrock-runtime",
                    region_name="us-west-2",
                    config=CLIENT_CONFIG,
                )
                self._client = await self._client_ctx.__aenter__()
                self._is_initialized = True
                logger.info("AWS Bedrock client initialized successfully")
            except Exception as e:
                logger.exception("Failed to initialize AWS Bedrock client: %s", e)
                raise RuntimeError(f"AWS client initialization failed: {e}") from e
    
    async def _ensure_client_ready(self):
        """
        Verify client is ready for use. Called before each inference.
        """
        if not self._is_initialized or self._client is None:
            raise RuntimeError(
                "AWS client not initialized. This should not happen. "
                "Please report this bug."
            )
    
    @classmethod
    async def shutdown(cls):
        """
        Gracefully shutdown the singleton instance.
        Should be called during application shutdown.
        """
        cls._ensure_locks()
        
        async with cls._instance_lock:
            if cls._instance is not None:
                instance = cls._instance
                if instance._client and instance._client_ctx:
                    try:
                        logger.info("Closing AWS Bedrock client...")
                        await instance._client_ctx.__aexit__(None, None, None)
                        instance._client = None
                        instance._client_ctx = None
                        instance._is_initialized = False
                        logger.info("AWS Bedrock client closed successfully")
                    except Exception as e:
                        logger.exception("Error closing AWS client: %s", e)
                
                cls._instance = None
                logger.info("AWSInference singleton shutdown complete")
    
    @classmethod
    def reset_for_testing(cls):
        """
        Reset singleton state for testing purposes.
        WARNING: Only use in tests!
        """
        cls._instance = None
        cls._instance_lock = None
        cls._initialization_lock = None
        cls._global_sem = None
        logger.warning("AWSInference singleton reset (testing mode)")
    
    # ==================== Helper Methods ====================
    
    async def _with_retries(self, fn, *args, **kwargs):
        """Retry wrapper with exponential backoff for transient errors."""
        import random
        
        attempt = 0
        while True:
            try:
                return await fn(*args, **kwargs)
            except asyncio.CancelledError:
                raise
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in {
                    "ThrottlingException",
                    "TooManyRequestsException",
                    "ProvisionedThroughputExceededException"
                }:
                    if attempt >= RETRY_ATTEMPTS - 1:
                        logger.error(
                            "Max retries (%d) exceeded for throttling error: %s",
                            RETRY_ATTEMPTS, code
                        )
                        raise
                else:
                    raise
            except (ReadTimeoutError, EndpointConnectionError, ConnectionClosedError) as e:
                if attempt >= RETRY_ATTEMPTS - 1:
                    logger.error("Max retries exceeded for network error: %s", type(e).__name__)
                    raise
            except Exception:
                raise
            
            # Exponential backoff with jitter
            delay = min(60, RETRY_BASE_DELAY * (2 ** attempt))
            jitter = random.uniform(0, delay * 0.1)
            await asyncio.sleep(delay + jitter)
            attempt += 1
            logger.warning("Retrying AWS request (attempt %d/%d)...", attempt + 1, RETRY_ATTEMPTS)
    
    @staticmethod
    def _parse_response(response: dict) -> tuple[str, str]:
        """Parse AWS Bedrock response to extract output text and reasoning."""
        parsed_text: list = (
            response.get("output", {})
            .get("message", {})
            .get("content", [{}])
        )
        reasoning: str = (
            parsed_text[0].get("reasoningContent", {})
            .get("reasoningText", {})
            .get("text", "")
        )
        output_text: str = parsed_text[-1].get("text", "")
        return (output_text, reasoning)
    
    @staticmethod
    def extract_token_details(response: dict[str, dict]) -> dict[str, int]:
        """Extract token usage from AWS response."""
        usage = response.get('usage', {})
        if usage is None:
            return {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}
        
        return {
            "input_tokens": usage.get("inputTokens", 0),
            "cached_tokens": usage.get("cachedTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
        }
    
    def _build_bedrock_prompt(
        self, messages: Any
    ) -> Tuple[Optional[list[dict[str, str]]], list[dict[str, Any]]]:
        """Normalize incoming prompts into Bedrock's system/messages structure."""
        
        def to_content_blocks(content: Any) -> list[dict[str, str]]:
            if isinstance(content, list):
                blocks: list[dict[str, str]] = []
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        blocks.append({"text": str(block["text"])})
                    else:
                        blocks.append({"text": str(block)})
                return blocks or [{"text": ""}]
            return [{"text": str(content)}]
        
        raw_messages: list[Any]
        if isinstance(messages, list):
            raw_messages = messages
        elif isinstance(messages, dict):
            if "role" in messages:
                raw_messages = [messages]
            else:
                raw_messages = []
                system_prompt = messages.get("system")
                if system_prompt is not None:
                    raw_messages.append({"role": "system", "content": system_prompt})
                user_prompt = (
                    messages.get("user") or 
                    messages.get("prompt") or 
                    messages.get("content")
                )
                if user_prompt is not None:
                    raw_messages.append({"role": "user", "content": user_prompt})
        else:
            raw_messages = [{"role": "user", "content": str(messages)}]
        
        normalized_messages: list[dict[str, Any]] = []
        for item in raw_messages:
            if isinstance(item, dict):
                role = item.get("role", "user")
                normalized_messages.append({
                    "role": role,
                    "content": to_content_blocks(item.get("content", "")),
                })
            else:
                normalized_messages.append({
                    "role": "user",
                    "content": to_content_blocks(item),
                })
        
        if not normalized_messages:
            raise ValueError("No messages provided for inference")
        
        system_blocks: Optional[list[dict[str, str]]] = None
        conversation = normalized_messages
        if normalized_messages[0]["role"] == "system":
            system_blocks = normalized_messages[0]["content"]
            conversation = normalized_messages[1:]
        
        return system_blocks, conversation
    
    # ==================== Public Inference Methods ====================
    
    async def infer(
        self,
        key: str,
        messages: list[dict],
        request_id: str,
        model: str = None,
        temperature: float = 0.0,
        json_mode: bool = False,
        reasoning_effort: Literal['low', 'medium'] = 'low',
        num_predict: int = 8 * 1024
    ) -> tuple[str, str, dict[str, Any]]:
        """
        Run inference using AWS Bedrock.
        
        Args:
            key: Operation identifier for logging
            messages: List of message dicts with 'role' and 'content'
            request_id: Request tracking ID
            model: Model ID (defaults to settings.AWS_MODEL_NAME)
            temperature: Sampling temperature
            json_mode: Enable JSON output mode
            reasoning_effort: Reasoning effort level
            num_predict: Max tokens to generate
        
        Returns:
            Tuple of (output_text, reasoning, tokens_detail)
        """
        await self._ensure_client_ready()
        
        if model is None:
            model = settings.AWS_MODEL_NAME
        
        # Build message format
        new_messages = [
            {**el, "content": [{"text": el["content"]}]} 
            for el in messages
        ]
        system_message = new_messages[0]["content"]
        conversation = new_messages[1:]
        
        async with self._global_sem:
            start_time = time.perf_counter()
            try:
                async def _call() -> dict[str, Any]:
                    return await self._client.converse(
                        modelId=model,
                        system=system_message,
                        messages=conversation,
                        inferenceConfig={
                            "maxTokens": num_predict,
                            "temperature": temperature,
                            "topP": 1,
                        },
                        additionalModelRequestFields={
                            "reasoning_effort": reasoning_effort,
                            "seed": 42
                        }
                    )
                
                response = await self._with_retries(_call)
                tokens_detail = self.extract_token_details(response)
                output_text, reasoning = self._parse_response(response)
                
                duration = time.perf_counter() - start_time
                logger.debug(
                    "Inference complete: key=%s, time=%.3fs, tokens=%d, rate=%.1f tok/s",
                    key, duration,
                    tokens_detail.get("output_tokens", 0),
                    tokens_detail.get("output_tokens", 0) / duration if duration > 0 else 0
                )
                
                return output_text, reasoning, tokens_detail
                
            except (ReadTimeoutError, EndpointConnectionError, ConnectionClosedError, ClientError) as e:
                logger.exception("Bedrock call failed for key=%s req=%s: %s", key, request_id, e)
                raise
            except Exception as e:
                logger.exception("Unexpected inference failure key=%s req=%s: %s", key, request_id, e)
                raise
    
    async def infer_qwen(
        self,
        key: str,
        messages,
        request_id: str,
        model: str = None,
        temperature: float = 0,
        json_mode: bool = False,
        reasoning_effort: Literal['low', 'medium'] = 'low',
        num_predict: int = 8 * 1024
    ) -> tuple[str, str, dict[str, Any]]:
        """
        Run inference using QWEN model.
        
        Similar to infer() but uses Bedrock prompt building.
        """
        await self._ensure_client_ready()
        
        if model is None:
            model = settings.AWS_MODEL_NAME_QWEN
        
        system_message, conversation = self._build_bedrock_prompt(messages)
        
        async with self._global_sem:
            start_time = time.perf_counter()
            try:
                converse_kwargs: dict[str, Any] = {
                    "modelId": model,
                    "messages": conversation,
                    "inferenceConfig": {
                        "maxTokens": num_predict,
                        "temperature": temperature,
                        "topP": 1,
                    },
                    "additionalModelRequestFields": {
                        "reasoning_effort": reasoning_effort,
                        "seed": 42
                    }
                }
                if system_message:
                    converse_kwargs["system"] = system_message
                
                response: dict[str, Any] = await self._with_retries(
                    self._client.converse, **converse_kwargs
                )
                
                tokens_detail = self.extract_token_details(response)
                output_text, reasoning = self._parse_response(response)
                
                duration = time.perf_counter() - start_time
                logger.debug(
                    "QWEN inference complete: key=%s, time=%.3fs, tokens=%d",
                    key, duration, tokens_detail.get("output_tokens", 0)
                )
                
                return output_text, reasoning, tokens_detail
                
            except Exception as e:
                logger.exception("QWEN inference failed for key=%s: %s", key, e)
                return "", "", {}


# ==================== Public Factory Function ====================

async def get_inference_instance() -> AWSInference:
    """
    Get the singleton AWS Inference instance.
    
    This is a convenience wrapper around AWSInference.get_instance().
    
    Returns:
        The initialized singleton instance
    """
    return await AWSInference.get_instance()


# ==================== Backward Compatibility ====================

# For code that uses the old synchronous get_inference_instance()
_legacy_instance: Optional[AWSInference] = None

def get_inference_instance_sync() -> AWSInference:
    """
    DEPRECATED: Synchronous accessor for backward compatibility.
    
    Returns an uninitialized instance. You MUST call await instance.init_client()
    before using it.
    
    Use await get_inference_instance() instead for new code.
    """
    global _legacy_instance
    if _legacy_instance is None:
        import warnings
        warnings.warn(
            "get_inference_instance_sync() is deprecated. "
            "Use 'await get_inference_instance()' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # This creates the instance but doesn't initialize the client
        # Client initialization must be done async in lifespan
        _legacy_instance = object.__new__(AWSInference)
        _legacy_instance.__dict__.update({
            '_client': None,
            '_client_ctx': None,
            '_is_initialized': False,
            'session': aioboto3.Session(
                aws_access_key_id=settings.AWS_ACCESS_KEY,
                aws_secret_access_key=settings.AWS_SECRET_KEY,
                region_name=settings.AWS_REGION,
            )
        })
    return _legacy_instance


a = {"success": true, "content": ".\n* 4. What is the difference between type 1 diabetes and type 2 diabetes? Compare and contrast the two.\n* 5. What are the 5 most common complications of diabetes that can affect the feet?\n* 6. How does diabetes affect the feet?\n* 7. What are the three main causes of foot problems in diabetes?\n* 8. What is the most common cause of foot ulcers in people with diabetes?\n* 9. What is the best way to manage diabetes and prevent foot problems?\n* 10. What should you do if you have diabetes and notice any changes in your feet?\n## Chapter 1 Introduction to Diabetes and Foot Care\n1. Diabetes is a group of metabolic disorders characterized by high blood sugar levels, which can lead to a variety of health complications, including foot problems.\n2. Symptoms of diabetes include increased thirst and hunger, frequent urination, fatigue, blurred vision, slow healing of cuts and wounds, tingling or numbness in the hands and feet, recurring skin, gum, or bladder infections, and fruity odor on the breath.\n3. Type 1 diabetes is an autoimmune disease in which the body's immune system attacks the insulin-producing beta cells in the pancreas, resulting in a lack of insulin production. Type 2 diabetes is a metabolic disorder in which the body becomes resistant to insulin, leading to high blood sugar levels.\n4. The two types of diabetes have different causes and characteristics, but share similar symptoms and complications. Type 1 diabetes typically develops in childhood or adolescence, while type 2 diabetes develops in adulthood. Type 1 diabetes requires insulin therapy, while type 2 diabetes may be managed with lifestyle changes and/or medication.\n5. The 5 most common complications of diabetes that can affect the feet are: * Neuropathy (nerve damage) * Peripheral artery disease (poor circulation) * Foot ulcers * Infections * Amputations\n6. Diabetes can affect the feet in several ways, including causing nerve damage, reducing blood flow, and increasing the risk of infection.\n7. The three main causes of foot problems in diabetes are: * Neuropathy (nerve damage) * Poor circulation * High blood sugar levels\n8. The most common cause of foot ulcers in people with diabetes is neuropathy, which can cause loss of sensation in the feet, making it difficult to detect injuries or pressure points.\n9. The best way to manage diabetes and prevent foot problems is to maintain good blood sugar control, practice good", "finish_reason": "length", "model": "us.meta.llama3-3-70b-instruct-v1:0", "token_details": {"input_tokens": 6, "output_tokens": 512, "total_tokens": 518}, "provider": "bedrock"}