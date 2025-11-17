from typing import Optional, List, Dict, Any
import logging
import json
from aiobotocore.session import get_session
from app.core.config import settings
from app.core.decorators import log_action

logger = logging.getLogger(__name__)


class BedrockService:
    """Service for AWS Bedrock API calls using aiobotocore."""
    
    def __init__(self):
        self.model_id = settings.BEDROCK_MODEL_ID
        self.max_tokens = settings.BEDROCK_MAX_TOKENS
        self.region = settings.AWS_REGION
        self.session = get_session()
    
    async def _get_client(self):
        """Get boto3 bedrock-runtime client."""
        return self.session.create_client(
            'bedrock-runtime',
            region_name=self.region,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )
    
    def _format_claude_messages(self, messages: List[Dict[str, str]]) -> tuple:
        """Format messages for Claude API."""
        system_prompt = None
        formatted_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        return system_prompt, formatted_messages
    
    @log_action(
        action_type="llm_call",
        log_result=True
    )
    async def chat_completion(
        self,
        request_payload: Dict[str, Any],
        key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a chat completion using AWS Bedrock.
        
        Args:
            request_payload: Dictionary containing:
                - model: Model ID to use
                - messages: List of message dicts
                - reasoning: Dict with effort and summary settings (optional)
                - top_p: Top-p sampling parameter (optional)
                - stream: Enable streaming (optional)
            key: Optional identifier for logging
        
        Returns:
            Dict containing response, token details, and metadata
        """
        log_key = key or "bedrock_request"
        logger.info(f"Starting request for {log_key}")
        
        model = request_payload.get("model", self.model_id)
        messages = request_payload.get("messages", [])
        top_p = request_payload.get("top_p", 1.0)
        
        try:
            async with await self._get_client() as client:
                # Format messages for Claude
                system_prompt, formatted_messages = self._format_claude_messages(messages)
                
                # Build request body for Claude 3
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.max_tokens,
                    "top_p": top_p,
                    "messages": formatted_messages
                }
                
                if system_prompt:
                    request_body["system"] = system_prompt
                
                # Invoke model
                response = await client.invoke_model(
                    modelId=model,
                    body=json.dumps(request_body)
                )
                
                # Parse response
                response_body = json.loads(await response['body'].read())
                
                # Extract token details
                token_details = {
                    "input_tokens": response_body['usage']['input_tokens'],
                    "output_tokens": response_body['usage']['output_tokens'],
                    "total_tokens": response_body['usage']['input_tokens'] + response_body['usage']['output_tokens'],
                    "reasoning_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_creation_tokens": 0
                }
                
                result = {
                    "success": True,
                    "content": response_body['content'][0]['text'],
                    "finish_reason": response_body.get('stop_reason'),
                    "model": model,
                    "response_id": response_body.get('id'),
                    "token_details": token_details,
                    "provider": "bedrock"
                }
                
                logger.info(f"Completed request for {log_key}")
                
                return result
                
        except Exception as e:
            logger.error(f"Bedrock API call failed for {log_key}: {str(e)}", exc_info=True)
            raise
    
    @log_action(
        action_type="llm_call",
        log_result=True
    )
    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Create a streaming chat completion using AWS Bedrock.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model_id: Model ID to use (defaults to config)
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Bedrock API parameters
        
        Yields:
            Response chunks
        """
        logger.info(f"Making streaming Bedrock API call with {len(messages)} messages")
        
        model = model_id or self.model_id
        
        try:
            async with await self._get_client() as client:
                system_prompt, formatted_messages = self._format_claude_messages(messages)
                
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens or self.max_tokens,
                    "temperature": temperature,
                    "messages": formatted_messages
                }
                
                if system_prompt:
                    request_body["system"] = system_prompt
                
                response = await client.invoke_model_with_response_stream(
                    modelId=model,
                    body=json.dumps(request_body)
                )
                
                async for event in response['body']:
                    chunk = json.loads(event['chunk']['bytes'])
                    
                    if chunk['type'] == 'content_block_delta':
                        if 'delta' in chunk and 'text' in chunk['delta']:
                            yield chunk['delta']['text']
                
                logger.info("Bedrock streaming API call completed")
                
        except Exception as e:
            logger.error(f"Bedrock streaming API call failed: {str(e)}", exc_info=True)
            raise
    
    @log_action(
        action_type="llm_call",
        log_result=True
    )
    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
        
        Returns:
            Dict containing generated text and metadata
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return await self.chat_completion(messages, **kwargs)