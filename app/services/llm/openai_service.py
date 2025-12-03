from typing import Optional, List, Dict, Any 
import logging 
from openai import AsyncOpenAI
from app.core.config import settings 
# from app.core.decorators import log_action 

logger = logging.getLogger(__name__)

class OpenAIService:
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None 
        self.default_model = settings.OPENAI_MODEL
    
    def _parse_response(self, response) -> Dict[str, Any]:
        try: 
            content = ""
            if hasattr(response, 'choices') and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content 
                elif hasattr(choice, 'text'):
                    content = choice.text 
            
            # Extract finish reason 
            finish_reason = None 
            if hasattr(response, 'choices') and len(response.choices) > 0:
                finish_reason = getattr(response.choices[0], 'finish_reason', None)
            return {
                'content': content,
                'finish_reason': finish_reason,
                'model': getattr(response, 'model', None),
                "id": getattr(response, 'id', None)
            }
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}", exc_info=True)
            return {
                'content': "",
                'finish_reason': "error",
                'model': None,
                "id": None
            }
            
    def extract_token_details(self, response) -> Dict[str, int]:
        try:
            token_details = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0
            }
            
            if hasattr(response, 'usage'):
                usage = response.usage 
                
                # standard token counts
                token_details['input_tokens'] = getattr(usage, 'prompt_tokens', 0)
                token_details['output_tokens'] = getattr(usage, 'completion_tokens', 0)
                token_details['total_tokens'] = getattr(usage, 'total_tokens', 0)
                
                # Extended token dtails
                if hasattr(usage, 'completion_tokens_details'):
                    details = usage.completion_tokens_details
                    token_details["reasoning_tokens"] = getattr(details, 'reasoning_tokens', 0)
                
                # Cache details (if available)
                if hasattr(usage, 'prompt_tokens_details'):
                    cache_details = usage.prompt_tokens_details
                    token_details["cache_read_tokens"] = getattr(cache_details, 'cached_tokens', 0)
                
            return token_details
        except Exception as e:
            logger.error(f"Error extracting token details: {str(e)}", exc_info=True)
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0
            }
    
    @log_action(
        action_type = "llm_call",
        log_result = True 
    )
    async def chat_completion(
        self,
        request_payload: Dict[str, Any],
        key: Optional[str] = None
    ):
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        log_key = key or "openai_request"
        logger.info(f"Starting request for {log_key}")
        
        try:
            # Extract parameters from payload
            model = request_payload.get("model", self.default_model)
            messages = request_payload.get("messages", [])
            reasoning = request_payload.get("reasoning", {"effort": "low", "summary": "auto"})
            prompt_cache_key = request_payload.get("prompt_cache_key")
            parallel_tool_calls = request_payload.get("parallel_tool_calls", False)
            top_p = request_payload.get("top_p", 1.0)
            stream = request_payload.get("stream", False)
            
            # Build API call parameters
            api_params = {
                "model": model,
                "messages": messages,
                "top_p": top_p,
                "stream": stream
            }
            
            # Add optional parameters if supported by model
            if reasoning:
                # Note: reasoning parameter support depends on model
                # For models that support it (like o1), add it
                if "o1" in model or "o3" in model:
                    api_params["reasoning_effort"] = reasoning.get("effort", "low")
            
            # Add parallel_tool_calls if tools are provided
            if not parallel_tool_calls and "tools" in request_payload:
                api_params["parallel_tool_calls"] = parallel_tool_calls
            
            # Make API call
            response = await self.client.chat.completions.create(**api_params)
            
            # Parse response
            response_out = self._parse_response(response)
            
            # Extract token details
            tokens_detail = self.extract_token_details(response)
            
            logger.info(f"Completed request for {log_key}")
            
            result = {
                "success": True,
                "content": response_out["content"],
                "finish_reason": response_out["finish_reason"],
                "model": response_out["model"],
                "response_id": response_out["id"],
                "token_details": tokens_detail,
                "provider": "openai"
            }
            
            return result
        except Exception as e:
            logger.error(f"OpenAI API call failed for {log_key}: {str(e)}", exc_info=True)
            raise