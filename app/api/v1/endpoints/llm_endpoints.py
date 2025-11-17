from fastapi import APIRouter, Request, HTTPException
import logging
import time
from typing import Dict, Any

from app.schemas.llm_schemas import (
    LLMRequest, LLMResponse, ProcessingStep, LLMProvider
)
from app.services.llm.openai_service import OpenAIService
from app.services.llm.bedrock_service import BedrockService
from app.core.decorators import log_action

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=LLMResponse)
# @log_action(action_type="api_endpoint", log_result=True)
async def chat_completion(
    request: Request,
    llm_request: LLMRequest
):
    """
    Generate chat completion using specified LLM provider.
    
    Supports:
    - OpenAI models with reasoning configuration
    - AWS Bedrock models
    - Prompt caching
    - Top-p sampling
    - Parallel tool calls configuration
    """
    start_time = time.time()
    request_id = request.state.request_id
    
    logger.info(f"Processing chat completion request with {llm_request.provider}")
    
    try:
        # Prepare request payload
        request_payload = {
            "model": llm_request.model,
            "messages": [msg.model_dump() for msg in llm_request.messages],
        }
        
        # Add reasoning config if provided
        if llm_request.reasoning:
            request_payload["reasoning"] = {
                "effort": llm_request.reasoning.effort,
                "summary": llm_request.reasoning.summary
            }
        
        # Select service based on provider
        if llm_request.provider == LLMProvider.OPENAI:
            service = OpenAIService()
            result = await service.chat_completion(
                request_payload=request_payload,
                key=request_id
            )
        elif llm_request.provider == LLMProvider.BEDROCK:
            service = BedrockService()
            result = await service.chat_completion(
                request_payload=request_payload,
                key=request_id
            )
        else:
            raise ValueError(f"Unsupported provider: {llm_request.provider}")
        
        duration = (time.time() - start_time) * 1000
        
        # Build response
        response = LLMResponse(
            success=True,
            request_id=request_id,
            provider=result["provider"],
            model=result["model"],
            content=result["content"],
            token_details=result["token_details"],
            finish_reason=result.get("finish_reason"),
            processing_steps=[
                ProcessingStep(
                    step_name="llm_call",
                    duration_ms=duration,
                    status="success",
                    result={
                        "total_tokens": result["token_details"]["total_tokens"],
                        "input_tokens": result["token_details"]["input_tokens"],
                        "output_tokens": result["token_details"]["output_tokens"],
                        "reasoning_tokens": result["token_details"].get("reasoning_tokens", 0),
                        "cache_read_tokens": result["token_details"].get("cache_read_tokens", 0)
                    }
                )
            ],
            total_duration_ms=duration
        )
        
        logger.info(
            f"LLM call completed successfully. "
            f"Tokens: {result['token_details']['total_tokens']}, "
            f"Duration: {duration:.2f}ms"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Chat completion failed: {str(e)}", exc_info=True)
        duration = (time.time() - start_time) * 1000
        
        return LLMResponse(
            success=False,
            request_id=request_id,
            provider=llm_request.provider.value,
            model=llm_request.model,
            processing_steps=[
                ProcessingStep(
                    step_name="llm_call",
                    duration_ms=duration,
                    status="error",
                    error=str(e)
                )
            ],
            total_duration_ms=duration,
            error=str(e)
        )