from pydantic import BaseModel, Field 
from typing import Optional, List, Dict, Any 
from enum import Enum 


class LLMProvider(str, Enum):
    OPENAI = "openai"
    BEDROCK = "bedrock"
    
class Message(BaseModel):
    role: str = Field(..., description="Message role: system, user or assistant")
    content: str = Field(..., description = "Message content")
    
class ReasoningConfig(BaseModel):
    effort: str = Field("low", description = "Reasoning Effore: low, medium, high")
    summary: str = Field("auto", description = "summary mode: auto, always, never")
    
class LLMRequest(BaseModel):
    provider: LLMProvider = Field(..., description = "LLM Provider to use")
    messages: List[Message] = Field(..., description = "List of messages")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description = "Sampling temperature")
    max_tokens: Optional[int] = Field(None, gt=0, description="maximum tokens to generate")
    model: Optional[str] = Field(None, description = "Specific model to use")
    reasoning: Optional[ReasoningConfig] = Field(
        ReasoningConfig(effort="low", summary = "auto"),
        description = "reasoning configuration"
    )

class TokenDetails(BaseModel):
    input_tokens:int = Field(..., description = "Number of input tokens")
    output_tokens:int = Field(..., description = "Number of input tokens")
    total_tokens:int = Field(..., description = "Total tokens used")
    reasoning_tokens:Optional[int] = Field(..., description = "Tokens used for reasoning")
    cache_read_tokens:Optional[int] = Field(..., description = "Tokens read from cache")
    cache_creation_tokens:Optional[int] = Field(..., description = "Tokens used to create cache")

    
class ProcessingStep(BaseModel):
    step_name: str
    duration_ms: float
    status: str 
    result: Optional[Any] = None 
    error: Optional[str] = None
    
class LLMResponse(BaseModel):
    success: bool 
    request_id: str 
    provider: str 
    model: str 
    content: Optional[str] = None 
    token_details: Optional[Dict[str, int]] = None 
    finish_reason:Optional[str] = None
    procesing_steps: List[ProcessingStep] = []
    total_duration_ms: float 
    error: Optional[str] = None 
    
