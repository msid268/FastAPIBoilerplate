# app/api/routes.py
from typing import Dict, Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings
from app.core.decorators import log_action
from app.services.llm.bedrock_service import BedrockService  
router = APIRouter()

bedrock_service = BedrockService()  # simple shared instance


class QueryRequest(BaseModel):
    query: str


@log_action(action_type="process_migration", log_result=True)
async def process_migration(query: str) -> Dict[str, Any]:
    """
    Orchestrates sending the query to AWS Bedrock and returning the response.
    Logged by @log_action.
    """
    # Call AWS Bedrock client
    response = await bedrock_service.chat_completion(
        prompt=query,
        key="migration_query",
        model = settings.BEDROCK_MODEL_ID
    )
    return {
        "status": "success",
        "message": "Migration completed successfully.",
        "llm_response": response,
    }


@router.post("/migration")
@log_action(action_type="api_endpoint", log_result=True)
async def migration(payload: QueryRequest) -> Dict[str, Any]:
    """
    API endpoint that receives a 'query', sends it to Bedrock,
    and returns the result. Fully logged via @log_action.
    """
    return await process_migration(payload.query)
