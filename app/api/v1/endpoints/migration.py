# app/api/routes.py
from fastapi import APIRouter
from typing import Dict
from app.core.decorators import log_action

router = APIRouter()

@log_action(action_type="process_migration", log_result=True)
def process_migration() -> Dict[str, str]:
    # nested actions also get logged under the same request
    step_one()
    return {"status": "success", "message": "Migration completed successfully."}

@log_action(action_type="step", action_name="step_one", log_result=False)
def step_one():
    # do something
    return "ok"

@router.post("/migration")
@log_action(action_type="api_endpoint", log_result=True)
def migration():
    # request_id will come from the middleware context; no need to pass Request
    return process_migration()
