# app/api/routes.py
from fastapi import APIRouter
from typing import Dict
from app.core.decorators import log_action

router = APIRouter()

@log_action(action_type="process_migration", log_result=True)
def process_migration(number) -> Dict[str, str]:
    step_one(number*2)
    return {"status": "success", "message": "Migration completed successfully."}

@log_action(action_type="step", action_name="step_one", log_result=False)
def step_one(number):
    return "ok "* number

@router.post("/migration")
@log_action(action_type="api_endpoint", log_result=True)
def migration():
    return process_migration(2)
