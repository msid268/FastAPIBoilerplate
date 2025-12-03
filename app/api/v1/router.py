from fastapi import APIRouter
import app.api.v1.endpoints.migration as migration
import app.api.v1.logs as logs
# import app.api.v1.endpoints.llm_endpoints as llm_endpoints

api_router = APIRouter()


# api_router.include_router(
#     health.router,
#     tags = ['health']
# )
api_router.include_router(
    logs.router,
    tags = ['logs']
)

api_router.include_router(
    migration.router,
    tags=['migration']
)

# api_router.include_router(
#     llm_endpoints.router,
#     prefix = "/llm",
#     tags=['llm_endpoints']
# )