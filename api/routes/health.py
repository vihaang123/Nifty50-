from fastapi import APIRouter

from api import API_VERSION, SERVICE_NAME
from api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health() -> dict:
    """Returns `ok` when the service is up. Used to verify a deployment."""
    return {"status": "ok", "service": SERVICE_NAME, "version": API_VERSION}
