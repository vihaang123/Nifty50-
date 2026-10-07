from fastapi import APIRouter, Depends

from api import API_VERSION, SERVICE_NAME, services
from api.dependencies import data_source
from api.schemas import HealthResponse
from api.settings import get_settings
from api.source import DataSource

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health(source: DataSource = Depends(data_source)) -> dict:
    """Reports the API status, the environment and the data provider.

    `status` is `ok` when the API and its data are ready, and `degraded` (still HTTP 200) when the API is up but the data
    provider is misconfigured or its data is missing. The data check is cheap (it does not load the dataset) and never
    exposes file-system paths, environment variables or credentials.
    """
    data = services.data_status(source)
    return {
        "status": "ok" if data["available"] else "degraded",
        "service": SERVICE_NAME,
        "version": API_VERSION,
        "environment": get_settings().environment,
        "data": data,
    }
