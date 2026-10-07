from fastapi import APIRouter, Depends

from api import services
from api.dependencies import config_path
from api.schemas import DatasetResponse, ErrorResponse, UniverseResponse

router = APIRouter(tags=["data"])


@router.get("/dataset", response_model=DatasetResponse, summary="Dataset information",
            responses={500: {"model": ErrorResponse}})
def dataset(path: str = Depends(config_path)) -> dict:
    """Describes the data the engine is running on (read from the actual files, nothing hard-coded).

    `is_synthetic` is **true** while the project uses generated development data instead of real market prices.
    """
    return services.dataset_info(path)


@router.get("/universe", response_model=UniverseResponse, summary="Stock universe",
            responses={500: {"model": ErrorResponse}})
def universe(path: str = Depends(config_path)) -> dict:
    """The development universe: each stock and its cap category (Large, Mid or Small Cap).

    The cap categories are synthetic development assignments, not real market-cap classifications.
    """
    return services.universe_info(path)
