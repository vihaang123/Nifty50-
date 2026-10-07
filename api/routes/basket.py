from fastapi import APIRouter, Depends

from api import services
from api.dependencies import data_source
from api.source import DataSource
from api.schemas import BasketRequest, BasketResponse, ErrorResponse

router = APIRouter(prefix="/basket", tags=["basket"])


@router.post("/generate", response_model=BasketResponse, summary="Generate a basket",
             responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
def generate(request: BasketRequest, source: DataSource = Depends(data_source)) -> dict:
    """Builds the equal-weight, multi-cap, behaviourally diversified basket with the project's Phase 6 generator.

    Invalid values (capital <= 0, basket_size < 3, threshold outside -1..1) return 422. A basket larger than the number of
    available stocks returns 400. Exploratory only: not tested against any benchmark, not an investment recommendation.
    """
    return services.generate_basket_response(source, request.capital, request.basket_size, request.similarity_threshold)
