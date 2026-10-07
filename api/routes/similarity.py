from fastapi import APIRouter, Depends, Query

from api import services
from api.dependencies import data_source
from api.source import DataSource
from api.schemas import ErrorResponse, SimilarityResponse

router = APIRouter(prefix="/similarity", tags=["similarity"])


@router.get("/{symbol}", response_model=SimilarityResponse, summary="Similar stocks",
            responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
def similarity(
    symbol: str,
    top_n: int = Query(5, ge=1, le=50, description="How many similar stocks to return"),
    source: DataSource = Depends(data_source),
) -> dict:
    """The stocks whose average PCA behaviour profile is closest (cosine similarity) to `symbol`. The stock itself is excluded.

    The symbol is case-insensitive. Unknown symbols return 404.
    """
    return services.similar_stocks(source, symbol.strip().upper(), top_n)
