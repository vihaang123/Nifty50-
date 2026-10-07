from typing import Optional

from fastapi import APIRouter, Depends, Query

from api import services
from api.dependencies import data_source
from api.source import DataSource
from api.schemas import ErrorResponse, LdaResponse, PcaResponse

router = APIRouter(prefix="/analysis", tags=["analysis"])
_errors = {400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}


@router.get("/pca", response_model=PcaResponse, summary="PCA analysis", responses=_errors)
def pca(
    components: Optional[int] = Query(None, ge=1, le=13, description="Number of components (default from config.yaml, 5)"),
    max_points: int = Query(2000, ge=1, le=50000, description="Size of the reproducible sample of stock-day scores returned"),
    source: DataSource = Depends(data_source),
) -> dict:
    """Explained variance, loadings and stock-day scores from PCA on the 13 market features.

    Structured numbers only: the frontend draws the charts. Fitted on the full dataset (exploratory).
    """
    return services.pca_analysis(source, components, max_points)


@router.get("/lda", response_model=LdaResponse, summary="LDA analysis", responses=_errors)
def lda(
    max_points: int = Query(2000, ge=1, le=50000, description="Size of the reproducible sample of stock-day points returned"),
    source: DataSource = Depends(data_source),
) -> dict:
    """Class counts, training accuracy, discriminant coefficients and stock-day LD1/LD2 points.

    The Defensive / Balanced / Aggressive classes are constructed from rankings, so the accuracy describes how well the
    features reproduce those rules, not predictive power.
    """
    return services.lda_analysis(source, max_points)
