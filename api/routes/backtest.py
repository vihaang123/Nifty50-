from fastapi import APIRouter, Depends

from api import services
from api.dependencies import config_path
from api.schemas import BacktestRequest, BacktestResponse, ErrorResponse

router = APIRouter(prefix="/backtest", tags=["backtest"])


@router.post("", response_model=BacktestResponse, summary="Run a walk-forward backtest",
             responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def backtest(request: BacktestRequest, path: str = Depends(config_path)) -> dict:
    """Runs the Phase 7 walk-forward backtest: at every rebalance the models are refitted using only earlier data.

    Equal weights, zero transaction costs, zero slippage, risk-free rate 0. The benchmark is the market index in the data
    (synthetic for the development data). A full run takes roughly 15 seconds. Results from synthetic data are pipeline
    validation only.
    """
    payload = request.model_dump()
    for key in ("start_date", "end_date"):
        if payload[key] is not None:
            payload[key] = payload[key].isoformat()
    return services.run_backtest_response(path, payload)
