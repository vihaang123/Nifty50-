"""
FastAPI application.  Run locally with:   uvicorn api.main:app --reload
Docs:  http://localhost:8000/docs  and  http://localhost:8000/redoc
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api import API_VERSION, SERVICE_NAME
from api.errors import ApiError, error_body
from api.routes import analysis, backtest, basket, data, health, similarity
from api.settings import Settings, get_settings

logger = logging.getLogger("stock-basket-api")

DESCRIPTION = """
Backend for the *Learning the Latent Structure of Financial Markets* project (BTech Data Science).

A thin layer over the existing Python research engine: **PCA, LDA, stock similarity, multi-cap basket construction and
walk-forward backtesting**. Every number returned comes from that engine; nothing is computed in the browser.

* The development dataset is **synthetic**: check `is_synthetic` in `/api/dataset`.
* PCA / LDA / similarity / basket are **exploratory** (fitted on the whole dataset). Only `/api/backtest` is walk-forward.
* This is a research project, **not investment advice**.

Errors always look like `{"error": {"status": 404, "code": "unknown_symbol", "message": "..."}}`.
"""

_STATUS_CODES = {400: "bad_request", 404: "not_found", 405: "method_not_allowed", 422: "invalid_request"}


def _response(status: int, code: str, message: str, details: list | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content=error_body(status, code, message, details))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="Stock Basket API",
        description=DESCRIPTION,
        version=API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.frontend_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    for router in (health.router, data.router, analysis.router, similarity.router, basket.router, backtest.router):
        app.include_router(router, prefix="/api")

    @app.exception_handler(ApiError)
    async def handle_api_error(_: Request, error: ApiError):
        return _response(error.status_code, error.code, error.message)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_: Request, error: StarletteHTTPException):
        code = _STATUS_CODES.get(error.status_code, "http_error")
        message = {404: "The requested resource was not found.", 405: "This method is not allowed for that endpoint."}.get(
            error.status_code, "The request could not be completed."
        )
        return _response(error.status_code, code, message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, error: RequestValidationError):
        details = []
        for item in error.errors():
            where = [str(part) for part in item["loc"] if part not in ("body", "query", "path")]
            message = str(item["msg"]).removeprefix("Value error, ")
            details.append({"field": ".".join(where) or "request", "message": message})
        return _response(422, "invalid_request", "Invalid request parameters.", details)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, error: Exception):
        logger.exception("Unhandled error")  # the traceback stays in the server log, never in the response
        return _response(500, "internal_error", "Unexpected internal error. Please try again later.")

    return app


app = create_app()
