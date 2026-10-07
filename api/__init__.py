"""
Phase 8A: the web API (FastAPI) over the existing research engine in src/.

The API is a thin layer: routes validate the request (Pydantic), call a service function, and return JSON.
The PCA, LDA, similarity, basket and backtest logic lives in src/ and is never re-implemented here.
"""

SERVICE_NAME = "stock-basket-api"
API_VERSION = "1.0.0"
