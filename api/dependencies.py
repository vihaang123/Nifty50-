from api.settings import get_settings
from api.source import DataSource


def data_source() -> DataSource:
    """Which data the engine runs on for this request. It is also the key of every in-memory cache (see api/source.py)."""
    settings = get_settings()
    return DataSource(
        config_path=str(settings.config_path),
        provider=(settings.data_provider or "").lower() or None,
        data_path=settings.data_path,
    )
