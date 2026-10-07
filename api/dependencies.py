from api.settings import get_settings


def config_path() -> str:
    """The config.yaml the engine runs on (a string so it can key the in-memory cache)."""
    return str(get_settings().config_path)
