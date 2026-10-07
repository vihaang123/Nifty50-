"""The identity of the data the engine runs on.

Every cached object (loaded data, PCA, LDA, similarity) is keyed by a `DataSource`, so changing the provider, the data path
or the config file can never hand back an object that was built from different data. It holds only plain strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DataSource:
    config_path: str
    provider: Optional[str] = None   # DATA_PROVIDER override; None = whatever config.yaml says
    data_path: Optional[str] = None  # DATA_PATH override; None = config.yaml's prices_file
