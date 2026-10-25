from __future__ import annotations

import os
from pathlib import Path

import yaml

from evotailor.contracts import CampaignConfig


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(path: Path) -> CampaignConfig:
    def expand(value):
        if isinstance(value, str):
            return os.path.expandvars(value)
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value
    payload = expand(yaml.safe_load(path.read_text()) or {})
    if os.environ.get("EVOTAILOR_V2_MODEL"):
        payload["model"] = os.environ["EVOTAILOR_V2_MODEL"]
    if os.environ.get("EVOTAILOR_V2_BASE_URL"):
        payload["base_url"] = os.environ["EVOTAILOR_V2_BASE_URL"]
    return CampaignConfig.model_validate(payload)
