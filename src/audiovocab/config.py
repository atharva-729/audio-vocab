"""Config loading and repo paths."""
import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights are local; the proxy blocks the HF CDN anyway


@lru_cache
def cfg():
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def path(key):
    """Path from the `paths` section; relative entries are resolved against the repo root."""
    return ROOT / cfg()["paths"][key]


def model_path(key):
    return str(ROOT / cfg()["models"][key])


def env(key, default=None):
    """Value from the environment or the repo's .env file."""
    if key in os.environ:
        return os.environ[key]
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() == key:
                    return v.strip().strip("'\"")
    return default
