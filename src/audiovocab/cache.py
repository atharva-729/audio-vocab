"""On-disk caches keyed by a hash of the request: JSON values, and embedding arrays.

Embedding keys must include dataset + clip id + model + preprocessing/perturbation params, so reruns are free and a
changed preprocessing step never reuses a stale vector.
"""
import hashlib
import json

import numpy as np

from .config import path


def digest(key):
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:24]


def _file(namespace, key, ext=".json"):
    return path("runs") / "cache" / namespace / f"{digest(key)}{ext}"


def get(namespace, key):
    f = _file(namespace, key)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def put(namespace, key, value):
    f = _file(namespace, key)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return value


def get_array(namespace, key):
    f = _file(namespace, key, ".npy")
    return np.load(f) if f.exists() else None


def put_array(namespace, key, value):
    f = _file(namespace, key, ".npy")
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, value)
    return value
