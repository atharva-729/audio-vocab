"""LLM Foundry chat client with a cache and a hard daily spend guard (copied from podcast-vibe-search).

Only for drafting word lists (Phase 5), and drafts are saved to a file for review before use.
"""
import datetime as dt
import json

import requests
import truststore

from . import cache
from .config import cfg, env, path

truststore.inject_into_ssl()  # corporate TLS inspection: trust the Windows store


class BudgetExceeded(RuntimeError):
    pass


def _spend_file():
    return path("runs") / "spend.json"


def spend_log():
    f = _spend_file()
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def spent_today(today=None):
    return spend_log().get(today or dt.date.today().isoformat(), {}).get("usd", 0.0)


def record_spend(usd, today=None):
    log = spend_log()
    day = log.setdefault(today or dt.date.today().isoformat(), {"usd": 0.0, "calls": 0})
    day["usd"] = round(day["usd"] + usd, 6)
    day["calls"] += 1
    f = _spend_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(log, indent=1), encoding="utf-8")


def cost_usd(usage, prices=None):
    p = prices or cfg()["llm"]["price_per_mtok"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    fresh = usage["prompt_tokens"] - cached
    return (fresh * p["input"] + cached * p["cached_input"] + usage["completion_tokens"] * p["output"]) / 1e6


def chat(messages, **params):
    """Return (text, info). Cached by request; refuses new calls once the daily budget is used."""
    c = cfg()["llm"]
    body = {"model": c["model"], "messages": messages, **c.get("params", {}), **params}
    hit = cache.get("llm", body)
    if hit:
        return hit["text"], {**hit["info"], "cached": True}

    if spent_today() >= c["daily_budget_usd"]:
        raise BudgetExceeded(f"spent ${spent_today():.4f} today, limit ${c['daily_budget_usd']}")

    token = env("LLMFOUNDRY_TOKEN")
    if not token:
        raise RuntimeError("LLMFOUNDRY_TOKEN not set")
    r = requests.post(f"{c['base_url']}/chat/completions", json=body, timeout=120,
                      headers={"Authorization": f"Bearer {token}:audio-vocab"})
    r.raise_for_status()
    data = r.json()
    usage = data["usage"]
    usd = cost_usd(usage)
    record_spend(usd)
    info = {"usage": usage, "usd": usd, "model": data.get("model")}
    cache.put("llm", body, {"text": data["choices"][0]["message"]["content"], "info": info})
    return data["choices"][0]["message"]["content"], {**info, "cached": False}
