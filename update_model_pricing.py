#!/usr/bin/env python3
import json
import os
from datetime import date
from pathlib import Path

import requests

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENAI_MODEL_PAGES = {
    "gpt-3.5-turbo": "https://developers.openai.com/api/docs/models/gpt-3.5-turbo",
    "gpt-4": "https://developers.openai.com/api/docs/models/gpt-4",
    "gpt-4-turbo": "https://developers.openai.com/api/docs/models/gpt-4-turbo",
    "gpt-4o": "https://developers.openai.com/api/docs/models/gpt-4o",
    "gpt-4o-mini": "https://developers.openai.com/api/docs/models/gpt-4o-mini",
    "gpt-4.1": "https://developers.openai.com/api/docs/models/gpt-4.1",
    "gpt-4.1-mini": "https://developers.openai.com/api/docs/models/gpt-4.1-mini",
    "gpt-4.1-nano": "https://developers.openai.com/api/docs/models/gpt-4.1-nano",
    "gpt-5": "https://developers.openai.com/api/docs/models/gpt-5",
    "gpt-5-mini": "https://openai.com/api/pricing/",
    "gpt-5-nano": "https://openai.com/api/pricing/",
    "o1-preview": "https://developers.openai.com/api/docs/models/o1-preview",
    "o1-mini": "https://developers.openai.com/api/docs/models/o1-mini",
    "text-embedding-3-large": "https://developers.openai.com/api/docs/models/text-embedding-3-large",
}

# Tarifs OpenAI explicites, à maintenir avec les pages officielles si vous ne scrapez pas.
OPENAI_STATIC_PRICING = {
    "gpt-3.5-turbo": {"input": 0.50, "cached_input": None, "output": 1.50},
    "gpt-4": {"input": 30.00, "cached_input": None, "output": 60.00},
    "gpt-4-turbo": {"input": 10.00, "cached_input": None, "output": 30.00},
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    "o1-preview": {"input": 15.00, "cached_input": 7.50, "output": 60.00},
    "o1-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
    "text-embedding-3-large": {"input": 0.13, "cached_input": None, "output": None},
}

def to_per_million(value):
    if value in (None, "", "0", 0):
        return None
    return float(value) * 1_000_000

def fetch_openrouter_catalog():
    resp = requests.get(OPENROUTER_MODELS_URL, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    return {item["id"]: item for item in payload.get("data", [])}

def enrich_openrouter(model_name, model_cfg, catalog, checked_at):
    item = catalog.get(model_name)
    if not item:
        model_cfg["pricing_error"] = f"Model not found in OpenRouter catalog: {model_name}"
        return

    p = item.get("pricing", {})
    model_cfg["pricing"] = {
        "currency": "USD",
        "unit": "per_1m_tokens",
        "input": to_per_million(p.get("prompt")),
        "cached_input": to_per_million(p.get("input_cache_read")),
        "output": to_per_million(p.get("completion")),
        "request": float(p["request"]) if p.get("request") not in (None, "", "0") else None,
        "image": float(p["image"]) if p.get("image") not in (None, "", "0") else None,
        "audio": to_per_million(p.get("audio")),
        "web_search_per_1k_calls": float(p["web_search"]) * 1000 if p.get("web_search") not in (None, "", "0") else None,
        "source_type": "openrouter_catalog",
        "source_url": OPENROUTER_MODELS_URL,
        "checked_at": checked_at,
    }

def enrich_openai(model_name, model_cfg, checked_at):
    if model_name not in OPENAI_STATIC_PRICING:
        model_cfg["pricing_error"] = f"No pricing mapping defined for OpenAI model: {model_name}"
        return
    p = OPENAI_STATIC_PRICING[model_name]
    model_cfg["pricing"] = {
        "currency": "USD",
        "unit": "per_1m_tokens",
        "input": p["input"],
        "cached_input": p["cached_input"],
        "output": p["output"],
        "source_type": "official_pricing_reference",
        "source_url": OPENAI_MODEL_PAGES.get(model_name, "https://openai.com/api/pricing/"),
        "checked_at": checked_at,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ajoute/actualise les prix dans un fichier de configuration modèles.")
    parser.add_argument("input_json", help="Chemin du fichier JSON source")
    parser.add_argument("-o", "--output", help="Chemin du fichier JSON de sortie")
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + ".priced.json")
    checked_at = str(date.today())

    cfg = json.loads(input_path.read_text(encoding="utf-8"))
    defaults = cfg.get("defaults", {})
    models = cfg.get("models", {})

    catalog = fetch_openrouter_catalog()

    for model_name, model_cfg in models.items():
        base_url = model_cfg.get("base_url", defaults.get("base_url"))

        if base_url == "https://openrouter.ai/api/v1":
            enrich_openrouter(model_name, model_cfg, catalog, checked_at)
        elif base_url == "https://api.openai.com/v1":
            enrich_openai(model_name, model_cfg, checked_at)
        else:
            model_cfg["pricing_error"] = f"Unsupported base_url: {base_url}"

    output_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Written: {output_path}")

if __name__ == "__main__":
    main()