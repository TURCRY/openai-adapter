# adapter.py
import os
import json
import typing as t
from typing import Any
from typing import Optional, Dict, Any, Annotated, List
import logging
from urllib.parse import urlparse
import httpx, asyncio, time
from httpx import Limits
from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Form, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, StreamingResponse
from pydantic import BaseModel
import traceback
import mimetypes, uuid

import re
import httpcore

import random
from hashlib import sha256
from uuid import uuid4
import tool_compat as tool_compat

from backend_selection import normalize_backend_url, parse_backend_candidates, select_backend_url


# -----------------------------------------------------------------------------
# Log
# -----------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
numeric_level = getattr(logging, LOG_LEVEL, logging.INFO)

logging.basicConfig(
    level=numeric_level,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
)

for name in ("adapter", "httpx", "uvicorn.error", "uvicorn.access"):
    logging.getLogger(name).setLevel(numeric_level)

log = logging.getLogger("adapter")
logger = log
log.info(f"🔧 LOG_LEVEL défini sur {LOG_LEVEL}")
log.info("ADAPTER BUILD MARKER 2026-05-24-pass1-schema-strict")

# -----------------------------------------------------------------------------
# App & CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="OpenAI-Compatible Adapter")

# ALLOW_ORIGINS peut être fourni comme CSV dans l'env (par ex: "http://openwebui:8080,http://openwebui:3000")
_allow_origins_env = os.getenv("ALLOW_ORIGINS", "")
_allow_origins = [o.strip() for o in _allow_origins_env.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Routes locales (Flask)
# -----------------------------------------------------------------------------
LOCAL_ROUTE_MAP = {
    # Compat historique
    # tu pourras un jour migrer openwebui/appflowy vers 'annoter_photo'
    "annoter": "/annoter",

    # Photo (libellé/commentaire)
    "annoter_photo": "/annoter",

    # Segments de réunion (JSON structuré)
    "annoter_segments_local": "/annoter_segments",
    "pass3e_local": "/chat_orchestre",
    "pass3e_local_alt": "/chat_orchestre",

    # RAG
    "annoter_rag": "/annoter_rag",
    "annoter_rag_vecteur": "/annoter_rag_vecteur",
    "annoter_web": "/annoter_web",
}

# (optionnel) exposer aussi les routes OCR comme “modèles-route” :
LOCAL_ROUTE_MAP.update({
    "ocr": "/ocr",
    "ocr_auto": "/ocr_auto",
    "ocr_grid": "/ocr_grid",
    "ocr_history": "/ocr_history",


    # Tous les modèles locaux → route orchestré
    "local-mistral": "/chat_orchestre",
    "local-gpt-oss_20B": "/chat_orchestre",
    "local-llama3": "/chat_orchestre",
    "local-llama2": "/chat_orchestre",
    "local-gemma": "/chat_orchestre",
    "local-phi2": "/chat_orchestre",
    "local-Qwen_2_5_0_5B": "/chat_orchestre",
    "local-Falcon3_10B": "/chat_orchestre",
    "local-DeepSeek_R1_7B": "/chat_orchestre",
    "local-SmallThinker_3B": "/chat_orchestre",
    "local-MiniCPM_V_2_6": "/chat_orchestre",


})

# ----- Extensions des routes locales via ENV -----
# Exemple d'ENV (CSV):
#   LOCAL_EXTRA_ROUTES='annoter_pdf:/annoter_pdf, asr_transcribe:/asr/transcribe'
# Exemple d'ENV (JSON):
#   LOCAL_EXTRA_ROUTES_JSON='{"annoter_pdf":"/annoter_pdf","asr_transcribe":"/asr/transcribe"}'
_LOCAL_EXTRA_ROUTES: dict[str, str] = {}
_extra_csv = os.getenv("LOCAL_EXTRA_ROUTES", "").strip()
_extra_json = os.getenv("LOCAL_EXTRA_ROUTES_JSON", "").strip()
if _extra_csv:
    for item in _extra_csv.split(","):
        if ":" in item:
            k, v = item.split(":", 1)
            k, v = k.strip(), v.strip()
            if k and v:
                _LOCAL_EXTRA_ROUTES[k] = v
if _extra_json:
    try:
        _LOCAL_EXTRA_ROUTES.update(json.loads(_extra_json))
    except Exception:
        logging.getLogger(__name__).warning("LOCAL_EXTRA_ROUTES_JSON invalide, ignoré")
LOCAL_ROUTE_MAP.update(_LOCAL_EXTRA_ROUTES)

# ComfyUI prefix (routing générique, ex: model='comfy:prompt' -> /comfyui/prompt)
COMFY_PREFIX = "/comfyui"
# -----------------------------------------------------------------------------
# Config de base & Providers
# -----------------------------------------------------------------------------

ADAPTER_API_KEY = os.getenv("ADAPTER_API_KEY", "")

# Provider 1 (principal) – ex: OpenRouter
REMOTE_BASE = os.getenv("REMOTE_BASE", "")
REMOTE_API_KEY = os.getenv("REMOTE_API_KEY", "")
REMOTE_MODELS = [m.strip() for m in os.getenv("REMOTE_MODELS", "").split(",") if m.strip()]

REMOTE_RETRY_PER_MODEL = int(os.getenv("REMOTE_RETRY_PER_MODEL", "2"))
REMOTE_RETRY_BACKOFF_SECS = float(os.getenv("REMOTE_RETRY_BACKOFF_SECS", "1.5"))

# Provider 2 (fallback) – ex: OpenAI direct
REMOTE2_BASE = os.getenv("REMOTE2_BASE", "")
REMOTE2_API_KEY = os.getenv("REMOTE2_API_KEY", "")
REMOTE2_MODELS = [m.strip() for m in os.getenv("REMOTE2_MODELS", "").split(",") if m.strip()]
FALLBACK_REMOTE2_MODEL = os.getenv("FALLBACK_REMOTE2_MODEL", "").strip()

REMOTE_MODELS_SET = set(REMOTE_MODELS + REMOTE2_MODELS)

# Embeddings routing
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "remote")  # "local" | "remote"
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
FALLBACK_ON_LOCAL_FAILURE = os.getenv("FALLBACK_ON_LOCAL_FAILURE", "1") == "1"
FALLBACK_REMOTE_MODEL = os.getenv("FALLBACK_REMOTE_MODEL", "").strip()
FALLBACK_LOCAL_EMBEDDINGS_MODEL = os.getenv("FALLBACK_LOCAL_EMBEDDINGS_MODEL", "").strip()
FALLBACK_EMBEDDINGS_MODEL = os.getenv("FALLBACK_EMBEDDINGS_MODEL", "").strip()

PREFER_LOCAL_EMBEDDINGS = os.getenv("PREFER_LOCAL_EMBEDDINGS", "1") == "1"
ALLOW_REMOTE_EMBEDDINGS = os.getenv("ALLOW_REMOTE_EMBEDDINGS", "0") == "1"
# (conserve EMBEDDINGS_MODEL, FALLBACK_LOCAL_EMBEDDINGS_MODEL, FALLBACK_EMBEDDINGS_MODEL,
#  REMOTE_BASE/REMOTE_API_KEY/TIMEOUT_LOCAL/TIMEOUT_REMOTE déjà présents)

# RAG (délégué au Flask)
RAG_MODE = os.getenv("RAG_MODE", "off")   # off|always|on_tool

MAX_TOOLS = int(os.getenv("MAX_TOOLS", "128"))
MAX_TOOL_SCHEMA_BYTES = int(os.getenv("MAX_TOOL_SCHEMA_BYTES", str(2 * 1024 * 1024)))
MAX_TOOL_CALLS_PER_RESPONSE = int(os.getenv("MAX_TOOL_CALLS_PER_RESPONSE", "32"))
MAX_FUNCTION_ARGUMENTS_BYTES = int(os.getenv("MAX_FUNCTION_ARGUMENTS_BYTES", "262144"))
MAX_FUNCTION_OUTPUT_BYTES = int(os.getenv("MAX_FUNCTION_OUTPUT_BYTES", "262144"))
FUNCTION_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
RAG_TOP_K = int(os.getenv("RAG_TOP_K") or "4")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "")
CHROMA_URL = os.getenv("CHROMA_URL", "")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "")

# Serveur local (Flask)
DEFAULT_FLASK_BACKEND_CANDIDATES = (
    "http://10.0.1.10:5050,"
    "http://10.0.1.2:5050,"
    "http://192.168.0.155:5050"
)
CONFIGURED_LOCAL_BASE = (
    os.getenv("LOCAL_LLM_BASE", "")
    or os.getenv("FLASK_BACKEND_URL", "")
    or os.getenv("LOCAL_BASE", "")
).strip()
FLASK_BACKEND_CANDIDATES = os.getenv(
    "FLASK_BACKEND_CANDIDATES",
    DEFAULT_FLASK_BACKEND_CANDIDATES,
)
LOCAL_BACKEND_CANDIDATES = parse_backend_candidates(CONFIGURED_LOCAL_BASE, FLASK_BACKEND_CANDIDATES)
LOCAL_BASE = LOCAL_BACKEND_CANDIDATES[0] if LOCAL_BACKEND_CANDIDATES else ""
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "")
LOCAL_PING_PATH = os.getenv("LOCAL_PING_PATH", "/ping")

# OCR
LOCAL_OCR_PATH = os.getenv("LOCAL_OCR_PATH", "/ocr/convert")

# WOL via Home Assistant
ENABLE_WOL = os.getenv("ENABLE_WOL", "1") == "1"
URL_HAY_PUBLIQUE = os.getenv("URL_HAY_PUBLIQUE", "").strip().strip("/")
WEBHOOK_WAKE_PCFIXE = os.getenv("WEBHOOK_WAKE_PCFIXE", "").strip()
WEBHOOK_URL = (f"https://{URL_HAY_PUBLIQUE}/api/webhook/{WEBHOOK_WAKE_PCFIXE}"
               if URL_HAY_PUBLIQUE and WEBHOOK_WAKE_PCFIXE else None)

USE_RESPONSES = os.getenv("OPENAI_USE_RESPONSES", "true").lower() == "true"
FORCE_CHAT    = os.getenv("OPENAI_FORCE_CHAT", "false").lower() == "true"
COMPAT_MODE   = os.getenv("OPENAI_COMPAT_MODE", "").lower()

# JSON

JSON_START_RE = re.compile(r"[{\[]")


# -----------------------------------------------------------------------------
# Timeouts & HTTP client partagé
# -----------------------------------------------------------------------------
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "20"))
READ_TIMEOUT = float(os.getenv("READ_TIMEOUT", "30"))
PING_READ_TIMEOUT = float(os.getenv("PING_READ_TIMEOUT", "10"))
WAIT_READY_SECS = int(os.getenv("WAIT_READY_SECS", "300"))
WAIT_SLEEP_SEC = float(os.getenv("WAIT_SLEEP_SEC", "5"))
PING_INTERVAL = float(os.getenv("PING_INTERVAL", "10800"))

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "900"))            # requêtes locales normales
LLM_INIT_TIMEOUT = float(os.getenv("LLM_INIT_TIMEOUT", "1500"))  # cold start / changement modèle
REMOTE_API_TIMEOUT = float(os.getenv("REMOTE_API_TIMEOUT", "600"))

OCR_TIMEOUT = float(os.getenv("OCR_TIMEOUT", "3900"))           # OCR lourds / fichiers

TIMEOUT_PING = httpx.Timeout(timeout=10.0, connect=CONNECT_TIMEOUT, read=PING_READ_TIMEOUT,
                             write=PING_READ_TIMEOUT, pool=CONNECT_TIMEOUT)
TIMEOUT_LOCAL = httpx.Timeout(timeout=LLM_TIMEOUT, connect=CONNECT_TIMEOUT, read=LLM_TIMEOUT,
                              write=LLM_TIMEOUT, pool=CONNECT_TIMEOUT)
TIMEOUT_INIT = httpx.Timeout(timeout=LLM_INIT_TIMEOUT, connect=CONNECT_TIMEOUT, read=LLM_INIT_TIMEOUT,
                             write=LLM_INIT_TIMEOUT, pool=CONNECT_TIMEOUT)
TIMEOUT_REMOTE = httpx.Timeout(timeout=REMOTE_API_TIMEOUT, connect=CONNECT_TIMEOUT, read=REMOTE_API_TIMEOUT,
                               write=REMOTE_API_TIMEOUT, pool=CONNECT_TIMEOUT)

# Infos sur le modèle local (n_ctx, max_tokens, etc.)
LOCAL_MODEL_INFO: dict[str, t.Any] = {}
_LOCAL_MODEL_INFO_TS: float = 0.0
LOCAL_MODEL_INFO_TTL: float = 300.0  # en secondes, par ex. 5 min
DEFAULT_LOCAL_N_CTX = 4096  # ou 8192 si votre modèle le supporte
DEFAULT_LOCAL_MAX_TOKENS = 1024  # valeur par défaut si /model_info ne précise rien
DEFAULT_LOCAL_MARGE_TOKENS = 128 # valeur par défaut
DEFAULT_LOCAL_MIN_PROMPT_TOKENS = 512 # valeur par défaut

# Ping throttling
_last_ping_ok_ts = 0.0
_ping_lock = asyncio.Lock()
_backend_reselect_lock = asyncio.Lock()


# Client HTTP partagé
HTTP_LIMITS = Limits(max_connections=100, max_keepalive_connections=20)
_http = httpx.AsyncClient(limits=HTTP_LIMITS, timeout=TIMEOUT_PING)


async def _probe_backend_ping(base_url: str, ping_path: str, headers: dict[str, str] | None = None) -> tuple[bool, str]:
    url = f"{normalize_backend_url(base_url)}{ping_path}"
    try:
        r = await _http.get(url, headers=headers or {}, timeout=TIMEOUT_PING)
        if r.status_code == 200:
            return True, "HTTP 200"
        return False, f"HTTP {r.status_code}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


async def configure_local_backend_from_candidates() -> None:
    global LOCAL_BASE, _last_ping_ok_ts
    configured_display = CONFIGURED_LOCAL_BASE or "(non definie)"
    log.info("Backend Flask configure: %s", configured_display)
    log.info("Backend Flask candidats: %s", ", ".join(LOCAL_BACKEND_CANDIDATES) or "(aucun)")
    selected, attempts = await select_backend_url(
        LOCAL_BACKEND_CANDIDATES,
        ping_path=LOCAL_PING_PATH,
        headers=_llm_headers(),
        probe=_probe_backend_ping,
    )
    for attempt in attempts:
        if attempt["ok"] == "true":
            log.info("Backend Flask candidat OK: %s (%s)", attempt["url"], attempt["detail"])
        else:
            log.warning("Backend Flask candidat KO: %s (%s)", attempt["url"], attempt["detail"])
    if selected:
        LOCAL_BASE = selected
        _last_ping_ok_ts = time.monotonic()
        log.info("Backend Flask retenu: %s", LOCAL_BASE)
        return
    log.error("Aucun backend Flask candidat ne repond a %s. URL conservee: %s", LOCAL_PING_PATH, LOCAL_BASE or "(aucune)")


def _is_retryable_local_failure(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return bool(status and status >= 500)
    if isinstance(exc, (httpx.TimeoutException, httpx.RequestError)):
        return True
    name = type(exc).__name__.lower()
    return "timeout" in name or "connect" in name or "network" in name


async def _reselect_local_backend(reason: str) -> bool:
    global LOCAL_BASE, _last_ping_ok_ts
    async with _backend_reselect_lock:
        old_base = LOCAL_BASE
        ordered_candidates = [u for u in LOCAL_BACKEND_CANDIDATES if u != old_base]
        if old_base:
            ordered_candidates.append(old_base)
        log.warning(
            "Backend Flask runtime fallback: cause=%s old=%s candidates=%s",
            reason,
            old_base or "(aucun)",
            ", ".join(ordered_candidates) or "(aucun)",
        )
        selected, attempts = await select_backend_url(
            ordered_candidates,
            ping_path=LOCAL_PING_PATH,
            headers=_llm_headers(),
            probe=_probe_backend_ping,
        )
        for attempt in attempts:
            if attempt["ok"] == "true":
                log.info("Backend Flask runtime candidat OK: %s (%s)", attempt["url"], attempt["detail"])
            else:
                log.warning("Backend Flask runtime candidat KO: %s (%s)", attempt["url"], attempt["detail"])
        if not selected:
            log.error("Backend Flask runtime fallback impossible: aucun candidat OK apres cause=%s old=%s", reason, old_base or "(aucun)")
            return False
        LOCAL_BASE = selected
        _last_ping_ok_ts = time.monotonic()
        log.warning("Backend Flask runtime fallback retenu: old=%s new=%s cause=%s", old_base or "(aucun)", LOCAL_BASE, reason)
        return True


async def _local_request_once_with_runtime_fallback(method: str, path: str, *, timeout: httpx.Timeout | float, **kwargs) -> httpx.Response:
    base = LOCAL_BASE
    url = f"{base}{path}"
    try:
        r = await _http.request(method, url, timeout=timeout, **kwargs)
        r.raise_for_status()
        return r
    except Exception as exc:
        if not _is_retryable_local_failure(exc):
            raise
        reason = f"{type(exc).__name__}: {exc}"
        if not await _reselect_local_backend(reason):
            raise HTTPException(status_code=502, detail=f"Local Flask backend unavailable after runtime fallback: {reason}") from exc
        retry_url = f"{LOCAL_BASE}{path}"
        log.warning("Backend Flask runtime retry: method=%s path=%s old=%s new=%s cause=%s", method, path, base or "(aucun)", LOCAL_BASE, reason)
        r = await _http.request(method, retry_url, timeout=timeout, **kwargs)
        r.raise_for_status()
        return r
@app.on_event("shutdown")
async def _shutdown_http_client():
    await _http.aclose()

# -----------------------------------------------------------------------------
# Modèles locaux / alias
# -----------------------------------------------------------------------------
# au chargement du module
DISCOVER_LOCAL_MODELS = os.getenv("DISCOVER_LOCAL_MODELS", "1") == "1"
LOCAL_DISCOVERY_PATH  = os.getenv("LOCAL_DISCOVERY_PATH", "/models")
_AVAILABLE_LOCAL_IDS: set[str] = set()
_DISCOVERED_LOCAL = set()
LOCAL_MODELS = [m.strip() for m in os.getenv("LOCAL_MODELS", "").split(",") if m.strip()]
MODEL_ALIAS = {
    # LLMs
    "local-mistral": "Mistral_7B",
    "local-gpt-oss_20B": "GPT_OSS_20B_4BIT",
    "local-llama3": "LLaMA_3_8B",
    "local-llama2": "LLaMA_2_7B",
    "local-gemma": "Gemma_7B",
    "local-phi2": "Phi-2_7B",
    "local-Qwen_2_5_0_5B": "Qwen_2_5_0_5B",
    "local-Falcon3_10B": "Falcon3_10B",
    "local-DeepSeek_R1_7B": "DeepSeek_R1_7B",
    "local-SmallThinker_3B": "SmallThinker_3B",
    "local-MiniCPM_V_2_6": "MiniCPM_V_2_6",
    # Embeddings
    "local-embed": "Nomic_Embed",
    # ASR
    "local-voxtral-mini": "Voxtral_Mini_3B_Transformers",
    "local-voxtral-large": "Voxtral_Small_24B_Transformers",
    # OCR / Tools
    "local-ocr": "OpenCV_Tesseract_OCR",
    # Compat routes
    "annoter": "Qwen_2_5_14B",
    "annoter_rag": "Qwen_2_5_14B",
    "annoter_rag_vecteur": "Qwen_2_5_14B",
    "annoter_web": "Qwen_2_5_14B",

    # compte-rendu annotés
    "pass3e_local": "Qwen_2_5_14B",
    "pass3e_local_alt": "DeepSeek_R1_7B",

}


SENSITIVE_LOCAL_ONLY = {"annoter_rag"}

MODEL_REGISTRY = {
    # modèle local par défaut (ce que vous avez déjà)
    "annoter_segments_local": {
        "backend": "gpt4all",
        "model": "Qwen_2_5_14B",
        "json_mode": True,   # on veut du JSON propre
    },
    # un second modèle local si besoin
    "annoter_segments_local_alt": {
        "backend": "gpt4all",
        "model": "DeepSeek_R1_7B",
        "json_mode": True,
    },
    # un modèle remote (OpenAI ou autre) spécialisé JSON
    "annoter_segments_remote": {
        "backend": "openai",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4.1-mini",
        "json_mode": True,
    },
    # un modèle remote (OpenAI ou autre) spécialisé JSON
    "annoter_segments_remote_alt": {
        "backend": "openai",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4.1-mini",
        "json_mode": True,
    },
    # un modèle remote (OpenAI ou autre) spécialisé JSON
    "annoter_segments_remote_alt2": {
        "backend": "openai",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-5-mini",
        "json_mode": True,
    },
    "pass2e_remote": {
        "backend": "openai",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4.1-mini",
        "json_mode": True,
    },
    # un modèle remote_model spécialisé JSON
    "report_remote": {
        "backend": "openai",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        # Benchmark A60: gpt-4.1-mini retenu pour report_remote en exploitation normale;
        # gpt-5-mini conserve en variante plus lente.
        "model": "gpt-4.1-mini",
        "json_mode": True,
    },
    "report_debrief_remote": {
        "backend": "openai",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4.1-mini",
        "json_mode": True,
    },
    "report_remote_alt": {
        "backend": "openai",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-5",
        "json_mode": True,
    },

    "report_remote_alt2": {
        "backend": "openai",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-5",
        "json_mode": True,
    },
    # un modèle remote_model spécialisé JSON
    "pass3_remote": {
        "backend": "openai",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-5-mini",
        "json_mode": True,
    },
    "pass3_remote_alt": {
        "backend": "openai",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-5",
        "json_mode": True,
    },

    "pass3_remote_alt2": {
        "backend": "openai",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-5",
        "json_mode": True,
    },
    # modèle local par défaut (ce que vous avez déjà)
    "pass3e_local": {
        "backend": "gpt4all",
        "model": "Qwen_2_5_14B",
        "json_mode": True,   # on veut du JSON propre
    },
    "pass3e_local_alt": {
        "backend": "gpt4all",
        "model": "DeepSeek_R1_7B",
        "json_mode": True,   # on veut du JSON propre
    },
    # un second modèle local si besoin
    "pass3e_remote": {
        "backend": "openai",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-5-mini",
        "json_mode": True,
    },
    # modèle “général”, sans contrainte JSON
    "chat_general": {
        "backend": "gpt4all",
        "model": "LLaMA_3_8B",
        "json_mode": False,
    },
}
MODEL_REGISTRY.update({
  "pass3a_remote": {
      "backend": "openai",
      "api_base": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY",
      "model": "gpt-5-mini",
      "json_mode": True,
  },
  "pass3b_remote": {
      "backend": "openai",
      "api_base": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY",
      "model": "gpt-5-mini",
      "json_mode": True,
  },
  "pass3c_remote": {
      "backend": "openai",
      "api_base": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY",
      "model": "gpt-5-mini",
      "json_mode": True,
  },
  "pass3d_remote": {
      "backend": "openai",
      "api_base": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY",
      "model": "gpt-5-mini",
      "json_mode": True,
  },
  "pass3a_remote_alt": {
      "backend": "openai",
      "api_base": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY",
      "model": "gpt-5",
      "json_mode": True,
  },
  "pass3b_remote_alt": {
      "backend": "openai",
      "api_base": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY",
      "model": "gpt-5",
      "json_mode": True,
  },
  "pass3c_remote_alt": {
      "backend": "openai",
      "api_base": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY",
      "model": "gpt-5",
      "json_mode": True,
  },
  "pass3d_remote_alt": {
      "backend": "openai",
      "api_base": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY",
      "model": "gpt-5",
      "json_mode": True,
  },
  "pass3a_remote_alt2": {
      "backend": "openai",
      "api_base": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY",
      "model": "gpt-5",
      "json_mode": True,
  },
  "pass3b_remote_alt2": {
      "backend": "openai",
      "api_base": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY",
      "model": "gpt-5",
      "json_mode": True,
  },
  "pass3c_remote_alt2": {
      "backend": "openai",
      "api_base": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY",
      "model": "gpt-5",
      "json_mode": True,
  },
  "pass3d_remote_alt2": {
      "backend": "openai",
      "api_base": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY",
      "model": "gpt-5",
      "json_mode": True,
  },
})


# -----------------------------------------------------------------------------
# Config distante par modèle (overrides)
# -----------------------------------------------------------------------------
# --- REMOTE_CONF: log explicite si absent ou invalide ---
REMOTE_CONF_PATH = os.getenv("REMOTE_CONF_PATH")
_REMOTE_CONF: Dict[str, Any] = {}
if REMOTE_CONF_PATH and os.path.exists(REMOTE_CONF_PATH):
    try:
        with open(REMOTE_CONF_PATH, "r", encoding="utf-8") as f:
            _REMOTE_CONF = json.load(f)
    except Exception as e:
        log.warning("REMOTE_CONF: impossible de lire %s (%r) -> defaults vides", REMOTE_CONF_PATH, e)
else:
    log.info("REMOTE_CONF: aucune configuration distante fournie (REMOTE_CONF_PATH manquant)")


def _remote_overrides(model_id: str) -> dict:
    d = dict(_REMOTE_CONF.get("defaults") or {})
    d.update(_REMOTE_CONF.get("models", {}).get(model_id, {}))
    return d


def _resolve_model_id(model_id: str) -> str:
    """
    Résout un identifiant logique vers un id de registry si alias.
    - Si model_id est déjà dans MODEL_REGISTRY => renvoie model_id
    - Sinon, tente MODEL_ALIAS[model_id]
    """
    if model_id in MODEL_REGISTRY:
        return model_id
    aliased = MODEL_ALIAS.get(model_id)
    if aliased and aliased in MODEL_REGISTRY:
        return aliased
    return model_id


def _model_cfg(model: str) -> dict:
    """
    Retourne la configuration fusionnée pour un modèle donné.
    - Si `model` est un ID logique connu (MODEL_REGISTRY), on injecte sa config (backend/base/api_key_env/model physique).
    - Puis on applique les overrides de REMOTE_CONF (config_remote.json) :
        - d'abord les defaults
        - puis l'entrée qui match exactement `model` (si elle existe)
    """
    cfg = _remote_overrides(model)  # ← defaults + entry modèle (REMOTE_CONF)

    # 2) BRIDGE : si l'ID demandé est un ID logique connu
    reg = MODEL_REGISTRY.get(model)
    if reg:
        backend = reg.get("backend")
        cfg.setdefault("backend", backend)
        # Propager json_mode du registry vers cfg
        if "json_mode" in reg:
            cfg["json_mode"] = bool(reg["json_mode"])


        # Cas remote openai-like : on fixe le modèle physique + base + envkey
        if backend == "openai":
            cfg.setdefault("model", reg.get("model"))  # ex: gpt-5-mini, gpt-5, anthropic/...
            cfg.setdefault("base_url", reg.get("api_base") or reg.get("base_url"))
            cfg.setdefault("api_key_env", reg.get("api_key_env", "OPENAI_API_KEY"))

        # Cas local : rien à faire ici (géré ailleurs)
        # (on peut laisser cfg tel quel)

    # 3) fallback environnement si toujours incomplet
    if not cfg.get("base_url"):
        cfg["base_url"] = os.getenv("REMOTE_BASE", "https://api.openai.com/v1")
    if not cfg.get("api_key_env"):
        cfg["api_key_env"] = "OPENAI_API_KEY"

    return cfg




# -----------------------------------------------------------------------------
# Helpers auth / CORS / URL
# -----------------------------------------------------------------------------
def _check_adapter_auth(authorization: str | None):
    if ADAPTER_API_KEY:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Bearer token")
        tok = authorization.replace("Bearer ", "", 1).strip()
        if tok != ADAPTER_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

ALLOWED_WEB_DOMAINS = [d.strip().lower() for d in os.getenv("ALLOWED_WEB_DOMAINS", "*").split(",") if d.strip()]

def _is_allowed_url(url: str) -> bool:
    if "*" in ALLOWED_WEB_DOMAINS:
        return True
    try:
        netloc = urlparse(url).netloc.lower()
        for allowed in ALLOWED_WEB_DOMAINS:
            a = allowed.strip().lower()
            if not a:
                continue
            if netloc == a or netloc.endswith("." + a):
                return True
        return False
    except Exception:
        return False

def _llm_headers():
    h = {}
    if LOCAL_API_KEY:
        h["x-api-key"] = LOCAL_API_KEY
    return h

def _as_list(x: list[str] | str) -> list[str]:
    return x if isinstance(x, list) else [x]

def _is_local_model(model: str) -> bool:
    m = model.strip() if isinstance(model, str) else model
    if not m:
        return False
    if m.startswith("route:"):  # force locale sur une route arbitraire
        return True
    # modèles explicitement locaux, alias 'local-*', routes connues (y.c. injectées via ENV),
    # et préfixe comfy:*
    return (
        m in LOCAL_MODELS
        or m.startswith("local-")
        or m in LOCAL_ROUTE_MAP
        or m.startswith("comfy:")
    )

def _resolve_local_path(route_hint: str, meta: dict | None) -> str:
    """
    Règles de résolution du path local :
      1) meta.path / meta.route ont la priorité
      2) 'route:<xxx>' force le path '/<xxx>'
      3) mapping LOCAL_ROUTE_MAP (y.c. via ENV)
      4) fallback générique: '/<route_hint>'
    """
    # 1) priorité aux métadonnées
    if meta:
        if meta.get("path"):
            return str(meta["path"]).strip()
        if meta.get("route"):
            r = str(meta["route"]).strip().lstrip("/")
            return f"/{r}"
    # 2) 'route:xxx'
    rh = (route_hint or "").strip()
    if rh.startswith("route:"):
        r = rh.split(":", 1)[1].strip().lstrip("/")
        return f"/{r}"
    # 3) mapping connu
    if rh in LOCAL_ROUTE_MAP:
        return LOCAL_ROUTE_MAP[rh]
    # 4) fallback générique
    return f"/{rh.lstrip('/')}"

def _pick_ids(headers: dict, body: dict) -> tuple[str, str]:
    app_id = (headers.get("x-app-id") or body.get("app_id") or "").strip()
    conv_id = (headers.get("x-conversation-id") or body.get("conversation_id") or body.get("memory_id") or "").strip()

    if not app_id:
        app_id = "unknown_app"

    if not conv_id:
        ua = (headers.get("user-agent") or "ua").encode("utf-8", "ignore")
        conv_id = "ua_" + sha256(ua).hexdigest()[:12]

    return app_id, conv_id


def _build_memory_append(messages: list[dict], n_turns: int = 6) -> str:
    """
    Construit une mémoire glissante sur N tours (USER+ASSISTANT).
    - n_turns=6 => ~6 paires Q/R max
    - Prend uniquement les messages user/assistant.
    """
    if not messages:
        return ""

    # On garde uniquement user/assistant
    ua = [m for m in messages if m.get("role") in ("user", "assistant")]

    # On coupe sur les 2*n_turns derniers messages (une paire = 2 messages)
    keep = max(2, int(n_turns) * 2)
    ua = ua[-keep:]

    lines = []
    for m in ua:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"USER: {content}")
        else:
            lines.append(f"ASSISTANT: {content}")

    return "\n".join(lines).strip()



def _derive_conv_id_from_messages(app_id: str, ua_fallback: str, messages: list[dict]) -> str:
    if not messages:
        base = f"{app_id}|{ua_fallback}"
        return "chat_" + sha256(base.encode("utf-8")).hexdigest()[:16]

    sys0 = ""
    u1 = ""
    u2 = ""

    # On ne regarde que le début (stabilisé)
    for m in messages[:20]:
        role = (m.get("role") or "").strip().lower()
        content = m.get("content")

        if role == "system" and not sys0 and isinstance(content, str) and content.strip():
            sys0 = content.strip()

        if role == "user" and isinstance(content, str) and content.strip():
            if not u1:
                u1 = content.strip()
            elif not u2:
                u2 = content.strip()
                break

    # Anchor stable (n'inclut pas la taille de messages, ni les réponses assistant)
    anchor = f"SYS:{sys0}\nU1:{u1}\nU2:{u2}".strip()
    if not u1 and ua_fallback:
        anchor = f"UA:{ua_fallback}"

    seed = f"{app_id}\n{anchor}".encode("utf-8", errors="ignore")
    return "chat_" + sha256(seed).hexdigest()[:16]

def _canonical_uses_structured_outputs(canonical: str) -> bool:
    return False


def _enforce_schema_additional_properties_false(node: Any) -> Any:
    if isinstance(node, dict):
        if node.get("type") == "object":
            node["additionalProperties"] = False
        for key in ("properties", "patternProperties", "$defs", "definitions"):
            child_map = node.get(key)
            if isinstance(child_map, dict):
                for child in child_map.values():
                    _enforce_schema_additional_properties_false(child)
        items = node.get("items")
        if isinstance(items, dict):
            _enforce_schema_additional_properties_false(items)
        elif isinstance(items, list):
            for child in items:
                _enforce_schema_additional_properties_false(child)
        for key in ("oneOf", "anyOf", "allOf"):
            variants = node.get(key)
            if isinstance(variants, list):
                for child in variants:
                    _enforce_schema_additional_properties_false(child)
    return node


def _stricten_response_format(response_format: dict | None) -> dict | None:
    if not isinstance(response_format, dict):
        return response_format
    cloned = json.loads(json.dumps(response_format, ensure_ascii=False))
    schema = ((cloned.get("json_schema") or {}).get("schema")) if isinstance(cloned.get("json_schema"), dict) else None
    if isinstance(schema, dict):
        _enforce_schema_additional_properties_false(schema)
    return cloned


def _response_format_marker(response_format: dict | None) -> str:
    if not response_format:
        return "none"
    payload = json.dumps(response_format, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


log.info("PASS1 response_format disabled: prompt JSON strict + local validation")


# -----------------------------------------------------------------------------
# Readiness & WOL
# -----------------------------------------------------------------------------
async def _is_local_up() -> bool:
    url = f"{LOCAL_BASE}{LOCAL_PING_PATH}"
    try:
        r = await _http.get(url, headers=_llm_headers(), timeout=TIMEOUT_PING)
        return r.status_code == 200
    except Exception:
        return False

async def _trigger_wol_via_ha() -> bool:
    if not WEBHOOK_URL:
        return False
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(WEBHOOK_URL, timeout=10)
            r.raise_for_status()
            return True
    except Exception:
        return False

async def _is_local_up_cached() -> bool:
    global _last_ping_ok_ts
    now = time.monotonic()
    if (now - _last_ping_ok_ts) < PING_INTERVAL:
        return True
    ok = await _is_local_up()
    if ok:
        _last_ping_ok_ts = now
    return ok

async def _ensure_local_ready() -> bool:
    if await _is_local_up_cached():
        return True
    if not ENABLE_WOL:
        return False
    await _trigger_wol_via_ha()
    deadline = time.monotonic() + WAIT_READY_SECS
    while time.monotonic() < deadline:
        if await _is_local_up():
            global _last_ping_ok_ts
            _last_ping_ok_ts = time.monotonic()
            return True
        await asyncio.sleep(WAIT_SLEEP_SEC)
    return False



# -----------------------------------------------------------------------------
# Providers (normalisation & fallback)
# -----------------------------------------------------------------------------
def _provider_name(base: str) -> str:
    base = (base or "").lower()
    if "openrouter.ai" in base:
        return "openrouter"
    if "api.openai.com" in base:
        return "openai"
    return "other"

def _normalize_model_for_base(model_id: str, base: str) -> str:
    prov = _provider_name(base)
    m = model_id
    if prov == "openai":
        m = m.replace("openai/", "")  # openrouter → openai direct
    return m

def _remote_model_ids(requested_model: str, cfg: dict, base: str) -> tuple[str, str, bool]:
    physical_model = str(cfg.get("model") or requested_model).strip()
    explicit_provider_model = str(cfg.get("provider_model") or "").strip()
    if explicit_provider_model:
        return physical_model, explicit_provider_model, True
    return physical_model, _normalize_model_for_base(physical_model, base), False


def _should_fallback_remote(status_code: int | None, exc: Exception | None) -> bool:
    if exc is not None:
        return True
    if status_code is None:
        return True
    return status_code in (401, 402, 403, 408, 409, 412, 413, 415, 422, 429, 500, 502, 503, 504)

def _messages_to_responses_payload(messages: list[dict]):
    """Convertit un historique de messages Chat en input pour /responses."""
    system = ""
    inputs = []
    for m in messages:
        if m["role"] == "system":
            system += m["content"] + "\n"
        else:
            inputs.append({"role": m["role"], "content": m["content"]})
    return (system.strip() or None, inputs)

def apply_json_constraint_to_messages(messages):
    """
    Ajoute une consigne JSON-only dans le dernier message utilisateur.
    Fonctionne pour un endpoint OpenAI-like de type chat/completions.
    """
    if not messages:
        return messages

    # On modifie le dernier message user
    last = messages[-1]
    if last.get("role") == "user":
        extra = (
            "\n\nRéponds uniquement par un objet JSON valide. "
            "Aucun texte hors JSON."
            )
        last["content"] = (last.get("content") or "") + extra

    return messages


def _extract_json_block(text: str) -> str:
    """
    Extrait le premier bloc JSON plausible d'un texte LLM.
    - supporte du texte avant/après,
    - supporte un wrapper <json> ... </json>.
    """
    if not text:
        raise ValueError("Réponse LLM vide.")

    t = text.strip()

    # 1) wrapper <json>...</json> éventuel
    m = re.search(r"<json>(.+?)</json>", t, flags=re.S | re.I)
    if m:
        candidate = m.group(1).strip()
    else:
        candidate = t

    # 2) prendre du premier { ou [ au dernier } ou ]
    start_match = JSON_START_RE.search(candidate)
    if not start_match:
        raise ValueError("Aucun début de JSON trouvé dans la réponse.")

    start = start_match.start()
    end_brace = candidate.rfind("}")
    end_bracket = candidate.rfind("]")
    end = max(end_brace, end_bracket)

    if end <= start:
        raise ValueError("Bornes JSON incohérentes dans la réponse.")

    return candidate[start : end + 1]


def _looks_incomplete_output(raw: str) -> bool:
    """
    Détecte une réponse de l'API OpenAI de type 'status: incomplete'
    ou un message clairement lié à max_output_tokens.
    """
    if not raw:
        return False
    txt = raw.lower()
    if '"status"' in txt and '"incomplete"' in txt:
        return True
    if "max_output_tokens" in txt:
        return True
    return False

def _fallback_chain_for(canonical: str) -> list[str]:
    if canonical == "annoter_segments_remote":
        return ["annoter_segments_remote", "annoter_segments_remote_alt", "annoter_segments_remote_alt2"]
    if canonical == "annoter_segments_remote_alt":
        return ["annoter_segments_remote_alt", "annoter_segments_remote_alt2"]

    if canonical == "pass2e_remote":
        return ["pass2e_remote"]

    if canonical == "pass3e_remote":
        return ["pass3e_remote"]

    if canonical == "report_remote":
        return ["report_remote", "report_remote_alt", "report_remote_alt2"]

    if canonical == "report_debrief_remote":
        return ["report_debrief_remote"]

    if canonical == "report_remote_alt":
        return ["report_remote_alt", "report_remote_alt2"]

    if canonical == "pass3_remote":
        return ["pass3_remote", "pass3_remote_alt", "pass3_remote_alt2"]
    if canonical == "pass3_remote_alt":
        return ["pass3_remote_alt", "pass3_remote_alt2"]
    # passe 3E
    if canonical == "pass3e_local":
        return ["pass3e_local", "pass3e_local_alt", "pass3e_remote"]

    if canonical == "pass3e_local_alt":
        return ["pass3e_local_alt", "pass3e_remote"]
    # passe 3A
    if canonical == "pass3a_remote":
        return ["pass3a_remote", "pass3a_remote_alt", "pass3a_remote_alt2"]

    if canonical == "pass3a_remote_alt":
        return ["pass3a_remote_alt", "pass3a_remote_alt2"]
    # passe 3B
    if canonical == "pass3b_remote":
        return ["pass3b_remote", "pass3b_remote_alt", "pass3b_remote_alt2"]

    if canonical == "pass3b_remote_alt":
        return ["pass3b_remote_alt", "pass3b_remote_alt2"]
    # passe 3C
    if canonical == "pass3c_remote":
        return ["pass3c_remote", "pass3c_remote_alt", "pass3c_remote_alt2"]

    if canonical == "pass3c_remote_alt":
        return ["pass3c_remote_alt", "pass3c_remote_alt2"]
    # passe 3D
    if canonical == "pass3d_remote":
        return ["pass3d_remote", "pass3d_remote_alt", "pass3d_remote_alt2"]

    if canonical == "pass3d_remote_alt":
        return ["pass3d_remote_alt", "pass3d_remote_alt2"]

    return [canonical]

def _is_effectively_empty_payload(obj) -> bool:
    """
    Détecte les payloads 'vides' même si JSON valide :
    - {}
    - {"text":""}
    - {"content":""}
    - {"choices":[{"message":{"content":""}}]} etc. (selon ce que vous logguez)
    """
    if obj is None:
        return True

    if isinstance(obj, str):
        return len(obj.strip()) == 0

    if isinstance(obj, dict):
        if len(obj.keys()) == 0:
            return True

        # cas fréquent observé : {"text": ""} ou {"text": "   "}
        if "text" in obj and isinstance(obj["text"], str) and not obj["text"].strip():
            return True

        # variantes
        for k in ("content", "output_text", "answer"):
            if k in obj and isinstance(obj[k], str) and not obj[k].strip():
                return True

        return False

    if isinstance(obj, list):
        return len(obj) == 0

    return False

def _looks_like_json(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    return (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]"))

def _is_effectively_empty_raw(raw: str) -> bool:
    """
    Détecte :
    - raw vide/whitespace
    - raw JSON avec {"text":""} / {} / [].
    """
    if raw is None or not str(raw).strip():
        return True

    t = str(raw).strip()

    # Si c’est du JSON brut, on tente un parse simple
    if _looks_like_json(t):
        try:
            import json
            obj = json.loads(t)
            return _is_effectively_empty_payload(obj)
        except Exception:
            # si JSON cassé, on ne considère pas "vide", on laissera réparer/parsers
            return False

    return False


def _repair_truncated_json(raw_text: str) -> Any | None:
    """
    Tentative best-effort pour récupérer un JSON à partir d'une réponse LLM
    possiblement tronquée.
    - ferme les { / [ manquants
    - supprime les fins de chaînes cassées
    - enlève les virgules finales
    Retourne un objet Python ou None si tout échoue.
    """
    if not raw_text:
        return None

    # 1) on essaie quand même la logique standard d'abord
    try:
        return extract_json_from_llm(raw_text)
    except Exception:
        pass

    # 2) on repart du texte brut sans wrapper <json>
    t = raw_text.strip()
    t = re.sub(r"^<json>\s*", "", t, flags=re.I)
    t = re.sub(r"\s*</json>\s*$", "", t, flags=re.I)

    # prendre à partir du premier { ou [
    m = JSON_START_RE.search(t)
    if not m:
        return None
    candidate = t[m.start():]

    def _light_cleanup(s: str) -> str:
        s = re.sub(r",\s*([}\]])", r"\1", s)
        s = re.sub(r",\s*$", "", s)
        return s

    def _balance(s: str) -> str | None:
        opens_curly = s.count("{")
        closes_curly = s.count("}")
        if closes_curly > opens_curly:
            return None
        opens_sq = s.count("[")
        closes_sq = s.count("]")
        if closes_sq > opens_sq:
            return None
        s = s + "}" * (opens_curly - closes_curly)
        s = s + "]" * (opens_sq - closes_sq)
        return s

    base = candidate
    max_trim = 2000  # borne pour ne pas tout couper
    start = max(0, len(base) - max_trim)

    for end in range(len(base), start, -1):
        prefix = base[:end]

        # si nombre de guillemets est impair, on coupe après le dernier "
        if prefix.count('"') % 2 == 1:
            last_q = prefix.rfind('"')
            if last_q != -1:
                prefix = prefix[:last_q]

        prefix = _light_cleanup(prefix)
        balanced = _balance(prefix)
        if not balanced:
            continue
        balanced = _light_cleanup(balanced)
        try:
            return json.loads(balanced)
        except Exception:
            continue

    return None


def _try_parse_json_strict(payload: str) -> Any:
    """
    Essaie de parser le JSON en plusieurs passes "légères".
    On veut rester simple mais robuste vis-à-vis des sorties LLM.
    """
    # PASS 1 : brut
    try:
        return json.loads(payload)
    except Exception:
        pass

    # PASS 2 : suppression des virgules finales avant } ou ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", payload)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # PASS 3 : remplacement ' → " (dict Python)
    cleaned2 = cleaned.replace("'", '"')
    try:
        return json.loads(cleaned2)
    except Exception:
        pass

    raise ValueError("Impossible de parser la réponse comme JSON après réparations légères.")


def extract_json_from_llm(raw_text: str) -> Any:
    """
    Version renforcée d'extraction JSON depuis une sortie LLM.
    Peut renvoyer dict ou list suivant le cas.
    """
    block = _extract_json_block(raw_text)
    return _try_parse_json_strict(block)

REPORT_KEYS = {
    "resume_global",
    "themes",
    "themes_abordes",
    "actions",
    "perspectives",
    "demandes_documents_globales",
    "problems",
}

def _is_report_schema(obj: Any) -> bool:
    return isinstance(obj, dict) and REPORT_KEYS.issubset(set(obj.keys()))

def _fallback_json_for_model(canonical: str) -> dict:
    # Passe 1A
    if canonical in (
        "annoter_segments_remote",
        "annoter_segments_remote_alt",
        "annoter_segments_remote_alt2",
        "annoter_segments_local",
        "annoter_segments_local_alt",
    ):
        return {"resume_segment": "", "themes": [], "actions": [], "problems": []}

    if canonical == "pass2e_remote":
        return {
            "resume_factuel": "",
            "points_cles": [],
            "actions": [],
            "desaccords": [],
            "documents_demandes": [],
            "elements_techniques": [],
        }

    if canonical == "pass3e_remote":
        return {
            "numero": None,
            "titre": "",
            "localisation": "",
            "description": "",
            "avis_participants": [],
            "synthese_echanges": "",
            "conclusion_expert": "Fallback adapter Pass3E: la sortie du modele etait vide, invalide ou impossible a normaliser.",
        }

    # Passe 2B / global report
    if canonical in ("report_remote", "report_remote_alt", "report_remote_alt2"):
        return {
            "resume_global": "",
            "themes": [],
            "themes_abordes": [],
            "actions": [],
            "perspectives": [],
            "demandes_documents_globales": [],
            "problems": [],
        }

    if canonical == "report_debrief_remote":
        return {
            "mode_debrief": "complement",
            "sujets": [],
            "demandes_documents_hors_sujet": [],
            "global_debrief": {
                "resume": "",
                "ordre_du_jour": [],
                "themes_abordes": [],
                "actions": [],
                "perspectives": [
                    {
                        "probleme": "adapter_fallback",
                        "solution": "La sortie debrief du LLM n'a pas pu etre normalisee par openai-adapter."
                    }
                ],
                "annexes": [],
            },
        }

    # Passe 3A
    if canonical in ("pass3a_remote", "pass3a_remote_alt", "pass3a_remote_alt2"):
        return {"date": None, "link": None, "resume": "", "ordre_du_jour": []}

    # Passe 3B
    if canonical in ("pass3b_remote", "pass3b_remote_alt", "pass3b_remote_alt2"):
        return {"themes_abordes": []}

    # Passe 3C
    if canonical in ("pass3c_remote", "pass3c_remote_alt", "pass3c_remote_alt2"):
        return {"actions": [], "perspectives": [], "annexes": []}

    # Passe 3D
    if canonical in ("pass3d_remote", "pass3d_remote_alt", "pass3d_remote_alt2"):
        return {"demandes_documents_globales": []}

    # Si vous gardez pass3_remote (modèle “fourre-tout”), fallback neutre
    # MAIS il ne peut pas respecter la règle "aucun champ supplémentaire" de 3C,
    # donc je recommande de ne plus l'utiliser pour 3A/3B/3C/3D.
    if canonical in ("pass3_remote", "pass3_remote_alt", "pass3_remote_alt2"):
        return {"text": ""}

    return {"text": ""}


def normalize_segment_annotation(parsed: Any, raw_out: str = "") -> Dict[str, Any]:
    """
    Normalise la sortie pour la route /annoter_segments (Passe 1).
    - Si parsed est une liste, on prend le premier élément.
    - Si ce n'est pas un dict, on part d'un squelette vide.
    - On force la présence des 4 clés, avec les bons types.
    """


    # Si le modèle renvoie une liste de segments, on prend le premier
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]

    if not isinstance(parsed, dict):
        parsed = {}

    out: Dict[str, Any] = {}

    # resume_segment : string
    val_res = parsed.get("resume_segment", "")
    if not isinstance(val_res, str):
        val_res = str(val_res) if val_res is not None else ""
    out["resume_segment"] = val_res.strip()

    # themes : liste
    val_themes = parsed.get("themes", [])
    if not isinstance(val_themes, list):
        val_themes = []
    out["themes"] = val_themes

    # actions : liste
    val_actions = parsed.get("actions", [])
    if not isinstance(val_actions, list):
        val_actions = []
    out["actions"] = val_actions

    # problems : liste
    val_problems = parsed.get("problems", [])
    if not isinstance(val_problems, list):
        val_problems = []
    out["problems"] = val_problems

    return out


SEGMENT_KEYS = {"resume_segment", "themes", "actions", "problems"}

def _is_segment_like(obj: Any) -> bool:
    return isinstance(obj, dict) and any(k in obj for k in SEGMENT_KEYS)

# -----------------------------------------------------------------------------
# Report schema helpers (Passe 2B / Passe 3)
# -----------------------------------------------------------------------------

def _is_report_like(obj: Any) -> bool:
    # Assez souple : dict + au moins une clé de report
    return isinstance(obj, dict) and any(k in obj for k in REPORT_KEYS)

def normalize_report_annotation(parsed: Any) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        parsed = {}

    def _as_list(x):
        if isinstance(x, list):
            return x
        # si c'est un dict/str/etc, on peut décider de le garder comme 1 item
        # mais restons simple: liste vide
        return []

    rg = parsed.get("resume_global", "")
    if not isinstance(rg, str):
        rg = str(rg) if rg is not None else ""
    rg = rg.strip()

    return {
        "resume_global": rg,
        "themes": _as_list(parsed.get("themes")),
        "themes_abordes": _as_list(parsed.get("themes_abordes")),
        "actions": _as_list(parsed.get("actions")),
        "perspectives": _as_list(parsed.get("perspectives")),
        "demandes_documents_globales": _as_list(parsed.get("demandes_documents_globales")),
        "problems": _as_list(parsed.get("problems")),
    }


DEBRIEF_ROOT_KEYS = {
    "mode_debrief",
    "sujets",
    "demandes_documents_hors_sujet",
    "global_debrief",
}


def _is_debrief_model(canonical: str) -> bool:
    return canonical == "report_debrief_remote"


PASS2E_KEYS = {
    "resume_factuel",
    "points_cles",
    "actions",
    "desaccords",
    "documents_demandes",
    "elements_techniques",
}


def _is_pass2e_model(canonical: str) -> bool:
    return canonical == "pass2e_remote"


def normalize_pass2e_compact(parsed: Any) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("pass2e payload must be a JSON object")

    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            t = value.strip()
            return [t] if t else []
        if isinstance(value, list):
            out = []
            for item in value:
                if item is None:
                    continue
                s = str(item).strip()
                if s:
                    out.append(s)
            return out
        s = str(value).strip()
        return [s] if s else []

    resume = parsed.get("resume_factuel", parsed.get("resume_segment", ""))
    if not isinstance(resume, str):
        resume = str(resume) if resume is not None else ""

    return {
        "resume_factuel": resume.strip(),
        "points_cles": _as_list(parsed.get("points_cles", parsed.get("themes"))),
        "actions": _as_list(parsed.get("actions")),
        "desaccords": _as_list(parsed.get("desaccords", parsed.get("problems"))),
        "documents_demandes": _as_list(parsed.get("documents_demandes")),
        "elements_techniques": _as_list(parsed.get("elements_techniques")),
    }


def _is_effectively_empty_pass2e(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return True
    if (obj.get("resume_factuel") or "").strip():
        return False
    for key in ("points_cles", "actions", "desaccords", "documents_demandes", "elements_techniques"):
        if isinstance(obj.get(key), list) and len(obj.get(key)) > 0:
            return False
    return True


PASS3E_KEYS = {
    "numero",
    "titre",
    "localisation",
    "description",
    "avis_participants",
    "synthese_echanges",
    "conclusion_expert",
}


def _is_pass3e_model(canonical: str) -> bool:
    return canonical == "pass3e_remote"


def normalize_pass3e_synthesis(parsed: Any) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("pass3e payload must be a JSON object")

    def _as_text(value):
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = [_as_text(item) for item in value]
            return "\n\n".join(part for part in parts if part)
        if isinstance(value, dict):
            for key in ("resume", "texte", "text", "content", "avis", "commentaire"):
                if key in value:
                    t = _as_text(value.get(key))
                    if t:
                        return t
            return json.dumps(value, ensure_ascii=False)
        return str(value).strip()

    def _as_participants(value):
        if value is None:
            return []
        raw_items = value if isinstance(value, list) else [value]
        out = []
        for item in raw_items:
            if item is None:
                continue
            if isinstance(item, str):
                t = item.strip()
                if t:
                    out.append({"nom": "", "role": "", "resume": t})
                continue
            if isinstance(item, dict):
                nom = _as_text(item.get("nom", item.get("name", item.get("participant", ""))))
                role = _as_text(item.get("role", item.get("qualite", item.get("fonction", ""))))
                resume = _as_text(item.get("resume", item.get("avis", item.get("texte", item.get("commentaire", item.get("position", ""))))))
                if nom or role or resume:
                    out.append({"nom": nom, "role": role, "resume": resume})
                continue
            t = _as_text(item)
            if t:
                out.append({"nom": "", "role": "", "resume": t})
        return out

    numero_raw = parsed.get("numero", parsed.get("num", parsed.get("sujet_numero")))
    numero = numero_raw
    if isinstance(numero_raw, str):
        t = numero_raw.strip()
        try:
            numero = int(t) if t else None
        except ValueError:
            numero = t

    synthese = _as_text(parsed.get("synthese_echanges", parsed.get("synthese", parsed.get("resume", parsed.get("resume_factuel")))))
    conclusion = _as_text(parsed.get("conclusion_expert", parsed.get("conclusion", parsed.get("avis_expert"))))

    return {
        "numero": numero,
        "titre": _as_text(parsed.get("titre", parsed.get("title"))),
        "localisation": _as_text(parsed.get("localisation", parsed.get("location", parsed.get("lieu")))),
        "description": _as_text(parsed.get("description", parsed.get("objet", parsed.get("contexte")))),
        "avis_participants": _as_participants(parsed.get("avis_participants", parsed.get("participants", parsed.get("avis")))),
        "synthese_echanges": synthese,
        "conclusion_expert": conclusion,
    }


def _is_effectively_empty_pass3e(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return True
    if obj.get("avis_participants"):
        return False
    for key in ("synthese_echanges", "conclusion_expert"):
        if isinstance(obj.get(key), str) and obj.get(key).strip():
            return False
    return True
def normalize_debrief_annotation(parsed: Any) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("debrief payload must be a JSON object")

    if REPORT_KEYS.intersection(parsed.keys()) and not DEBRIEF_ROOT_KEYS.intersection(parsed.keys()):
        raise ValueError("report_annotation schema is not a valid debrief payload")

    def _as_list(value):
        return value if isinstance(value, list) else []

    def _as_dict(value):
        return value if isinstance(value, dict) else {}

    mode = parsed.get("mode_debrief", "complement")
    if isinstance(mode, bool):
        mode = "complement" if mode else "substitution"
    mode = str(mode).strip().lower() if mode is not None else "complement"
    if mode not in ("complement", "substitution"):
        mode = "complement"

    sujets = []
    for item in _as_list(parsed.get("sujets")):
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        normalized["demandes_documents"] = _as_list(normalized.get("demandes_documents"))
        sujets.append(normalized)

    global_debrief = _as_dict(parsed.get("global_debrief"))
    resume = global_debrief.get("resume", "")
    if not isinstance(resume, str):
        resume = str(resume) if resume is not None else ""

    normalized_global = dict(global_debrief)
    normalized_global["resume"] = resume.strip()
    normalized_global["ordre_du_jour"] = _as_list(global_debrief.get("ordre_du_jour"))
    normalized_global["themes_abordes"] = _as_list(global_debrief.get("themes_abordes"))
    normalized_global["actions"] = _as_list(global_debrief.get("actions"))
    normalized_global["perspectives"] = _as_list(global_debrief.get("perspectives"))
    normalized_global["annexes"] = _as_list(global_debrief.get("annexes"))

    return {
        "mode_debrief": mode,
        "sujets": sujets,
        "demandes_documents_hors_sujet": _as_list(parsed.get("demandes_documents_hors_sujet")),
        "global_debrief": normalized_global,
    }

def _is_effectively_empty_report(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return True
    if (obj.get("resume_global") or "").strip():
        return False
    for k in ("themes", "themes_abordes", "actions", "perspectives", "demandes_documents_globales", "problems"):
        if isinstance(obj.get(k), list) and len(obj.get(k)) > 0:
            return False
    return True



def _is_effectively_empty_debrief(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return True
    if obj.get("sujets"):
        return False
    if obj.get("demandes_documents_hors_sujet"):
        return False
    gd = obj.get("global_debrief")
    if isinstance(gd, dict):
        if (gd.get("resume") or "").strip():
            return False
        for key in ("ordre_du_jour", "themes_abordes", "actions", "perspectives", "annexes"):
            if isinstance(gd.get(key), list) and len(gd.get(key)) > 0:
                return False
    return True
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

def _extract_first_json(text: str) -> str | None:
    """
    Extrait le premier JSON valide (objet OU tableau) présent dans un texte.
    Gère les code fences ```json ... ```.
    Retourne une chaîne JSON compacte, sinon None.
    """
    if not text:
        return None

    t = text.strip()

    # 1) si code fence, on tente d'abord son contenu
    m = _FENCE_RE.search(t)
    if m:
        inside = m.group(1).strip()
        s = _extract_first_json(inside)
        if s is not None:
            return s

    dec = json.JSONDecoder()

    # 2) cas direct : tout le contenu est du JSON
    try:
        obj = json.loads(t)
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        pass

    # 3) recherche du premier '{' ou '[' et raw_decode à partir de là
    starts = [i for i in (t.find("{"), t.find("[")) if i != -1]
    if not starts:
        return None
    i = min(starts)

    # Avance de proche en proche vers le prochain '{' ou '[' si échec
    while i != -1 and i < len(t):
        try:
            obj, end = dec.raw_decode(t[i:])
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            # cherche le prochain début JSON potentiel
            j1 = t.find("{", i + 1)
            j2 = t.find("[", i + 1)
            candidates = [j for j in (j1, j2) if j != -1]
            i = min(candidates) if candidates else -1

    return None


def _extract_first_json_object(text: str) -> str | None:
    """
    Tente de récupérer un objet JSON { ... } depuis une réponse qui pourrait contenir
    du texte avant/après. Retourne un JSON string valide, sinon None.
    """
    if not text:
        return None

    # cas simple : déjà JSON
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        try:
            json.loads(t)
            return t
        except Exception:
            pass

    # recherche d'un bloc { ... } (greedy, puis on rétrécit si besoin)
    start = t.find("{")
    if start == -1:
        return None

    # on tente des suffixes décroissants pour trouver un JSON parsable
    for end in range(len(t), start + 1, -1):
        if t[end - 1] != "}":
            continue
        candidate = t[start:end]
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            continue

    return None


#-----------------------------------------------------------
async def _remote_responses_native(
    *,
    requested_model: str,
    canonical_model: str,
    input_value: Any,
    instructions: str | None = None,
    tools: list | None = None,
    tool_choice: Any | None = None,
    reasoning: Any | None = None,
    max_output_tokens: int | None = None,
    parallel_tool_calls: bool | None = None,
    metadata: dict | None = None,
) -> dict:
    cfg = _model_cfg(canonical_model)
    base0 = (cfg.get("base_url") or cfg.get("api_base") or "").rstrip("/")
    if not base0:
        raise HTTPException(502, f"Remote config missing base_url/api_base for model '{canonical_model}'")
    physical_model, provider_model, explicit_provider_model = _remote_model_ids(canonical_model, cfg, base0)
    if not explicit_provider_model and provider_model != canonical_model:
        cfg.update(_remote_overrides(provider_model))
    cfg_phys = (_REMOTE_CONF.get("models") or {}).get(provider_model, {})
    if not explicit_provider_model and cfg_phys:
        cfg.update(cfg_phys)
    base = (cfg.get("base_url") or cfg.get("api_base") or "").rstrip("/")
    physical_model, provider_model, explicit_provider_model = _remote_model_ids(canonical_model, cfg, base)
    api_env = cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(api_env, "").strip()
    if not base or not api_key:
        raise HTTPException(502, f"Remote config/keys missing for model '{canonical_model}' (base={base}, env={api_env})")

    payload: dict[str, Any] = {
        "model": provider_model,
        "input": _responses_native_input(input_value),
    }
    if instructions:
        payload["instructions"] = str(instructions)
    native_tools = _responses_tools_to_native(tools, requested_model)
    if native_tools:
        payload["tools"] = native_tools
    native_tool_choice = _responses_tool_choice_to_native(tool_choice)
    if native_tool_choice is not None and _native_responses_should_send_tool_choice(cfg, reasoning):
        payload["tool_choice"] = native_tool_choice
    elif native_tool_choice is not None:
        log.debug("[remote_responses_native] omitting tool_choice because thinking mode does not support it requested_model=%s", requested_model)
    if reasoning is not None:
        payload["reasoning"] = reasoning
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
    if parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = bool(parallel_tool_calls)
    if metadata and cfg.get("supports_metadata", True):
        payload["metadata"] = metadata

    endpoint = f"{base}/responses"
    log.debug("[remote_responses_native] payload_summary=%s", _native_responses_payload_summary(
        endpoint=endpoint,
        requested_model=requested_model,
        provider_model=provider_model,
        payload=payload,
    ))
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    timeout_s = cfg.get("timeout")
    timeout = httpx.Timeout(timeout_s) if timeout_s else TIMEOUT_REMOTE
    async with httpx.AsyncClient(timeout=timeout) as client:
        t0 = time.monotonic()
        r = await client.post(endpoint, json=payload, headers=headers)
        elapsed = time.monotonic() - t0
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        body = ""
        try:
            body = e.response.text
        except Exception:
            pass
        log.error("Upstream native Responses HTTP %s endpoint=%s body=%s", e.response.status_code, endpoint, body[:2000])
        raise HTTPException(
            status_code=e.response.status_code,
            detail={"message": f"Upstream error {e.response.status_code}", "endpoint": endpoint, "body": body[:500]},
        )
    data = r.json() or {}
    log.info(
        "[remote_responses_native] done requested_model=%s provider_model=%s status=%s elapsed=%.3fs usage=%s",
        requested_model,
        provider_model,
        data.get("status") if isinstance(data, dict) else None,
        elapsed,
        data.get("usage") if isinstance(data, dict) else None,
    )
    if isinstance(data, dict):
        data = _normalize_native_response_model(data, requested_model)
        data.setdefault("output_text", _native_response_output_text(data))
    return data


async def _remote_responses_native_stream(
    *,
    requested_model: str,
    canonical_model: str,
    input_value: Any,
    instructions: str | None = None,
    tools: list | None = None,
    tool_choice: Any | None = None,
    reasoning: Any | None = None,
    max_output_tokens: int | None = None,
    parallel_tool_calls: bool | None = None,
    metadata: dict | None = None,
):
    cfg = _model_cfg(canonical_model)
    base0 = (cfg.get("base_url") or cfg.get("api_base") or "").rstrip("/")
    physical_model, provider_model, explicit_provider_model = _remote_model_ids(canonical_model, cfg, base0)
    if not explicit_provider_model and provider_model != canonical_model:
        cfg.update(_remote_overrides(provider_model))
    cfg_phys = (_REMOTE_CONF.get("models") or {}).get(provider_model, {})
    if not explicit_provider_model and cfg_phys:
        cfg.update(cfg_phys)
    base = (cfg.get("base_url") or cfg.get("api_base") or "").rstrip("/")
    physical_model, provider_model, explicit_provider_model = _remote_model_ids(canonical_model, cfg, base)
    api_env = cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(api_env, "").strip()
    if not base or not api_key:
        yield _sse_event("error", {"type": "error", "error": {"message": f"Remote config/keys missing for model '{canonical_model}'"}})
        return

    payload: dict[str, Any] = {
        "model": provider_model,
        "input": _responses_native_input(input_value),
        "stream": True,
    }
    if instructions:
        payload["instructions"] = str(instructions)
    native_tools = _responses_tools_to_native(tools, requested_model)
    if native_tools:
        payload["tools"] = native_tools
    native_tool_choice = _responses_tool_choice_to_native(tool_choice)
    if native_tool_choice is not None and _native_responses_should_send_tool_choice(cfg, reasoning):
        payload["tool_choice"] = native_tool_choice
    elif native_tool_choice is not None:
        log.debug("[remote_responses_native_stream] omitting tool_choice because thinking mode does not support it requested_model=%s", requested_model)
    if reasoning is not None:
        payload["reasoning"] = reasoning
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
    if parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = bool(parallel_tool_calls)
    if metadata and cfg.get("supports_metadata", True):
        payload["metadata"] = metadata

    endpoint = f"{base}/responses"
    log.debug("[remote_responses_native_stream] payload_summary=%s", _native_responses_payload_summary(
        endpoint=endpoint,
        requested_model=requested_model,
        provider_model=provider_model,
        payload=payload,
    ))
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    timeout_s = cfg.get("timeout")
    timeout = httpx.Timeout(timeout_s) if timeout_s else TIMEOUT_REMOTE
    event_name = "message"
    data_lines: list[str] = []

    def flush_event():
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = "message"
            return None
        raw_data = "\n".join(data_lines)
        event = event_name or "message"
        event_name = "message"
        data_lines = []
        if raw_data.strip() == "[DONE]":
            return None
        try:
            payload_obj = json.loads(raw_data)
        except Exception:
            payload_obj = {"type": event, "data": raw_data}
        if isinstance(payload_obj, dict):
            return _native_response_sse_event(event, payload_obj, requested_model)
        return _sse_event(event, {"type": event, "data": payload_obj})

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", endpoint, json=payload, headers=headers) as resp:
                if resp.status_code >= 400:
                    body = ""
                    try:
                        body = (await resp.aread()).decode("utf-8", errors="replace")
                    except Exception:
                        pass
                    yield _sse_event("error", {"type": "error", "error": {"status_code": resp.status_code, "endpoint": endpoint, "body": body[:500]}})
                    return
                async for line in resp.aiter_lines():
                    line = (line or "").rstrip("\r")
                    if not line:
                        event = flush_event()
                        if event:
                            yield event
                        continue
                    if line.startswith("event:"):
                        event_name = line[6:].strip() or "message"
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                        continue
                event = flush_event()
                if event:
                    yield event
    except asyncio.CancelledError:
        log.debug("[remote_responses_native_stream] client disconnected requested_model=%s", requested_model)
        raise
    except Exception as exc:
        log.warning("[remote_responses_native_stream] error requested_model=%s err=%s", requested_model, exc)
        yield _sse_event("error", {"type": "error", "error": {"message": str(exc)}})

async def _remote_chat(
    messages: list[dict],
    model: str,
    temperature: float | None,
    response_format: dict | None = None,
    max_tokens_override: int | None = None,
    source_route: str = "chat/completions",
    tools: list[dict] | None = None,
    tool_choice: Any | None = None,
    reasoning: Any | None = None,
    return_raw_response: bool = False,
) -> Any:
    # 1) cfg + modèle physique etc. (gardez votre code existant jusqu'à prov/is_modern)

    cfg = _model_cfg(model)
    base0 = (cfg.get("base_url") or cfg.get("api_base") or "").rstrip("/")
    if not base0:
        raise HTTPException(502, f"Remote config missing base_url/api_base for model '{model}'")

    physical_model, provider_model, explicit_provider_model = _remote_model_ids(model, cfg, base0)
    # Ajout : overrides REMOTE_CONF par modele normalise (si different), sauf si
    # provider_model fixe explicitement le nom attendu par le fournisseur.
    if not explicit_provider_model and provider_model != model:
        cfg.update(_remote_overrides(provider_model))

    cfg_phys = (_REMOTE_CONF.get("models") or {}).get(provider_model, {})
    if not explicit_provider_model and cfg_phys:
        cfg.update(cfg_phys)

    base = (cfg.get("base_url") or cfg.get("api_base") or "").rstrip("/")
    physical_model, provider_model, explicit_provider_model = _remote_model_ids(model, cfg, base)
    api_env = cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(api_env, "").strip()
    if not base or not api_key:
        raise HTTPException(502, f"Remote config/keys missing for model '{model}' (base={base}, env={api_env})")

    prov = _provider_name(base)
    is_modern = provider_model.startswith(("gpt-5", "o1", "o3", "o4"))

    use_responses_flag = cfg.get("use_responses_api", USE_RESPONSES)
    force_chat_flag    = cfg.get("force_chat", FORCE_CHAT)


    log.info(
        "[remote_chat] route=%s requested_model=%s physical_model=%s provider_model=%s base_url=%s api_key_env=%s tools=%s tool_choice=%s",
        source_route, model, physical_model, provider_model, base, api_env, bool(tools), tool_choice,
    )

    # 2) flags (UNE SEULE FOIS)
    use_responses_flag = cfg.get("use_responses_api", USE_RESPONSES)
    force_chat_flag = cfg.get("force_chat", FORCE_CHAT)

    use_resp = (not force_chat_flag) and bool(use_responses_flag) and (prov == "openai" and is_modern)

    # 3) paramètres communs
    temp = temperature if temperature is not None else cfg.get("temperature")
    top_p = cfg.get("top_p")
    max_tokens = max_tokens_override if max_tokens_override is not None else cfg.get("max_tokens")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if "openrouter.ai" in base:
        headers.setdefault("HTTP-Referer", "https://openwebui.local")
        headers.setdefault("X-Title", "Adapter Bridge")

    if cfg.get("embedding_model"):
        raise HTTPException(400, f"Model '{model}' is an embedding model; call /v1/embeddings instead.")

    timeout_s = cfg.get("timeout")
    timeout = httpx.Timeout(timeout_s) if timeout_s else TIMEOUT_REMOTE

    async with httpx.AsyncClient(timeout=timeout) as client:
        # 4) chemin Responses API
        if use_resp:
            instr, input_msgs = _messages_to_responses_payload(messages)

            p = {"model": provider_model, "input": input_msgs}
            if instr:
                p["instructions"] = instr

            # JSON mode pour Responses API (si vous le voulez systématique)
            if cfg.get("json_mode", False):
                p["text"] = {"format": {"type": "json_object"}}

            # sampling : sur certains modèles "modern", température/top_p peuvent être refusés
            include_sampling = not is_modern
            if include_sampling:
                if temp is not None:
                    p["temperature"] = temp
                if top_p is not None:
                    p["top_p"] = top_p

            if max_tokens is not None:
                # côté Responses API, OpenAI attend généralement max_output_tokens
                # (si vous utilisez max_completion_tokens ailleurs, adaptez ici de façon cohérente)
                p["max_output_tokens"] = max_tokens

            endpoint = f"{base}/responses"
            log.debug("[remote_chat] route=%s endpoint=%s keys=%s", source_route, endpoint, sorted(p.keys()))
            upstream_t0 = time.monotonic()
            r = await client.post(endpoint, json=p, headers=headers)
            upstream_elapsed = time.monotonic() - upstream_t0

            # retry si 400 sur sampling
            if r.status_code == 400 and include_sampling:
                try:
                    err = r.json().get("error", {}) or {}
                    msg = (err.get("message") or "").lower()
                    if err.get("param") in ("temperature", "top_p") or "temperature" in msg or "top_p" in msg:
                        p.pop("temperature", None)
                        p.pop("top_p", None)
                        log.debug("Retry /responses without temperature/top_p keys=%s", sorted(p.keys()))
                        upstream_t0 = time.monotonic()
                        r = await client.post(endpoint, json=p, headers=headers)
                        upstream_elapsed = time.monotonic() - upstream_t0
                except Exception:
                    pass
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = ""
                try:
                    body = e.response.text
                except Exception:
                    pass
                log.error("Upstream HTTP %s endpoint=%s body=%s", e.response.status_code, endpoint, body[:2000])
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail={
                        "message": f"Upstream error {e.response.status_code}",
                        "endpoint": endpoint,
                        "body": body[:500],
                    },
                )
            j = r.json() or {}
            usage = j.get("usage") if isinstance(j, dict) else None
            log.info(
                "[remote_chat] upstream responses done logical=%s physical=%s status=%s elapsed=%.3fs usage=%s",
                model,
                provider_model,
                j.get("status"),
                upstream_elapsed,
                usage,
            )

            if j.get("status") == "incomplete":
                raise HTTPException(status_code=502, detail="OpenAI Responses returned status=incomplete")

            out = []
            for item in j.get("output", []):
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            out.append(c.get("text", ""))
            return "\n".join(out) if out else (j.get("output_text") or str(j))

        # 5) chemin Chat Completions
        payload = {"model": provider_model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if reasoning is not None and cfg.get("supports_reasoning"):
            payload["reasoning"] = reasoning

        # tokens
        if max_tokens is not None:
            if prov == "openai" and is_modern:
                payload["max_completion_tokens"] = max_tokens
            else:
                payload["max_tokens"] = max_tokens

        # sampling : ne pas envoyer sur openai+modern
        include_sampling = not (prov == "openai" and is_modern)
        if include_sampling:
            if temp is not None:
                payload["temperature"] = temp
            if top_p is not None:
                payload["top_p"] = top_p


        # json mode / structured outputs
        if prov == "openai":
            if response_format is not None:
                payload["response_format"] = _stricten_response_format(response_format)
                rf_name = ((payload["response_format"].get("json_schema") or {}).get("name")
                           if isinstance(payload["response_format"], dict) else None)
                log.info(
                    "[remote_chat] response_format logical=%s physical=%s name=%s marker=%s",
                    model,
                    provider_model,
                    rf_name,
                    _response_format_marker(payload["response_format"]),
                )
            elif cfg.get("json_mode", False):
                payload["response_format"] = {"type": "json_object"}


        endpoint = f"{base}/chat/completions"
        log.debug("[remote_chat] route=%s endpoint=%s keys=%s payload_summary=%s", source_route, endpoint, sorted(payload.keys()), _chat_payload_debug_summary(
            requested_model=model,
            provider_model=provider_model,
            messages=messages,
            payload=payload,
            endpoint=endpoint,
        ))
        _log_deepseek_tools_payload(
            route="remote_chat",
            requested_model=model,
            provider_model=provider_model,
            endpoint=endpoint,
            payload=payload,
        )
        upstream_t0 = time.monotonic()
        r = await client.post(endpoint, json=payload, headers=headers)
        upstream_elapsed = time.monotonic() - upstream_t0
        log.debug("[remote_chat] payload_done=%s", _chat_payload_debug_summary(
            requested_model=model,
            provider_model=provider_model,
            messages=messages,
            payload=payload,
            endpoint=endpoint,
            duration=upstream_elapsed,
        ))

        # Retry automatique si 400 sur sampling (temp/top_p)
        if r.status_code == 400 and prov == "openai":
            try:
                err = (r.json() or {}).get("error", {}) or {}
                msg = (err.get("message") or "").lower()
                p = (err.get("param") or "").lower()

                if p in ("temperature", "top_p") or "temperature" in msg or "top_p" in msg:
                    payload.pop("temperature", None)
                    payload.pop("top_p", None)
                    log.debug(
                        "Retry /chat/completions without temperature/top_p keys=%s",
                        sorted(payload.keys())
                    )
                    upstream_t0 = time.monotonic()
                    r = await client.post(endpoint, json=payload, headers=headers)
                    upstream_elapsed = time.monotonic() - upstream_t0
            except Exception:
                pass

        # ✅ IMPORTANT : ce bloc doit être hors du if, sinon pas de parse quand status != 400
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = ""
            try:
                body = e.response.text
            except Exception:
                pass
            log.error("Upstream HTTP %s endpoint=%s body=%s", e.response.status_code, endpoint, body[:2000])
            raise HTTPException(
                status_code=e.response.status_code,
                detail={
                    "message": f"Upstream error {e.response.status_code}",
                    "endpoint": endpoint,
                    "body": body[:500],
                },
            )
        j = r.json() or {}

        if return_raw_response:
            return j

        content = None
        finish_reason = ""
        if isinstance(j.get("choices"), list) and j["choices"]:
            ch0 = j["choices"][0] or {}
            msg0 = ch0.get("message") or {}
            finish_reason = str(ch0.get("finish_reason") or "").strip().lower()
            content = msg0.get("content")
            if content is None:
                content = ch0.get("text")
        usage = j.get("usage") if isinstance(j, dict) else None
        log.info(
            "[remote_chat] upstream chat done logical=%s physical=%s elapsed=%.3fs finish_reason=%s usage=%s",
            model,
            provider_model,
            upstream_elapsed,
            finish_reason,
            usage,
        )

        content_str = "" if content is None else str(content)

        if finish_reason == "length" and not content_str.strip():
            log.warning(
                "[remote_chat] upstream truncated with empty content model=%s finish_reason=%s",
                model,
                finish_reason,
            )
            raise HTTPException(
                status_code=502,
                detail="Upstream truncated: finish_reason=length with empty content",
            )

        if content is None:
            return str(j)

        content = content_str

        # règle spéciale "pass3_" : forcer JSON
        if model.startswith("pass3_"):
            json_only = _extract_first_json_object(content)
            if json_only is None:
                log.warning("Pass3 invalid JSON. First 800 chars: %r", content[:800])
                raise HTTPException(status_code=502, detail="Pass3: upstream did not return valid JSON")
            return json_only

        return content



@app.on_event("startup")
async def _startup_discovery():
    global LOCAL_MODELS, _AVAILABLE_LOCAL_IDS
    if getattr(app.state, "discovered", False):
        return

    await configure_local_backend_from_candidates()

    # 1) attendre que le Flask réponde à /ping (ou /models à défaut)
    await _wait_local_ready()  # ← nouvelle fonction ci-dessous

    # 2) tenter la découverte (avec retries)
    kept, available_ids = await _discover_local_models_with_ids()
    if kept:
        LOCAL_MODELS = kept
        _AVAILABLE_LOCAL_IDS = available_ids
        log.info("✅ LOCAL_MODELS après découverte: %s", ", ".join(LOCAL_MODELS))
    else:
        log.warning("⚠️ Découverte vide: conservation de LOCAL_MODELS=%s", LOCAL_MODELS)

    app.state.discovered = True

async def _wait_local_ready():
    deadline = time.time() + WAIT_READY_SECS
    url_ping = f"{LOCAL_BASE}{LOCAL_PING_PATH}"
    headers = _llm_headers()
    last_err = None
    while time.time() < deadline:
        try:
            async with httpx.AsyncClient(timeout=PING_READ_TIMEOUT) as c:
                r = await c.get(url_ping, headers=headers)
                if r.status_code == 200:
                    log.info("🟢 Local LLM prêt: %s", url_ping)
                    return
        except Exception as e:
            last_err = e
        await asyncio.sleep(WAIT_SLEEP_SEC)
    log.warning("⏱️ Attente readiness expirée (%ss): %s (%s)", WAIT_READY_SECS, url_ping, last_err)


async def _discover_local_models_with_ids() -> tuple[list[str], set[str]]:
    if not DISCOVER_LOCAL_MODELS:
        return [], set()

    url = f"{LOCAL_BASE}{LOCAL_DISCOVERY_PATH}"  # ex: /models
    headers = _llm_headers()  # inclut x-api-key si nécessaire

    # Timeout de découverte = plus large qu’un simple ping
    timeout = httpx.Timeout(
        connect=CONNECT_TIMEOUT,
        read=READ_TIMEOUT,
        write=READ_TIMEOUT,
        pool=CONNECT_TIMEOUT,
    )

    attempts = 5
    last_err = None
    j = None

    for i in range(1, attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.get(url, headers=headers)

                r.raise_for_status()
                j = r.json()
            break
        except Exception as e:
            last_err = e
            log.warning("Découverte des modèles: tentative %d/%d échouée (%s)", i, attempts, e)
            await asyncio.sleep(WAIT_SLEEP_SEC)

    if j is None:
        log.warning("Découverte des modèles: échec (%s)", last_err)
        return [], set()

    # Formats possibles
    if isinstance(j, dict) and isinstance(j.get("data"), list):
        available_ids = {str(d.get("id")) for d in j["data"] if isinstance(d, dict) and d.get("id")}
    elif isinstance(j, dict):
        available_ids = set(j.keys())
    else:
        available_ids = set()

    kept = [alias for alias, target in MODEL_ALIAS.items() if target in available_ids]
    for alias, target in MODEL_ALIAS.items():
        if alias not in kept:
            log.warning("🧹 LOCAL_MODELS: retrait de '%s' (cible '%s' introuvable sur Flask)", alias, target)

    return kept, available_ids

async def _ensure_local_model_info() -> dict[str, t.Any]:
    """
    Charge /model_info une fois, met à jour LOCAL_MODEL_INFO et DEFAULT_LOCAL_N_CTX,
    avec un cache TTL basé sur LOCAL_MODEL_INFO_TTL.
    """
    global LOCAL_MODEL_INFO, _LOCAL_MODEL_INFO_TS, DEFAULT_LOCAL_N_CTX

    now = time.monotonic()
    # Cache encore valide
    if LOCAL_MODEL_INFO and (now - _LOCAL_MODEL_INFO_TS) < LOCAL_MODEL_INFO_TTL:
        return LOCAL_MODEL_INFO

    if not LOCAL_BASE:
        # Pas de backend local configuré
        return LOCAL_MODEL_INFO

    url = f"{LOCAL_BASE}/model_info"
    headers = _llm_headers()

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_PING) as client:
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            info = r.json() or {}
    except Exception as e:
        log.warning("Impossible de charger model_info : %r", e)
        return LOCAL_MODEL_INFO

    if isinstance(info, dict):
        LOCAL_MODEL_INFO = info
        _LOCAL_MODEL_INFO_TS = now
        # Mise à jour automatique du n_ctx local si présent
        n_ctx = info.get("n_ctx")
        if isinstance(n_ctx, int):
            DEFAULT_LOCAL_N_CTX = n_ctx
        log.info("Model info loaded: %s", LOCAL_MODEL_INFO)

    return LOCAL_MODEL_INFO

def _resolve_registry(model_name: str) -> dict:
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    raise HTTPException(400, f"Unknown model id: {model_name}")



def _is_model_not_found(detail: str) -> bool:
    # detail est souvent le JSON texte renvoyé par OpenAI/OpenRouter
    d = (detail or "").lower()
    return ("model_not_found" in d) or ("does not exist" in d) or ("do not have access" in d)

async def _remote_chat_with_retry(
    messages,
    model_name,
    temperature,
    retries=2,
    base_delay=1.0,
    response_format=None,
):

    last_exc = None

    for attempt in range(1, retries + 2):
        attempt_t0 = time.monotonic()
        try:
            raw = await _remote_chat(
                messages,
                model_name,
                temperature,
                response_format=response_format,
            )
            log.info(
                "[remote_chat_with_retry] OK model=%s attempt=%s/%s elapsed=%.3fs",
                model_name,
                attempt,
                retries + 1,
                time.monotonic() - attempt_t0,
            )
            # IMPORTANT : raw vide => on force une erreur transitoire pour fallback/retry
            if _is_effectively_empty_raw(raw):
                raise HTTPException(status_code=502, detail="Empty completion (raw)")

            return raw

        # timeouts réseau
        except (httpx.TimeoutException, httpcore.ReadTimeout, httpcore.ConnectTimeout) as e:
            last_exc = e
            log.warning(
                "[remote_chat_with_retry] timeout model=%s attempt=%s/%s elapsed=%.3fs err=%s: %s",
                model_name,
                attempt,
                retries + 1,
                time.monotonic() - attempt_t0,
                type(e).__name__,
                e,
            )

        # IMPORTANT : ton _remote_chat() remonte des HTTPException(502, detail=...)
        except HTTPException as e:
            last_exc = e
            code = getattr(e, "status_code", None)
            detail = getattr(e, "detail", "") or ""

            # pas de retry sur erreurs “définitives”
            if code in (400, 401, 402, 403, 404):
                if code == 404 and _is_model_not_found(detail):
                    log.error(f"[annoter_segments] model not found / no access: model={model_name} detail={detail}")
                if code == 402:
                    log.error(f"[annoter_segments] payment/credits issue upstream 402 model={model_name} detail={str(detail)[:500]}")
                raise

            # retry seulement sur transitoires
            if code not in (429, 500, 502, 503, 504):
                raise

            log.warning(
                "[remote_chat_with_retry] transient HTTPException code=%s model=%s attempt=%s/%s elapsed=%.3fs detail=%s",
                code,
                model_name,
                attempt,
                retries + 1,
                time.monotonic() - attempt_t0,
                str(detail)[:200],
            )

        # éventuellement attraper les erreurs httpx non encapsulées (au cas où)
        except httpx.HTTPStatusError as e:
            last_exc = e
            sc = e.response.status_code if e.response is not None else None

            # Non retryable (crédits, clé, auth, etc.)
            if sc in (402, 403):
                log.error(f"[annoter_segments] non-retryable httpx {sc} model={model_name}: {e.response.text if e.response else ''}")
                raise

            # Retryable
            if sc in (429, 500, 502, 503, 504):
                log.warning(
                    "[remote_chat_with_retry] httpx status=%s model=%s attempt=%s/%s elapsed=%.3fs",
                    sc,
                    model_name,
                    attempt,
                    retries + 1,
                    time.monotonic() - attempt_t0,
                )
            else:
                raise


        # backoff avant le prochain essai (sauf si dernier)
        if attempt < retries + 1:
            # exponentiel + jitter
            delay = base_delay * (2 ** (attempt - 1))
            delay = delay * (0.8 + 0.4 * random.random())
            await asyncio.sleep(delay)

    # si on sort de la boucle, on remonte la dernière erreur
    raise last_exc if last_exc is not None else RuntimeError("remote_chat_with_retry: unknown failure")


# -----------------------------------------------------------------------------
# Local chat (vers Flask)
# -----------------------------------------------------------------------------
async def _local_chat(
    prompt: str,
    route_hint: str,
    temperature: float | None,
    meta: dict | None = None,
    app_id: str | None = None,
    conversation_id: str | None = None,
    messages: list[dict] | None = None,
    use_memory: bool = False,
) -> str:
    if not await _ensure_local_ready():
        raise HTTPException(status_code=502, detail="Local LLM unreachable after WOL attempt")

    meta = meta or {}


    path = "/chat_orchestre"
    url = f"{LOCAL_BASE}{path}"

    reg = MODEL_REGISTRY.get(route_hint, {}) if isinstance(MODEL_REGISTRY, dict) else {}
    real_model = reg.get("model") or MODEL_ALIAS.get(route_hint, route_hint)

    payload = {
        "prompt": prompt,
        "model": real_model,
        "model_name": real_model,
        "temperature": (temperature or 0.4),
    }
    payload.update(meta)

    # Toujours propager l’identité
    if app_id:
        payload["app_id"] = str(app_id)
    if conversation_id:
        payload["conversation_id"] = str(conversation_id)

    # memory_id seulement si mémoire Flask activée
    if use_memory and conversation_id:
        payload["memory_id"] = str(conversation_id)
    elif use_memory:
        payload.setdefault("memory_id", str(meta.get("memory_id") or "default"))

    # réglages
    payload.setdefault("top_k", int(meta.get("top_k") or 4))

    memory_turns = int(meta.get("memory_turns") or meta.get("n_turns") or 8)

    if use_memory:
        mem = _build_memory_append(messages or [], n_turns=memory_turns)
        if mem:
            payload["memory_append"] = mem
        payload.setdefault("memory_turns", memory_turns)


    headers = _llm_headers()
    if app_id:
        headers["x-app-id"] = str(app_id)
    if conversation_id:
        headers["x-conversation-id"] = str(conversation_id)

    try:
        info = await _ensure_local_model_info() or {}
        n_ctx = int(info.get("n_ctx", DEFAULT_LOCAL_N_CTX))
        max_tokens = int(info.get("max_tokens", DEFAULT_LOCAL_MAX_TOKENS))

        marge = int(meta.get("marge", DEFAULT_LOCAL_MARGE_TOKENS))
        min_prompt_tokens = int(meta.get("min_prompt_tokens", DEFAULT_LOCAL_MIN_PROMPT_TOKENS))

        msg = prompt or ""
        approx_tokens = len(msg) // 4
        available = max(n_ctx - max_tokens - marge, min_prompt_tokens)

        if approx_tokens > available:
            frac = available / float(approx_tokens)
            keep_chars = max(int(len(msg) * frac), 256)
            msg = msg[-keep_chars:]
            log.warning("[adapter][_local_chat] prompt tronqué: approx=%d → new≈%d, budget=%d, n_ctx=%d",
                        approx_tokens, len(msg)//4, available, n_ctx)

        payload["prompt"] = msg

        log.info(
            "[adapter][_local_chat] to_flask app_id=%s conv_id=%s memory_id=%s use_memory=%s",
            payload.get("app_id"), payload.get("conversation_id"), payload.get("memory_id"), use_memory
        )


        r = await _local_request_once_with_runtime_fallback("POST", path, timeout=TIMEOUT_LOCAL, json=payload, headers=headers)
        j = r.json()



        if isinstance(j, dict):
            for key in ("reponse_with_refs", "reponse", "result", "text", "output", "message", "content"):
                val = j.get(key)
                if isinstance(val, str) and val.strip():
                    return val
            return ""


    except (httpx.ReadTimeout, httpx.ConnectTimeout):
        r = await _local_request_once_with_runtime_fallback("POST", path, timeout=TIMEOUT_INIT, json=payload, headers=headers)
        j = r.json()
        if isinstance(j, dict):
            if meta.get("return_html"):
                return (j.get("reponse_with_refs_html")
                        or j.get("reponse_html")
                        or j.get("reponse_with_refs")
                        or j.get("reponse")
                        or j.get("result")
                        or j.get("text")
                        or str(j))
            return (j.get("reponse_with_refs")
                    or j.get("reponse")
                    or j.get("result")
                    or j.get("text")
                    or str(j))
        return str(j)

# -----------------------------------------------------------------------------
# Schemas OpenAI compat
# -----------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    stream: bool | None = None
    metadata: dict | None = None
    response_format: dict | None = None

class ChatRespChoice(BaseModel):
    index: int
    message: dict
    finish_reason: str = "stop"

class ChatResp(BaseModel):
    id: str = "cmpl_local"
    object: str = "chat.completion"
    model: str
    choices: list[ChatRespChoice]

class ResponsesReq(BaseModel):
    model: str
    input: Any
    instructions: str | None = None
    stream: bool | None = None
    max_output_tokens: int | None = None
    tools: list | None = None
    tool_choice: Any | None = None
    reasoning: Any | None = None
    parallel_tool_calls: bool | None = None
    previous_response_id: str | None = None
    metadata: dict | None = None

class EmbReq(BaseModel):
    model: str | None = None
    input: list[str] | str

class ImageGenReq(BaseModel):
    prompt: str
    n: int | None = 1
    size: str | None = "1024x1024"
    metadata: dict | None = None

class AnalyzeReq(BaseModel):
    file_urls: Optional[List[str]] = None
    model: str = "annoter_rag_vecteur"
    analysis_prompt: Optional[str] = None
    return_html: bool = False
    use_vector_rag: bool = True
    collection: Optional[str] = None
    vec_backend: Optional[str] = None
    metadata: Optional[dict] = None

class TranscribeReq(BaseModel):
    file_url: Optional[str] = None
    model: Optional[str] = None  # ignoré, compat OpenAI
    language: Optional[str] = None
    timestamps: Optional[bool] = None
    diarize: Optional[bool] = None
    chunk: Optional[int] = None
    stride: Optional[int] = None
    summarize: Optional[bool] = False
    summary_prompt: Optional[str] = None
    return_html: Optional[bool] = False
    llm_model: Optional[str] = "annoter_rag_vecteur"  # LLM par défaut pour résumé
    metadata: Optional[dict] = None
    speakers: Optional[int] = None
    min_speaker_duration: Optional[float] = None
    collar: Optional[float] = None
    allow_overlap: Optional[bool] = None

class SearchReq(BaseModel):
    query: str
    engine: Optional[str] = None        # "searxng", "brave", etc. (si tu routes vers plusieurs moteurs)
    num: Optional[int] = 10
    lang: Optional[str] = "fr"
    site: Optional[str] = None
    freshness: Optional[str] = None

# -----------------------------------------------------------------------------
# REST: Models
# -----------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models(authorization: t.Annotated[str | None, Header()] = None):
    _check_adapter_auth(authorization)
    data = []
    data += [{"id": m, "object": "model"} for m in LOCAL_MODELS]
    # Expose aussi les "modèles-route" (y.c. injectés via ENV)
    data += [{"id": k, "object": "model"} for k in LOCAL_ROUTE_MAP.keys()]
    data += [{"id": m, "object": "model"} for m in REMOTE_MODELS]
    data += [{"id": m, "object": "model"} for m in REMOTE2_MODELS]
    return {"object": "list", "data": data}



def _responses_observed_roles(input_value: Any) -> list[str]:
    roles: list[str] = []
    if isinstance(input_value, list):
        for item in input_value:
            if isinstance(item, dict) and item.get("type") in (None, "message") and item.get("role") is not None:
                roles.append(str(item.get("role")))
    return roles


def _responses_input_to_messages(
    input_value: Any,
    instructions: str | None = None,
    *,
    cfg: dict | None = None,
    requested_model: str | None = None,
) -> list[dict]:
    messages: list[dict] = []
    developer_conversions = 0
    preserve_developer = bool((cfg or {}).get("supports_developer_role"))
    if instructions:
        messages.append({"role": "system", "content": str(instructions)})

    if isinstance(input_value, str):
        if not input_value.strip():
            raise HTTPException(status_code=400, detail="Responses input string cannot be empty")
        messages.append({"role": "user", "content": input_value})
        return messages

    if not isinstance(input_value, list) or not input_value:
        raise HTTPException(status_code=400, detail="Responses input must be a string or a non-empty list")

    for item in input_value:
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="Responses input items must be objects")

        item_type = item.get("type")
        if item_type in (None, "message"):
            developer_conversions += _append_responses_message(
                messages,
                item,
                preserve_developer=preserve_developer,
            )
            continue
        if item_type == "function_call":
            messages.append(_responses_function_call_to_chat_message(item))
            continue
        if item_type == "function_call_output":
            messages.append(_responses_function_output_to_chat_message(item))
            continue
        raise HTTPException(status_code=400, detail=f"Responses input item type '{item_type}' is not supported")

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            "[responses_input] requested_model=%s input_item_types=%s roles=%s developer_to_system=%s preserve_developer=%s",
            requested_model,
            _responses_input_item_types(input_value),
            _responses_observed_roles(input_value),
            developer_conversions,
            preserve_developer,
        )
    return messages


def _usage_zero() -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _convert_openai_usage(usage: Any) -> dict:
    if not isinstance(usage, dict):
        return _usage_zero()
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    converted = {
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0,
        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0,
        "total_tokens": usage.get("total_tokens", 0) or 0,
    }
    cached_tokens = prompt_details.get("cached_tokens")
    reasoning_tokens = completion_details.get("reasoning_tokens")
    if cached_tokens is not None:
        converted["cached_tokens"] = cached_tokens
    if reasoning_tokens is not None:
        converted["reasoning_tokens"] = reasoning_tokens
    return converted


def _sse_event(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _responses_stream_headers() -> dict:
    return {"Cache-Control": "no-cache", "Connection": "keep-alive"}

def _json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _tool_schema_stats(tools: list | None) -> dict:
    return tool_compat.inspect_tools(
        tools,
        max_tools=MAX_TOOLS,
        max_schema_bytes=MAX_TOOL_SCHEMA_BYTES,
    )


def _tool_structure_summary(tools: list | None) -> list[dict]:
    return tool_compat.inspect_tool_structures(tools)


def _log_tool_schema_stats(route: str, requested_model: str | None, tools: list | None) -> None:
    if not log.isEnabledFor(logging.DEBUG):
        return
    stats = _tool_schema_stats(tools)
    log.debug(
        "[tool_schema] route=%s requested_model=%s tool_count=%s tools_size_bytes=%s "
        "max_tool_size_bytes=%s tool_types=%s max_tools=%s max_tool_schema_bytes=%s",
        route,
        requested_model,
        stats["tool_count"],
        stats["tools_size_bytes"],
        stats["max_tool_size_bytes"],
        stats["tool_types"],
        stats["max_tools"],
        stats["max_tool_schema_bytes"],
    )
    log.debug(
        "[tool_schema_structure] route=%s requested_model=%s tools=%s",
        route,
        requested_model,
        _tool_structure_summary(tools),
    )


def _tool_capabilities_from_cfg(cfg: dict, *, native_responses: bool = False) -> tool_compat.ProviderToolCapabilities:
    return tool_compat.capabilities_from_config(cfg, native_responses=native_responses)


def _tool_compat_http_error(exc: tool_compat.ToolCompatibilityError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def _safe_tool_name_part(value: Any) -> str:
    part = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "").strip())
    part = part.strip("_")
    if not part or not re.match(r"^[A-Za-z_]", part):
        part = f"tool_{part}" if part else "tool"
    return part


def _flattened_namespace_tool_name(namespace: str, name: str) -> str:
    base = f"{_safe_tool_name_part(namespace)}__{_safe_tool_name_part(name)}"
    if len(base) <= 64:
        return base
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:10]
    keep = max(1, 64 - len(digest) - 1)
    return f"{base[:keep]}_{digest}"


def _ensure_unique_tool_name(name: str, seen: set[str]) -> str:
    if name in seen:
        raise HTTPException(status_code=400, detail=f"Tool name collision after conversion: {name}")
    seen.add(name)
    return name


def _chat_function_tools_from_responses(tools: list, requested_model: str) -> list[tuple[dict, str | None]]:
    flattened: list[tuple[dict, str | None]] = []
    seen: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            raise HTTPException(status_code=400, detail="Responses tools must be objects")
        tool_type = tool.get("type")
        if tool_type == "function":
            name = _validate_function_name(tool.get("name"))
            _ensure_unique_tool_name(name, seen)
            flattened.append((tool, None))
            continue
        if tool_type == "namespace":
            namespace = _validate_function_name(tool.get("name"))
            subtools = tool.get("tools")
            if not isinstance(subtools, list) or not subtools:
                raise HTTPException(status_code=400, detail="Namespace tool must contain a non-empty tools list")
            for subtool in subtools:
                if not isinstance(subtool, dict):
                    raise HTTPException(status_code=400, detail="Namespace subtools must be objects")
                if subtool.get("type") != "function":
                    raise HTTPException(status_code=400, detail="Only function subtools are supported inside namespace tools")
                original_name = _validate_function_name(subtool.get("name"))
                flattened_name = _ensure_unique_tool_name(_flattened_namespace_tool_name(namespace, original_name), seen)
                flattened.append((subtool, flattened_name))
            continue
        if tool_type in ("web_search", "web_search_preview"):
            raise HTTPException(
                status_code=400,
                detail=f"Tool type '{tool_type}' cannot be transported faithfully to Chat Completions for model '{requested_model}'",
            )
        raise HTTPException(status_code=400, detail=f"Unsupported tool type '{tool_type}'")
    return flattened


def _tool_capability(cfg: dict, name: str) -> bool:
    return bool(cfg.get(name))


def _serialize_function_payload(value: Any, *, max_bytes: int, field_name: str) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            raise HTTPException(status_code=400, detail=f"{field_name} must be JSON serializable")
    if len(text.encode("utf-8")) > max_bytes:
        raise HTTPException(status_code=400, detail=f"{field_name} exceeds configured byte limit")
    return text


def _validate_function_name(name: Any) -> str:
    name_s = str(name or "")
    if not FUNCTION_NAME_RE.match(name_s):
        raise HTTPException(status_code=400, detail="Function tool name is invalid")
    return name_s


def _responses_tools_to_chat_result(tools: list | None, cfg: dict, requested_model: str, tool_choice: Any | None = None) -> tool_compat.ToolTranslationResult:
    if not tools:
        return tool_compat.ToolTranslationResult(tools=[])
    _log_tool_schema_stats("responses_to_chat", requested_model, tools)
    try:
        result = tool_compat.translate_tools_for_chat(
            tools,
            _tool_capabilities_from_cfg(cfg),
            requested_model=requested_model,
            max_tools=MAX_TOOLS,
            max_schema_bytes=MAX_TOOL_SCHEMA_BYTES,
            tool_choice=tool_choice,
        )
    except tool_compat.ToolCompatibilityError as exc:
        raise _tool_compat_http_error(exc)
    for warning in result.warnings:
        log.debug("[responses_tools] %s requested_model=%s", warning, requested_model)
    return result


def _responses_tools_to_chat_tools(tools: list | None, cfg: dict, requested_model: str) -> list[dict] | None:
    result = _responses_tools_to_chat_result(tools, cfg, requested_model)
    return result.tools or None

def _responses_tool_choice_to_chat(tool_choice: Any, cfg: dict) -> Any | None:
    if tool_choice is None:
        return None
    if tool_choice in ("auto", "none", "required"):
        return tool_choice
    if not isinstance(tool_choice, dict):
        raise HTTPException(status_code=400, detail="Unsupported tool_choice")
    if tool_choice.get("type") != "function":
        raise HTTPException(status_code=400, detail="Unsupported tool_choice type")
    fn = tool_choice.get("function")
    name = fn.get("name") if isinstance(fn, dict) else tool_choice.get("name")
    return {"type": "function", "function": {"name": _validate_function_name(name)}}


def _input_contains_tool_items(input_value: Any) -> bool:
    return isinstance(input_value, list) and any(isinstance(item, dict) and item.get("type") in ("function_call", "function_call_output") for item in input_value)

def _responses_has_tool_outputs(input_value: Any) -> bool:
    return isinstance(input_value, list) and any(isinstance(item, dict) and item.get("type") == "function_call_output" for item in input_value)


def _responses_tool_call_count(messages: list[dict]) -> int:
    return sum(len(m.get("tool_calls") or []) for m in messages if isinstance(m, dict))


def _responses_final_tool_result_turn(input_value: Any, tool_choice: Any) -> bool:
    return _responses_has_tool_outputs(input_value) and tool_choice in (None, "none")


def _responses_effective_max_tokens(requested_max: int | None, final_tool_result_turn: bool) -> int | None:
    if not final_tool_result_turn:
        return requested_max
    if requested_max is None:
        return 256
    return min(int(requested_max), 256)


def _chat_payload_debug_summary(
    *,
    requested_model: str,
    provider_model: str,
    messages: list[dict],
    payload: dict,
    endpoint: str,
    duration: float | None = None,
) -> dict:
    summary = {
        "requested_model": requested_model,
        "provider_model": provider_model,
        "message_count": len(messages),
        "roles": [m.get("role") for m in messages if isinstance(m, dict)],
        "tool_call_count": _responses_tool_call_count(messages),
        "has_tool_messages": any(isinstance(m, dict) and m.get("role") == "tool" for m in messages),
        "tools_present": "tools" in payload,
        "tool_choice": payload.get("tool_choice"),
        "max_tokens": payload.get("max_tokens", payload.get("max_completion_tokens")),
        "payload_size_bytes": len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")),
        "endpoint": endpoint,
    }
    if duration is not None:
        summary["duration"] = round(duration, 3)
    return summary


def _native_responses_enabled(cfg: dict) -> bool:
    return bool(cfg.get("use_responses_api")) and not bool(cfg.get("force_chat")) and bool(cfg.get("native_responses_provider"))


def _responses_input_item_types(input_value: Any) -> list[str]:
    if isinstance(input_value, str):
        return ["input_text"]
    if isinstance(input_value, list):
        item_types: list[str] = []
        for item in input_value:
            if isinstance(item, dict):
                item_types.append(str(item.get("type") or item.get("role") or "message"))
            else:
                item_types.append(type(item).__name__)
        return item_types
    return [type(input_value).__name__]


def _native_responses_payload_summary(
    *,
    endpoint: str,
    requested_model: str,
    provider_model: str,
    payload: dict,
) -> dict:
    reasoning = payload.get("reasoning")
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    return {
        "endpoint": endpoint,
        "requested_model": requested_model,
        "provider_model": provider_model,
        "input_item_types": _responses_input_item_types(payload.get("input")),
        "roles": _responses_observed_roles(payload.get("input")),
        "tools_present": "tools" in payload,
        "tool_choice": payload.get("tool_choice"),
        "reasoning_effort": effort,
        "stream": bool(payload.get("stream")),
        "payload_size_bytes": _json_bytes(payload),
    }


def _log_deepseek_tools_payload(
    *,
    route: str,
    requested_model: str,
    provider_model: str,
    endpoint: str,
    payload: dict,
) -> None:
    if "api.deepseek.com" not in endpoint or "tools" not in payload:
        return
    log.debug(
        "[%s] deepseek_tools_payload_summary=%s",
        route,
        _native_responses_payload_summary(
            endpoint=endpoint,
            requested_model=requested_model,
            provider_model=provider_model,
            payload=payload,
        ),
    )

def _append_responses_message(messages: list[dict], item: dict, *, preserve_developer: bool = False) -> int:
    role = item.get("role")
    if role not in ("system", "user", "assistant", "developer"):
        raise HTTPException(status_code=400, detail="Responses input messages require role system, developer, user or assistant")
    developer_converted = 0
    if role == "developer" and not preserve_developer:
        role = "system"
        developer_converted = 1
    content = item.get("content")
    if isinstance(content, str):
        messages.append({"role": role, "content": content})
        return developer_converted
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                raise HTTPException(status_code=400, detail="Responses content parts must be objects")
            part_type = part.get("type")
            if part_type not in ("input_text", "output_text", "text"):
                raise HTTPException(status_code=400, detail=f"Responses content part type '{part_type}' is not supported")
            parts.append(str(part.get("text") or ""))
        messages.append({"role": role, "content": "\n".join(parts)})
        return developer_converted
    raise HTTPException(status_code=400, detail="Responses input message content must be text")


def _responses_tools_to_native_result(tools: list | None, requested_model: str | None = None) -> tool_compat.ToolTranslationResult:
    if not tools:
        return tool_compat.ToolTranslationResult(tools=[])
    _log_tool_schema_stats("responses_native", requested_model, tools)
    try:
        result = tool_compat.translate_tools_for_native_responses(
            tools,
            _tool_capabilities_from_cfg({"native_responses_provider": True, "supports_strict_tools": True}, native_responses=True),
            max_tools=MAX_TOOLS,
            max_schema_bytes=MAX_TOOL_SCHEMA_BYTES,
        )
    except tool_compat.ToolCompatibilityError as exc:
        raise _tool_compat_http_error(exc)
    return result


def _responses_tools_to_native(tools: list | None, requested_model: str | None = None) -> list[dict] | None:
    result = _responses_tools_to_native_result(tools, requested_model)
    return result.tools or None

def _responses_tool_choice_to_native(tool_choice: Any) -> Any | None:
    if tool_choice is None:
        return None
    if tool_choice in ("auto", "none", "required"):
        return tool_choice
    if not isinstance(tool_choice, dict):
        raise HTTPException(status_code=400, detail="Unsupported tool_choice")
    if tool_choice.get("type") != "function":
        raise HTTPException(status_code=400, detail="Unsupported tool_choice type")
    fn = tool_choice.get("function")
    name = fn.get("name") if isinstance(fn, dict) else tool_choice.get("name")
    return {"type": "function", "name": _validate_function_name(name)}

def _native_responses_should_send_tool_choice(cfg: dict, reasoning: Any | None) -> bool:
    if cfg.get("supports_tool_choice_in_thinking") is not False:
        return True
    thinking_enabled = reasoning is not None or str(cfg.get("thinking_default") or "").lower() == "enabled"
    return not thinking_enabled


def _responses_native_input(input_value: Any) -> Any:
    if isinstance(input_value, str):
        if not input_value.strip():
            raise HTTPException(status_code=400, detail="Responses input string cannot be empty")
        return input_value
    if not isinstance(input_value, list) or not input_value:
        raise HTTPException(status_code=400, detail="Responses input must be a string or a non-empty list")
    native: list[Any] = []
    for item in input_value:
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="Responses input items must be objects")
        item_type = item.get("type")
        if item_type == "function_call":
            if not str(item.get("call_id") or ""):
                raise HTTPException(status_code=400, detail="function_call requires call_id")
            _validate_function_name(item.get("name"))
            if "arguments" in item:
                _serialize_function_payload(item.get("arguments") or "", max_bytes=MAX_FUNCTION_ARGUMENTS_BYTES, field_name="function arguments")
        if item_type == "function_call_output":
            if not str(item.get("call_id") or ""):
                raise HTTPException(status_code=400, detail="function_call_output requires call_id")
            _serialize_function_payload(item.get("output") or "", max_bytes=MAX_FUNCTION_OUTPUT_BYTES, field_name="function output")
        native.append(dict(item))
    return native


def _normalize_native_response_model(value: Any, requested_model: str) -> Any:
    if isinstance(value, dict):
        copied = dict(value)
        if copied.get("object") == "response" or "output" in copied or "output_text" in copied:
            copied["model"] = requested_model
        response = copied.get("response")
        if isinstance(response, dict):
            response_copy = dict(response)
            response_copy["model"] = requested_model
            copied["response"] = response_copy
        return copied
    return value


def _native_response_output_text(response: dict) -> str:
    text = response.get("output_text")
    if isinstance(text, str):
        return text
    out: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                out.append(str(part.get("text") or ""))
    return "".join(out)


def _native_response_sse_event(event: str, payload: dict, requested_model: str) -> str:
    normalized = _normalize_native_response_model(payload, requested_model)
    return _sse_event(event, normalized if isinstance(normalized, dict) else payload)

def _responses_function_call_to_chat_message(item: dict) -> dict:
    call_id = str(item.get("call_id") or item.get("id") or "")
    if not call_id:
        raise HTTPException(status_code=400, detail="function_call requires call_id")
    name = _validate_function_name(item.get("name"))
    arguments = _serialize_function_payload(item.get("arguments") or "", max_bytes=MAX_FUNCTION_ARGUMENTS_BYTES, field_name="function arguments")
    return {
        "role": "assistant",
        "content": item["content"] if "content" in item else None,
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def _responses_function_output_to_chat_message(item: dict) -> dict:
    call_id = str(item.get("call_id") or "")
    if not call_id:
        raise HTTPException(status_code=400, detail="function_call_output requires call_id")
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": _serialize_function_payload(item.get("output"), max_bytes=MAX_FUNCTION_OUTPUT_BYTES, field_name="function output"),
    }


def _chat_tool_calls_to_responses_items(tool_calls: Any, reverse_name_map: dict | None = None) -> list[dict]:
    if not isinstance(tool_calls, list):
        return []
    if len(tool_calls) > MAX_TOOL_CALLS_PER_RESPONSE:
        raise HTTPException(status_code=502, detail="Upstream returned too many tool calls")
    items: list[dict] = []
    for index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        raw_arguments = fn.get("arguments") or ""
        if len(str(raw_arguments).encode("utf-8")) > MAX_FUNCTION_ARGUMENTS_BYTES:
            raise HTTPException(status_code=502, detail="Upstream function arguments exceed configured byte limit")
        arguments = _serialize_function_payload(raw_arguments, max_bytes=MAX_FUNCTION_ARGUMENTS_BYTES, field_name="function arguments")
        call_id = str(call.get("id") or f"call_{uuid4().hex}")
        provider_name = str(fn.get("name") or "")
        restored = tool_compat.restore_tool_call_name(provider_name, reverse_name_map)
        item = {
            "id": "fc_" + uuid4().hex,
            "type": "function_call",
            "status": "completed",
            "call_id": call_id,
            "name": restored["name"],
            "arguments": arguments,
        }
        if restored.get("namespace"):
            item["namespace"] = restored["namespace"]
        items.append(item)
    return items


def _responses_payload_from_chat(model: str, upstream: dict, reverse_name_map: dict | None = None) -> dict:
    now = int(time.time())
    choice = ((upstream.get("choices") or [{}])[0] or {}) if isinstance(upstream, dict) else {}
    message = choice.get("message") or {}
    finish_reason = str(choice.get("finish_reason") or "stop").lower()
    text = "" if message.get("content") is None else str(message.get("content"))
    output: list[dict] = []
    if text:
        output.append({
            "id": "msg_" + uuid4().hex,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        })
    output.extend(_chat_tool_calls_to_responses_items(message.get("tool_calls"), reverse_name_map))
    status = "incomplete" if finish_reason == "length" else "completed"
    response = {
        "id": "resp_" + uuid4().hex,
        "object": "response",
        "created_at": now,
        "status": status,
        "model": model,
        "output": output,
        "output_text": text,
        "usage": _convert_openai_usage(upstream.get("usage") if isinstance(upstream, dict) else None),
    }
    if status == "incomplete":
        response["incomplete_details"] = {"reason": "max_output_tokens"}
    return response



def _responses_payload(model: str, text: str, usage: dict | None = None) -> dict:
    now = int(time.time())
    return {
        "id": "resp_" + uuid4().hex,
        "object": "response",
        "created_at": now,
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": "msg_" + uuid4().hex,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "output_text": text,
        "usage": usage or _usage_zero(),
    }


def _responses_stream_sequence(model: str, deltas: list[str], usage: dict | None = None, error: dict | None = None):
    response_id = "resp_" + uuid4().hex
    message_id = "msg_" + uuid4().hex
    created_at = int(time.time())
    sequence = 0
    text = ""

    response_started = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "in_progress",
        "model": model,
        "output": [],
    }
    yield _sse_event("response.created", {
        "type": "response.created",
        "sequence_number": sequence,
        "response": response_started,
    })
    sequence += 1
    yield _sse_event("response.in_progress", {
        "type": "response.in_progress",
        "sequence_number": sequence,
        "response": response_started,
    })
    sequence += 1

    item = {
        "id": message_id,
        "type": "message",
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }
    yield _sse_event("response.output_item.added", {
        "type": "response.output_item.added",
        "sequence_number": sequence,
        "output_index": 0,
        "item": item,
    })
    sequence += 1
    yield _sse_event("response.content_part.added", {
        "type": "response.content_part.added",
        "sequence_number": sequence,
        "item_id": message_id,
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": ""},
    })
    sequence += 1

    if error:
        yield _sse_event("error", {"type": "error", "sequence_number": sequence, "error": error})
        return

    for delta in deltas:
        if not delta:
            continue
        text += delta
        yield _sse_event("response.output_text.delta", {
            "type": "response.output_text.delta",
            "sequence_number": sequence,
            "item_id": message_id,
            "output_index": 0,
            "content_index": 0,
            "delta": delta,
        })
        sequence += 1

    yield _sse_event("response.output_text.done", {
        "type": "response.output_text.done",
        "sequence_number": sequence,
        "item_id": message_id,
        "output_index": 0,
        "content_index": 0,
        "text": text,
    })
    sequence += 1
    content_part = {"type": "output_text", "text": text}
    yield _sse_event("response.content_part.done", {
        "type": "response.content_part.done",
        "sequence_number": sequence,
        "item_id": message_id,
        "output_index": 0,
        "content_index": 0,
        "part": content_part,
    })
    sequence += 1
    completed_item = {
        "id": message_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [content_part],
    }
    yield _sse_event("response.output_item.done", {
        "type": "response.output_item.done",
        "sequence_number": sequence,
        "output_index": 0,
        "item": completed_item,
    })
    sequence += 1
    completed_response = _responses_payload(model, text, usage)
    completed_response["id"] = response_id
    completed_response["created_at"] = created_at
    completed_response["output"][0]["id"] = message_id
    yield _sse_event("response.completed", {
        "type": "response.completed",
        "sequence_number": sequence,
        "response": completed_response,
    })


async def _remote_chat_stream_parts(
    messages: list[dict],
    model: str,
    temperature: float | None,
    max_tokens_override: int | None = None,
    source_route: str = "responses",
    tools: list[dict] | None = None,
    tool_choice: Any | None = None,
    reasoning: Any | None = None,
):
    cfg = _model_cfg(model)
    base0 = (cfg.get("base_url") or cfg.get("api_base") or "").rstrip("/")
    if not base0:
        raise HTTPException(502, f"Remote config missing base_url/api_base for model '{model}'")
    physical_model, provider_model, explicit_provider_model = _remote_model_ids(model, cfg, base0)
    if not explicit_provider_model and provider_model != model:
        cfg.update(_remote_overrides(provider_model))
    cfg_phys = (_REMOTE_CONF.get("models") or {}).get(provider_model, {})
    if not explicit_provider_model and cfg_phys:
        cfg.update(cfg_phys)

    base = (cfg.get("base_url") or cfg.get("api_base") or "").rstrip("/")
    physical_model, provider_model, explicit_provider_model = _remote_model_ids(model, cfg, base)
    api_env = cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(api_env, "").strip()
    if not base or not api_key:
        raise HTTPException(502, f"Remote config/keys missing for model '{model}' (base={base}, env={api_env})")

    prov = _provider_name(base)
    is_modern = provider_model.startswith(("gpt-5", "o1", "o3", "o4"))
    temp = temperature if temperature is not None else cfg.get("temperature")
    top_p = cfg.get("top_p")
    max_tokens = max_tokens_override if max_tokens_override is not None else cfg.get("max_tokens")
    endpoint = f"{base}/chat/completions"
    payload = {"model": provider_model, "messages": messages, "stream": True}
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if reasoning is not None and cfg.get("supports_reasoning"):
        payload["reasoning"] = reasoning
    if max_tokens is not None:
        if prov == "openai" and is_modern:
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
    if not (prov == "openai" and is_modern):
        if temp is not None:
            payload["temperature"] = temp
        if top_p is not None:
            payload["top_p"] = top_p

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if "openrouter.ai" in base:
        headers.setdefault("HTTP-Referer", "https://openwebui.local")
        headers.setdefault("X-Title", "Adapter Bridge")

    timeout_s = cfg.get("timeout")
    timeout = httpx.Timeout(timeout_s) if timeout_s else TIMEOUT_REMOTE
    log.info(
        "[remote_chat_stream] route=%s requested_model=%s physical_model=%s provider_model=%s base_url=%s api_key_env=%s endpoint=%s stream=true tools=%s tool_choice=%s",
        source_route, model, physical_model, provider_model, base, api_env, endpoint, bool(tools), tool_choice,
    )
    log.debug("[remote_chat_stream] payload_summary=%s", _chat_payload_debug_summary(
        requested_model=model,
        provider_model=provider_model,
        messages=messages,
        payload=payload,
        endpoint=endpoint,
    ))

    _log_deepseek_tools_payload(
        route="remote_chat_stream",
        requested_model=model,
        provider_model=provider_model,
        endpoint=endpoint,
        payload=payload,
    )
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", endpoint, json=payload, headers=headers) as resp:
            if resp.status_code >= 400:
                body = ""
                try:
                    body = (await resp.aread()).decode("utf-8", errors="replace")
                except Exception:
                    pass
                raise HTTPException(
                    status_code=resp.status_code,
                    detail={"message": f"Upstream error {resp.status_code}", "endpoint": endpoint, "body": body[:500]},
                )
            async for line in resp.aiter_lines():
                line = (line or "").strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                except Exception:
                    continue
                usage = chunk.get("usage")
                if usage:
                    yield {"usage": _convert_openai_usage(usage)}
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice0 = choices[0] or {}
                finish_reason = choice0.get("finish_reason")
                if finish_reason:
                    yield {"finish_reason": str(finish_reason)}
                delta_obj = choice0.get("delta") or {}
                delta = delta_obj.get("content")
                if delta:
                    yield {"delta": str(delta)}
                for tool_delta in delta_obj.get("tool_calls") or []:
                    if not isinstance(tool_delta, dict):
                        continue
                    fn = tool_delta.get("function") or {}
                    yield {
                        "tool_call_delta": {
                            "index": int(tool_delta.get("index") or 0),
                            "id": tool_delta.get("id"),
                            "type": tool_delta.get("type") or "function",
                            "name": fn.get("name"),
                            "arguments": fn.get("arguments"),
                        }
                    }


async def _responses_stream_generator(
    *,
    requested_model: str,
    messages: list[dict],
    metadata: dict | None = None,
    max_output_tokens: int | None = None,
    tools: list[dict] | None = None,
    tool_choice: Any | None = None,
    reasoning: Any | None = None,
    reverse_name_map: dict | None = None,
):
    canonical = _resolve_model_id(requested_model)
    response_id = "resp_" + uuid4().hex
    message_id = "msg_" + uuid4().hex
    created_at = int(time.time())
    sequence = 0
    output_text = ""
    usage: dict | None = None
    finish_reason = "stop"
    started = False
    text_started = False
    output_items: list[dict] = []
    tool_states: dict[int, dict] = {}

    def next_event(event: str, payload: dict):
        nonlocal sequence
        payload.setdefault("sequence_number", sequence)
        sequence += 1
        return _sse_event(event, payload)

    def completed_response(status: str = "completed") -> dict:
        response = {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": status,
            "model": requested_model,
            "output": output_items,
            "output_text": output_text,
            "usage": usage or _usage_zero(),
        }
        if status == "incomplete":
            response["incomplete_details"] = {"reason": "max_output_tokens"}
        return response

    async def ensure_text_item():
        nonlocal text_started
        if text_started:
            return
        text_started = True
        output_index = len(output_items)
        yield next_event("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {
                "id": message_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        })
        yield next_event("response.content_part.added", {
            "type": "response.content_part.added",
            "item_id": message_id,
            "output_index": output_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": ""},
        })

    async def emit_text_delta(delta: str):
        nonlocal output_text
        async for event in ensure_text_item():
            yield event
        output_index = next((i for i, item in enumerate(output_items) if item.get("id") == message_id), None)
        if output_index is None:
            output_index = len(output_items)
        output_text += delta
        yield next_event("response.output_text.delta", {
            "type": "response.output_text.delta",
            "item_id": message_id,
            "output_index": output_index,
            "content_index": 0,
            "delta": delta,
        })

    async def emit_tool_delta(tool_delta: dict):
        index = int(tool_delta.get("index") or 0)
        state = tool_states.get(index)
        if state is None:
            state = {
                "item_id": "fc_" + uuid4().hex,
                "call_id": str(tool_delta.get("id") or f"call_{uuid4().hex}"),
                "name": "",
                "arguments": "",
                "output_index": len(output_items) + len(tool_states),
                "added": False,
            }
            tool_states[index] = state
        if tool_delta.get("id"):
            state["call_id"] = str(tool_delta.get("id"))
        if tool_delta.get("name"):
            state["name"] += str(tool_delta.get("name"))
        if not state["added"]:
            state["added"] = True
            yield next_event("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": state["output_index"],
                "item": {
                    "id": state["item_id"],
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": state["call_id"],
                    "name": state["name"],
                    "arguments": "",
                },
            })
        arg_delta = tool_delta.get("arguments")
        if arg_delta:
            arg_delta = str(arg_delta)
            if len((state["arguments"] + arg_delta).encode("utf-8")) > MAX_FUNCTION_ARGUMENTS_BYTES:
                raise HTTPException(status_code=502, detail="Upstream function arguments exceed configured byte limit")
            state["arguments"] += arg_delta
            yield next_event("response.function_call_arguments.delta", {
                "type": "response.function_call_arguments.delta",
                "item_id": state["item_id"],
                "output_index": state["output_index"],
                "delta": arg_delta,
            })

    try:
        started = True
        response_started = {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "in_progress",
            "model": requested_model,
            "output": [],
        }
        yield next_event("response.created", {"type": "response.created", "response": response_started})
        yield next_event("response.in_progress", {"type": "response.in_progress", "response": response_started})

        if _is_local_model(requested_model):
            log.info("[responses_stream] requested_model=%s backend=local stream_fallback=single_delta", requested_model)
            text = await _route_text_completion(
                requested_model=requested_model,
                messages=messages,
                metadata=metadata,
                max_output_tokens=max_output_tokens,
                source_route="responses",
            )
            async for event in emit_text_delta(str(text)):
                yield event
        elif not _is_known_remote_model(requested_model, canonical):
            raise HTTPException(status_code=400, detail=f"Unknown model '{requested_model}'")
        else:
            cfg = _model_cfg(canonical)
            if tools and cfg.get("supports_stream") is False:
                raise HTTPException(status_code=400, detail=f"Model '{requested_model}' does not support streamed tool calls")
            if cfg.get("supports_stream") is False:
                log.info("[responses_stream] requested_model=%s stream_fallback=single_delta", requested_model)
                text = await _route_text_completion(
                    requested_model=requested_model,
                    messages=messages,
                    metadata=metadata,
                    max_output_tokens=max_output_tokens,
                    source_route="responses",
                )
                async for event in emit_text_delta(str(text)):
                    yield event
            else:
                async for part in _remote_chat_stream_parts(
                    messages=messages,
                    model=canonical,
                    temperature=None,
                    max_tokens_override=max_output_tokens,
                    source_route="responses",
                    tools=tools,
                    tool_choice=tool_choice,
                    reasoning=reasoning,
                ):
                    if "usage" in part:
                        usage = part["usage"]
                    if "finish_reason" in part:
                        finish_reason = str(part["finish_reason"]).lower()
                    delta = part.get("delta")
                    if delta:
                        async for event in emit_text_delta(delta):
                            yield event
                    tool_delta = part.get("tool_call_delta")
                    if tool_delta:
                        async for event in emit_tool_delta(tool_delta):
                            yield event

        if usage is None:
            log.debug("[responses_stream] usage_unavailable=true requested_model=%s", requested_model)
            usage = _usage_zero()

        if text_started:
            output_index = len(output_items)
            yield next_event("response.output_text.done", {
                "type": "response.output_text.done",
                "item_id": message_id,
                "output_index": output_index,
                "content_index": 0,
                "text": output_text,
            })
            content_part = {"type": "output_text", "text": output_text}
            yield next_event("response.content_part.done", {
                "type": "response.content_part.done",
                "item_id": message_id,
                "output_index": output_index,
                "content_index": 0,
                "part": content_part,
            })
            completed_item = {
                "id": message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [content_part],
            }
            output_items.append(completed_item)
            yield next_event("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": completed_item,
            })

        for index in sorted(tool_states):
            state = tool_states[index]
            yield next_event("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "item_id": state["item_id"],
                "output_index": state["output_index"],
                "arguments": state["arguments"],
            })
            restored = tool_compat.restore_tool_call_name(state["name"], reverse_name_map)
            item = {
                "id": state["item_id"],
                "type": "function_call",
                "status": "completed",
                "call_id": state["call_id"],
                "name": restored["name"],
                "arguments": state["arguments"],
            }
            if restored.get("namespace"):
                item["namespace"] = restored["namespace"]
            output_items.append(item)
            yield next_event("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": state["output_index"],
                "item": item,
            })

        status = "incomplete" if finish_reason == "length" else "completed"
        yield next_event("response.completed", {
            "type": "response.completed",
            "response": completed_response(status),
        })
    except asyncio.CancelledError:
        log.debug("[responses_stream] client disconnected requested_model=%s", requested_model)
        raise
    except HTTPException as exc:
        if started:
            yield _sse_event("error", {"type": "error", "error": {"status_code": exc.status_code, "detail": exc.detail}})
        else:
            raise
    except Exception as exc:
        log.warning("[responses_stream] error requested_model=%s err=%s", requested_model, exc)
        yield _sse_event("error", {"type": "error", "error": {"message": str(exc)}})

def _is_known_remote_model(model: str, canonical: str) -> bool:
    models_conf = _REMOTE_CONF.get("models") or {}
    if model in models_conf or canonical in models_conf:
        return True
    if model in REMOTE_MODELS_SET or canonical in REMOTE_MODELS_SET:
        return True
    cfg = MODEL_REGISTRY.get(canonical)
    return bool(cfg and cfg.get("backend") == "openai")


async def _route_text_completion(
    *,
    requested_model: str,
    messages: list[dict],
    temperature: float | None = None,
    metadata: dict | None = None,
    response_format: dict | None = None,
    max_output_tokens: int | None = None,
    source_route: str = "chat/completions",
) -> str:
    canonical = _resolve_model_id(requested_model)
    meta = metadata or {}

    if _is_local_model(requested_model):
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        user_prompt = messages[-1].get("content") or ""
        log.info("[route_text_completion] route=%s requested_model=%s backend=local", source_route, requested_model)
        return str(await _local_chat(
            user_prompt,
            route_hint=requested_model,
            temperature=temperature,
            meta=meta,
            messages=messages,
            use_memory=False,
        ))

    cfg = MODEL_REGISTRY.get(canonical)
    if cfg and cfg.get("backend") == "gpt4all":
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        user_prompt = messages[-1].get("content") or ""
        log.info("[route_text_completion] route=%s requested_model=%s canonical=%s backend=local-registry", source_route, requested_model, canonical)
        return str(await _local_chat(
            user_prompt,
            route_hint=canonical,
            temperature=temperature,
            meta=meta,
            messages=messages,
            use_memory=False,
        ))

    if not _is_known_remote_model(requested_model, canonical):
        raise HTTPException(status_code=400, detail=f"Unknown model '{requested_model}'")

    log.info("[route_text_completion] route=%s requested_model=%s canonical=%s backend=remote", source_route, requested_model, canonical)
    return await _remote_chat(
        messages=messages,
        model=canonical,
        temperature=temperature,
        response_format=response_format,
        max_tokens_override=max_output_tokens,
        source_route=source_route,
    )


@app.get("/ping")
async def ping():
    return {"status": "ok"}


@app.post("/v1/responses")
async def responses_create(
    req: ResponsesReq,
    authorization: Annotated[str | None, Header()] = None,
):
    _check_adapter_auth(authorization)

    if req.previous_response_id:
        raise HTTPException(
            status_code=400,
            detail="previous_response_id is not supported in stateless mode; resend the complete input history",
        )

    canonical = _resolve_model_id(req.model)
    cfg = _model_cfg(canonical)
    native_responses = _native_responses_enabled(cfg)
    if native_responses:
        if (req.tools or _input_contains_tool_items(req.input)) and not _tool_capability(cfg, "supports_tools"):
            raise HTTPException(status_code=400, detail=f"Model '{req.model}' does not declare supports_tools=true")
        if req.stream:
            return StreamingResponse(
                _remote_responses_native_stream(
                    requested_model=req.model,
                    canonical_model=canonical,
                    input_value=req.input,
                    instructions=req.instructions,
                    tools=req.tools,
                    tool_choice=req.tool_choice,
                    reasoning=getattr(req, "reasoning", None),
                    max_output_tokens=req.max_output_tokens,
                    parallel_tool_calls=getattr(req, "parallel_tool_calls", None),
                    metadata=req.metadata,
                ),
                media_type="text/event-stream",
                headers=_responses_stream_headers(),
            )
        raw_native = await _remote_responses_native(
            requested_model=req.model,
            canonical_model=canonical,
            input_value=req.input,
            instructions=req.instructions,
            tools=req.tools,
            tool_choice=req.tool_choice,
            reasoning=getattr(req, "reasoning", None),
            max_output_tokens=req.max_output_tokens,
            parallel_tool_calls=getattr(req, "parallel_tool_calls", None),
            metadata=req.metadata,
        )
        return raw_native

    final_tool_result_turn = _responses_final_tool_result_turn(req.input, req.tool_choice)
    chat_tool_result = _responses_tools_to_chat_result(req.tools, cfg, req.model, req.tool_choice)
    chat_tools = chat_tool_result.tools or None
    reverse_name_map = chat_tool_result.reverse_name_map
    chat_tool_choice = _responses_tool_choice_to_chat(req.tool_choice, cfg)
    if final_tool_result_turn:
        if chat_tools:
            log.debug("[responses_tools] omitting tools on final tool-result turn requested_model=%s", req.model)
        chat_tools = None
        chat_tool_choice = None
    effective_max_output_tokens = _responses_effective_max_tokens(req.max_output_tokens, final_tool_result_turn)
    if (req.tools or _input_contains_tool_items(req.input)) and not _tool_capability(cfg, "supports_tools"):
        raise HTTPException(status_code=400, detail=f"Model '{req.model}' does not declare supports_tools=true")

    messages = _responses_input_to_messages(req.input, req.instructions, cfg=cfg, requested_model=req.model)

    if req.stream:
        return StreamingResponse(
            _responses_stream_generator(
                requested_model=req.model,
                messages=messages,
                metadata=req.metadata,
                max_output_tokens=effective_max_output_tokens,
                tools=chat_tools,
                tool_choice=chat_tool_choice,
                reasoning=getattr(req, "reasoning", None),
                reverse_name_map=reverse_name_map,
            ),
            media_type="text/event-stream",
            headers=_responses_stream_headers(),
        )

    if _is_local_model(req.model):
        if req.tools or _input_contains_tool_items(req.input):
            raise HTTPException(status_code=400, detail=f"Model '{req.model}' does not declare supports_tools=true")
        text = await _route_text_completion(
            requested_model=req.model,
            messages=messages,
            metadata=req.metadata,
            max_output_tokens=effective_max_output_tokens,
            source_route="responses",
        )
        return _responses_payload(req.model, str(text))

    if not _is_known_remote_model(req.model, canonical):
        raise HTTPException(status_code=400, detail=f"Unknown model '{req.model}'")

    raw = await _remote_chat(
        messages=messages,
        model=canonical,
        temperature=None,
        max_tokens_override=effective_max_output_tokens,
        source_route="responses",
        tools=chat_tools,
        tool_choice=chat_tool_choice,
        reasoning=getattr(req, "reasoning", None),
        return_raw_response=True,
    )
    return _responses_payload_from_chat(req.model, raw, reverse_name_map)


# -----------------------------------------------------------------------------
# REST: Chat
# -----------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatReq,
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_app_id: Annotated[str | None, Header()] = None,
    x_conversation_id: Annotated[str | None, Header()] = None,
):


    _check_adapter_auth(authorization)

    model      = req.model
    canonical  = _resolve_model_id(model)
    messages   = [m.model_dump() for m in req.messages]
    temperature = req.temperature
    meta       = req.metadata or {}
    requested_response_format = req.response_format
    log.info("[chat] meta_keys=%s", sorted(list((meta or {}).keys())))
    for k in ("conversation_id", "chat_id", "thread_id", "id"):
        if k in (meta or {}):
            v = meta.get(k)
            log.info("[chat] meta[%s]=%r", k, v if isinstance(v, (str, int)) else type(v).__name__)

    # en haut du fichier : import hashlib  (si vous gardez sha1)

    # ---- IDs (priorité: metadata -> headers -> fallback) ----

    # ---- IDs (priorité: headers OpenWebUI -> metadata -> headers génériques -> fallback stable) ----

    # 0) lire d'abord les headers "forwardés" par OpenWebUI (noms possibles selon versions)
    ow_chat_id = (
        request.headers.get("x-openwebui-chat-id")
        or request.headers.get("x-chat-id")
        or request.headers.get("x-conversation-id")   # si OpenWebUI/Proxy l’envoie ainsi
        or ""
    ).strip()

    # 1) app_id : metadata -> header explicite -> défaut
    app_id = (meta.get("app_id") or meta.get("x_app_id") or x_app_id or "").strip()
    if not app_id:
        app_id = "openwebui"

    # 2) conv_id : priorité au chat_id venant d’OpenWebUI, sinon metadata, sinon header explicite
    conv_id = (
        ow_chat_id
        or (meta.get("conversation_id") or meta.get("thread_id") or meta.get("chat_id") or "").strip()
        or (x_conversation_id or "").strip()
    )

    fallback_conv = None  # pour logs

    # 3) fallback stable si toujours vide
    if not conv_id:
        h = {
            "x-app-id": (x_app_id or ""),
            "x-conversation-id": (x_conversation_id or ""),
            "user-agent": request.headers.get("user-agent", ""),
        }
        b = {
            "app_id": (meta.get("app_id") or meta.get("x_app_id") or x_app_id or ""),
            "conversation_id": (meta.get("conversation_id") or meta.get("thread_id") or meta.get("chat_id") or ""),
            "memory_id": (meta.get("memory_id") or ""),
        }

        _, fallback_conv = _pick_ids(h, b)  # fallback_conv sera typiquement "ua_xxx" si rien d'autre
        fallback_conv = fallback_conv or "ua-unknown"

        # Option A (simple) : utiliser directement fallback_conv comme conv_id
        # conv_id = fallback_conv

        # Option B (homogène) : transformer en "chat_<hash>" stable
        conv_id = "chat_" + sha256(f"{app_id}:{fallback_conv}".encode("utf-8")).hexdigest()[:16]

    # 4) logs
    log.info(
        "[chat] app_id=%s conv_id=%s ow_chat_id=%s fallback=%s",
        app_id, conv_id, (ow_chat_id or None), (fallback_conv or None)
    )



    # (option) activer mémoire si demandé par client
    use_memory = bool(meta.get("use_memory") or meta.get("memory") or meta.get("memory_id"))

    # ----------------------------------------------------------------
    # CAS 1) Modèles "spéciaux" déclarés dans MODEL_REGISTRY (annoter_segments_*, chat_general, etc.)
    # ----------------------------------------------------------------

    cfg = MODEL_REGISTRY.get(canonical)
    if cfg:
        backend   = cfg.get("backend")
        json_mode = bool(cfg.get("json_mode", False))

        # JSON-mode → contrainte sur le dernier message user
        if json_mode:
            messages = apply_json_constraint_to_messages(messages)

        # ----------------------------------------------------------------
        # ----- CAS LOCAL (gpt4all / Flask) -----
        # ----------------------------------------------------------------

        if backend == "gpt4all":
            try:
                user_prompt = messages[-1]["content"]
                text = await _local_chat(
                    user_prompt,
                    route_hint=canonical,
                    temperature=temperature,
                    meta=meta,
                )

                if json_mode:
                    if isinstance(text, dict):
                        content = json.dumps(text, ensure_ascii=False)
                    else:
                        content = str(text)
                else:
                    content = str(text)

            except Exception as e:
                raise HTTPException(502, f"Local LLM error: {e}")

            return ChatResp(
                model=model,
                choices=[ChatRespChoice(
                    index=0,
                    message={"role": "assistant", "content": content}
                )]
            )
        # ----------------------------------------------------------------
        # ----- CAS REMOTE (OpenAI / OpenRouter) -----
        # ----------------------------------------------------------------

        elif backend == "openai":


            # ─────────────────────────────────────────────
            # 1er appel remote (en général gpt-5-mini)
            # ─────────────────────────────────────────────

            raw = None
            parsed = None
            last_exc = None
            used_model = canonical
            normalized_report = None
            normalized_debrief = None
            normalized_pass2e = None
            normalized_pass3e = None
            for m in _fallback_chain_for(canonical):

                response_format = None
                if _canonical_uses_structured_outputs(canonical):
                    response_format = None

                try:
                    raw = await _remote_chat_with_retry(
                        messages,
                        m,
                        temperature,
                        retries=2,
                        response_format=response_format,
                    )
                    if _is_effectively_empty_raw(raw):
                        log.warning("[adapter][json-model] empty raw from model=%s -> next fallback", m)
                        raw = None
                        continue

                except Exception as e:
                    last_exc = e
                    log.warning(f"[annoter_segments] remote call failed model={m}: {type(e).__name__}: {e}")
                    continue




                parsed = None
                try:
                    parsed = extract_json_from_llm(raw)
                except Exception:
                    parsed = None
                if parsed is None:
                    parsed = _repair_truncated_json(raw)

                if _is_effectively_empty_payload(parsed):
                    log.warning("[adapter][json-model] empty parsed payload model=%s -> next fallback", m)
                    parsed = None
                    continue


                if _is_debrief_model(canonical):
                    if not isinstance(parsed, dict):
                        log.warning("[adapter][debrief-json] parse OK but not a dict model=%s -> next fallback", m)
                        parsed = None
                        continue

                    try:
                        normalized_debrief = normalize_debrief_annotation(parsed)
                    except Exception as e:
                        log.warning("[adapter][debrief-json] normalize failed model=%s err=%r -> next fallback", m, e)
                        normalized_debrief = None
                        parsed = None
                        continue

                    if _is_effectively_empty_debrief(normalized_debrief):
                        log.warning("[adapter][debrief-json] normalized but empty model=%s -> next fallback", m)
                        normalized_debrief = None
                        parsed = None
                        continue

                    used_model = m
                    break
                if _is_pass2e_model(canonical):
                    if not isinstance(parsed, dict):
                        log.warning("[adapter][pass2e-json] parse OK but not a dict model=%s -> next fallback", m)
                        parsed = None
                        continue

                    try:
                        normalized_pass2e = normalize_pass2e_compact(parsed)
                    except Exception as e:
                        log.warning("[adapter][pass2e-json] normalize failed model=%s err=%r -> next fallback", m, e)
                        normalized_pass2e = None
                        parsed = None
                        continue

                    if _is_effectively_empty_pass2e(normalized_pass2e):
                        log.warning("[adapter][pass2e-json] normalized but empty model=%s -> next fallback", m)
                        normalized_pass2e = None
                        parsed = None
                        continue

                    used_model = m
                    break
                if _is_pass3e_model(canonical):
                    if not isinstance(parsed, dict):
                        log.warning("[adapter][pass3e-json] parse OK but not a dict model=%s -> next fallback", m)
                        parsed = None
                        continue

                    try:
                        normalized_pass3e = normalize_pass3e_synthesis(parsed)
                    except Exception as e:
                        log.warning("[adapter][pass3e-json] normalize failed model=%s err=%r -> next fallback", m, e)
                        normalized_pass3e = None
                        parsed = None
                        continue

                    if _is_effectively_empty_pass3e(normalized_pass3e):
                        log.warning("[adapter][pass3e-json] normalized but empty model=%s -> next fallback", m)
                        normalized_pass3e = None
                        parsed = None
                        continue

                    used_model = m
                    break
                if canonical in ("report_remote", "report_remote_alt", "report_remote_alt2"):
                    # 1) on n'exige pas le schéma complet, juste "report-like" ou même dict
                    if not isinstance(parsed, dict):
                        log.warning("[adapter][json-model] report parse OK but not a dict model=%s -> next fallback", m)
                        parsed = None
                        continue

                    # 2) normalisation tolérante
                    try:
                        normalized_report = normalize_report_annotation(parsed)
                    except Exception as e:
                        log.warning("[adapter][json-model] report normalize failed model=%s err=%r -> next fallback", m, e)
                        normalized_report = None
                        parsed = None
                        continue

                    # 3) si c'est vide, on tente le modèle suivant (au lieu de tomber en fallback JSON vide)
                    if _is_effectively_empty_report(normalized_report):
                        log.warning("[adapter][json-model] report normalized but empty model=%s -> next fallback", m)
                        normalized_report = None
                        parsed = None
                        continue

                    used_model = m
                    break

                # sinon (autres modèles)
                if parsed is not None:
                    used_model = m
                    break

                log.warning("[adapter][json-model] parse/repair failed model=%s -> next fallback", m)

            if raw is None:
                if last_exc is not None:
                    raise last_exc
                raise HTTPException(502, "Remote call failed: no response captured")

            if _is_debrief_model(canonical):
                if normalized_debrief is None:
                    snippet = (raw or "")[:4000]
                    log.error("[adapter][debrief-json] raw snippet=%r", snippet)
                    fb = _fallback_json_for_model(canonical)
                    log.error("[adapter][debrief-json] impossible a normaliser pour canonical=%s (fallback JSON debrief)", canonical)
                    content = json.dumps(fb, ensure_ascii=False)
                else:
                    content = json.dumps(normalized_debrief, ensure_ascii=False)

                return ChatResp(
                    model=model,
                    choices=[ChatRespChoice(index=0, message={"role": "assistant", "content": content})]
                )
            if _is_pass2e_model(canonical):
                if normalized_pass2e is None:
                    snippet = (raw or "")[:4000]
                    log.error("[adapter][pass2e-json] raw snippet=%r", snippet)
                    fb = _fallback_json_for_model(canonical)
                    log.error("[adapter][pass2e-json] impossible a normaliser pour canonical=%s (fallback JSON pass2e)", canonical)
                    content = json.dumps(fb, ensure_ascii=False)
                else:
                    content = json.dumps(normalized_pass2e, ensure_ascii=False)

                return ChatResp(
                    model=model,
                    choices=[ChatRespChoice(index=0, message={"role": "assistant", "content": content})]
                )
            if _is_pass3e_model(canonical):
                if normalized_pass3e is None:
                    snippet = (raw or "")[:4000]
                    log.error("[adapter][pass3e-json] raw snippet=%r", snippet)
                    fb = _fallback_json_for_model(canonical)
                    log.error("[adapter][pass3e-json] impossible a normaliser pour canonical=%s (fallback JSON pass3e)", canonical)
                    content = json.dumps(fb, ensure_ascii=False)
                else:
                    content = json.dumps(normalized_pass3e, ensure_ascii=False)

                return ChatResp(
                    model=model,
                    choices=[ChatRespChoice(index=0, message={"role": "assistant", "content": content})]
                )
            #----------------------------------------------------------------
            # si report : on exige normalized_report (pas juste parsed)
            if canonical in ("report_remote", "report_remote_alt", "report_remote_alt2"):
                if normalized_report is None:
                    snippet = (raw or "")[:4000]
                    log.error("[adapter][json-model] raw snippet=%r", snippet)
                    fb = _fallback_json_for_model(canonical)
                    log.error("[adapter][json-model] report impossible à normaliser pour canonical=%s (fallback JSON vide)", canonical)
                    content = json.dumps(fb, ensure_ascii=False)
                else:
                    content = json.dumps(normalized_report, ensure_ascii=False)

                return ChatResp(
                    model=model,
                    choices=[ChatRespChoice(index=0, message={"role": "assistant", "content": content})]
                )

            # non-report : on renvoie un JSON valide quoi qu’il arrive
            if parsed is None:
                snippet = (raw or "")[:4000]
                log.error("[adapter][json-model] raw snippet=%r", snippet)
                fb = _fallback_json_for_model(canonical)
                log.error("[adapter][json-model] JSON invalide/tronqué pour canonical=%s (fallback JSON vide)", canonical)
                parsed = fb

            # normalisation Pass 1 (segments) si applicable, même partiel
            if _is_segment_like(parsed):
                parsed = normalize_segment_annotation(parsed)

            content = json.dumps(parsed, ensure_ascii=False)
            return ChatResp(
                model=model,
                choices=[ChatRespChoice(index=0, message={"role": "assistant", "content": content})]
            )

        else:
            raise HTTPException(500, f"Backend inconnu : {backend}")


    # ----------------------------------------------------------------
    # 2) CAS GÉNÉRIQUE : modèle non présent dans MODEL_REGISTRY
    # ----------------------------------------------------------------
    # use_remote = bool(meta.get("use_remote"))
    #
    # if use_remote:
    #    raw = await _remote_chat(messages=messages, model=canonical, temperature=temperature)
    #    content = raw
    # else:
    #     user_prompt = messages[-1]["content"]
    #     text = await _local_chat(user_prompt, route_hint=model, temperature=temperature, meta=meta)
    #    content = str(text)

    # ----------------------------------------------------------------
    #  CAS utilisé par OpenWebUI, PAPERLESS, APPLOWY, etc.
    #  pour modele remotes et locaux (gpt-5-mini, local-llama3, etc.) -----
    # ----------------------------------------------------------------

    # On décide "local vs remote" SUR LE NOM DEMANDÉ par le client (model),
    # pas sur le nom canonique (canonical).

    if _is_local_model(model):
        try:
            meta = meta or {}
            user_prompt = messages[-1]["content"]

            use_memory = bool(conv_id)
            real_model = MODEL_ALIAS.get(model, model)

            log.info(
                "[adapter] route_hint=%r real_model=%r conv_id=%r use_memory=%r",
                model, real_model, conv_id, use_memory
            )

            meta.setdefault("memory_turns", 6)

            text = await _local_chat(
                user_prompt,
                route_hint=model,
                temperature=temperature,
                meta=meta,
                app_id=app_id,
                conversation_id=conv_id,
                messages=messages,
                use_memory=use_memory,
            )
            content = str(text)

        except Exception as e:
            raise HTTPException(502, f"Local LLM error: {e}")

    else:
        # Remote standard : on route avec le nom canonique (gpt-5-mini, etc.)
        try:
            raw = await _remote_chat(
                messages=messages,
                model=canonical,
                temperature=temperature,
                response_format=requested_response_format,
            )
            content = raw
        except Exception as e:
            raise HTTPException(502, f"Remote LLM error: {e}")

    return ChatResp(
            model=model,
            choices=[ChatRespChoice(index=0, message={"role": "assistant", "content": content})],
        )


# -----------------------------------------------------------------------------
# Documents Pipeline “docs → OCR → LLM” (par défaut RAG vectoriel)
# -----------------------------------------------------------------------------

@app.post("/v1/documents/analyze")
async def documents_analyze(
    authorization: t.Annotated[str | None, Header()] = None,
    files: List[UploadFile] = File(default=[]),
    model: str = Form(default="annoter_rag_vecteur"),
    analysis_prompt: Optional[str] = Form(default=None),
    return_html: bool = Form(default=False),
    use_vector_rag: bool = Form(default=True),
    collection: Optional[str] = Form(default=None),
    vec_backend: Optional[str] = Form(default=None),
    metadata_form: Optional[str] = Form(default=None),
    json_body: AnalyzeReq = Body(default=None),
):
    _check_adapter_auth(authorization)

    # Unifier les params (multipart OU JSON)
    if json_body:
        model = json_body.model or model
        analysis_prompt = json_body.analysis_prompt or analysis_prompt
        return_html = json_body.return_html or return_html
        use_vector_rag = json_body.use_vector_rag or use_vector_rag
        collection = json_body.collection or collection
        vec_backend = json_body.vec_backend or vec_backend
        if json_body.metadata:
            meta = dict(json_body.metadata)
        else:
            meta = {}
        file_urls = json_body.file_urls or []
    else:
        meta = {}
        file_urls = []

        if metadata_form:
            try:
                meta.update(json.loads(metadata_form))
            except Exception:
                pass

    if not await _ensure_local_ready():
        raise HTTPException(502, "Local service unreachable after WOL")

    # 1) Récupérer/ocriser chaque fichier
    parts: list[tuple[str,str]] = []  # (label, text)
    async def _ocr_bytes(name: str, content: bytes) -> str:
        mt, _ = mimetypes.guess_type(name)
        mt = mt or ""
        headers = _llm_headers()
        async with httpx.AsyncClient(timeout=TIMEOUT_LOCAL) as c:
            if mt.startswith("image/"):
                # ton serveur traite l'image via /ocr
                r = await c.post(f"{LOCAL_BASE}/ocr", files={"file": (name, content, mt)}, headers=headers)
                r.raise_for_status()
                return r.json().get("text","")
            if mt == "application/pdf":
                # PDF : même endpoint /ocr, il détecte
                r = await c.post(f"{LOCAL_BASE}/ocr", files={"file": (name, content, mt)}, headers=headers)
                r.raise_for_status()
                return r.json().get("text","")
        # sinon, texte brut
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    # a) Fichiers uploadés
    for f in files or []:
        blob = await f.read()
        text = await _ocr_bytes(f.filename, blob)
        if text.strip():
            parts.append((f.filename, text))

    # b) Fichiers par URL (si fournis)
    for u in file_urls:
        if not _is_allowed_url(u):
            raise HTTPException(403, f"URL non autorisée : {u}")
        async with httpx.AsyncClient(timeout=TIMEOUT_LOCAL) as c:
            r = await c.get(u)
            r.raise_for_status()
            content = r.content
        name = u.split("?")[0].split("/")[-1] or f"file-{uuid.uuid4().hex}"
        text = await _ocr_bytes(name, content)
        if text.strip():
            parts.append((name, text))

    if not parts:
        raise HTTPException(400, "Aucun texte exploitable après OCR")

    # 2) Assembler un contexte propre
    def _sanitize(s: str) -> str:
        return s.replace("\x00", " ").strip()
    context_chunks = []
    for label, txt in parts:
        context_chunks.append(f"=== FICHIER: {label} ===\n{_sanitize(txt)}\n")
    full_context = "\n\n".join(context_chunks)

    # 3) Construire le prompt utilisateur
    user_prompt = analysis_prompt or "Analyse et synthèse des documents fournis."
    final_prompt = f"{user_prompt}\n\nCONTEXTE:\n{full_context}"

    # 4) Choisir la route LLM + métadonnées  (par défaut: RAG vectoriel)
    route_hint = model
    meta_llm = meta.copy()
    if return_html:
        meta_llm["return_html"] = True
    # Option RAG vectoriel (collection existante ou temporaire)
    if use_vector_rag:
        route_hint = "annoter_rag_vecteur"
        if collection:
            meta_llm.update({"collection": collection})
        if vec_backend:
            meta_llm.update({"vec_backend": vec_backend})

    # 5) Appeler le LLM local
    out = await _local_chat(final_prompt, route_hint=route_hint, temperature=None, meta=meta_llm)
    return {"object":"doc.analyze.result", "model":route_hint, "content": out}

# -----------------------------------------------------------------------------
# REST: Embeddings
# -----------------------------------------------------------------------------

# ---- remplace entièrement _embed par ceci ----
async def _embed(texts: list[str]) -> list[list[float]]:
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        raise HTTPException(400, "No texts provided for embeddings")

    model_local_primary  = EMBEDDINGS_MODEL                     # ex: "Nomic_Embed"
    model_local_fallback = os.getenv("FALLBACK_LOCAL_EMBEDDINGS_MODEL", "").strip()  # ex: "E5_multilingual_large"
    model_remote         = os.getenv("FALLBACK_EMBEDDINGS_MODEL", "").strip()        # ex: "text-embedding-3-small"

    # 1) LOCAL primaire (toujours d'abord)
    try:
        headers = {"x-api-key": LOCAL_API_KEY} if LOCAL_API_KEY else {}
        payload = {"texts": texts, "model": model_local_primary}
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{LOCAL_BASE}/embeddings", json=payload, headers=headers, timeout=TIMEOUT_LOCAL)
            r.raise_for_status()
            j = r.json()
            return j.get("embeddings") or [d["embedding"] for d in j.get("data", [])]
    except Exception as e_primary:
        pass  # on tente le fallback local

    # 2) LOCAL fallback (si défini)
    if model_local_fallback:
        try:
            headers = {"x-api-key": LOCAL_API_KEY} if LOCAL_API_KEY else {}
            payload = {"texts": texts, "model": model_local_fallback}
            async with httpx.AsyncClient() as client:
                r = await client.post(f"{LOCAL_BASE}/embeddings", json=payload, headers=headers, timeout=TIMEOUT_LOCAL)
                r.raise_for_status()
                j = r.json()
                return j.get("embeddings") or [d["embedding"] for d in j.get("data", [])]
        except Exception as e_local_fb:
            pass  # on envisage le remote seulement si autorisé

    # 3) REMOTE (en dernier, uniquement si explicitement autorisé)
    if ALLOW_REMOTE_EMBEDDINGS and REMOTE_BASE and REMOTE_API_KEY and (model_remote or EMBEDDINGS_MODEL):
        try:
            headers = {"Authorization": f"Bearer {REMOTE_API_KEY}"}
            payload = {"model": (model_remote or EMBEDDINGS_MODEL), "input": texts}
            async with httpx.AsyncClient() as client:
                r = await client.post(f"{REMOTE_BASE}/embeddings", json=payload, headers=headers, timeout=TIMEOUT_REMOTE)
                r.raise_for_status()
                j = r.json()
                return [d["embedding"] for d in j["data"]]
        except Exception as e_remote:
            pass

    # si on arrive ici, tout a échoué
    raise HTTPException(502, "Embeddings indisponibles (local primaire, local fallback, puis remote si autorisé).")

ENABLE_SEARCH = os.getenv("ENABLE_SEARCH", "0") == "1"
if ENABLE_SEARCH:
    @app.post("/v1/search")
    async def v1_search(req: SearchReq, authorization: t.Annotated[str | None, Header()] = None):
        _check_adapter_auth(authorization)
        if not await _ensure_local_ready():
            raise HTTPException(502, "Local search service unreachable after WOL")
        headers = _llm_headers()
        async with httpx.AsyncClient(timeout=TIMEOUT_LOCAL) as c:
            r = await c.post(f"{LOCAL_BASE}/search_web", json=req.dict(), headers=headers)
            if r.status_code >= 400:
                return JSONResponse(status_code=r.status_code, content=r.json())
            return JSONResponse(status_code=200, content=r.json())

@app.post("/v1/embeddings")
async def embeddings(req: EmbReq, authorization: t.Annotated[str | None, Header()] = None):
    _check_adapter_auth(authorization)
    texts = _as_list(req.input)
    # si un model est fourni et présent dans config, on peut l’ignorer ici car on force le local d'abord,
    # ou bien router selon cfg.get("embedding_model") == True (optionnel).
    vecs = await _embed(texts)
    return {"data": [{"embedding": v} for v in vecs]}

# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True}

# -----------------------------------------------------------------------------
# OCR bridge (relay to Flask)
# -----------------------------------------------------------------------------

@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile | None = File(default=None),
    file_url: str | None = Form(default=None),
    authorization: t.Annotated[str | None, Header()] = None,
    **extras,
):
    _check_adapter_auth(authorization)
    if not await _ensure_local_ready():
        raise HTTPException(502, "Local OCR unreachable after WOL")
    headers = _llm_headers()
    url = f"{LOCAL_BASE}/ocr"
    async with httpx.AsyncClient(timeout=TIMEOUT_LOCAL) as c:
        if file is not None:
            blob = await file.read()
            mt, _ = mimetypes.guess_type(file.filename or "")
            files = {"file": (file.filename or "upload.bin", blob, mt or "application/octet-stream")}
            r = await c.post(url, files=files, data=extras, headers=headers)
        else:
            data = {"file_url": file_url, **extras}
            r = await c.post(url, data=data, headers=headers)
        r.raise_for_status()
        return Response(content=r.content, media_type=r.headers.get("content-type", "application/json"))

@app.post("/ocr_auto")
async def ocr_auto_endpoint(
    payload: dict = Body(default={}),
    authorization: t.Annotated[str | None, Header()] = None,
):
    _check_adapter_auth(authorization)
    if not await _ensure_local_ready():
        raise HTTPException(502, "Local OCR unreachable after WOL")
    headers = _llm_headers()
    url = f"{LOCAL_BASE}/ocr_auto"
    async with httpx.AsyncClient(timeout=TIMEOUT_LOCAL) as c:
        r = await c.post(url, json=payload, headers=headers)
        r.raise_for_status()
        return Response(content=r.content, media_type=r.headers.get("content-type", "application/json"))




@app.get("/ocr_grid")
async def ocr_grid_endpoint(
    name: str | None = None,
    authorization: t.Annotated[str | None, Header()] = None,
):
    _check_adapter_auth(authorization)
    if not await _ensure_local_ready():
        raise HTTPException(502, "Local OCR grid unreachable after WOL attempt")

    headers = _llm_headers()
    url = f"{LOCAL_BASE}/ocr_grid"
    params = {"name": (name or "").strip()} if name else {}

    async with httpx.AsyncClient(timeout=TIMEOUT_LOCAL) as c:
        r = await c.get(url, params=params, headers=headers)
        # ⇧⇧⇧ Ce bloc DOIT être indenté par rapport au 'with'

        if r.status_code >= 400:
            # renvoyer proprement l’erreur du serveur Flask
            try:
                return JSONResponse(status_code=r.status_code, content=r.json())
            except Exception:
                return JSONResponse(status_code=r.status_code, content={"error": r.text})

        return Response(
            content=r.content,
            media_type=r.headers.get("content-type", "application/json"),
        )

@app.get("/ocr_history")
async def ocr_history_endpoint(
    limit: int | None = None,
    grid_name: str | None = None,
    doc_hash: str | None = None,
    since: str | None = None,
    authorization: t.Annotated[str | None, Header()] = None,
):
    _check_adapter_auth(authorization)
    if not await _ensure_local_ready():
        raise HTTPException(502, "Local OCR history unreachable after WOL")
    headers = _llm_headers()
    url = f"{LOCAL_BASE}/ocr_history"
    params = {}
    if limit is not None: params["limit"] = limit
    if grid_name: params["grid_name"] = grid_name
    if doc_hash: params["doc_hash"] = doc_hash
    if since: params["since"] = since
    async with httpx.AsyncClient(timeout=TIMEOUT_LOCAL) as c:
        r = await c.get(url, params=params, headers=headers)
        r.raise_for_status()
        return Response(content=r.content, media_type=r.headers.get("content-type", "application/json"))


async def _relay_ocr(
    path: str,
    file: Optional[UploadFile],
    file_url: Optional[str],
    extra: Dict[str, Any],
):
    if not file and not file_url:
        raise HTTPException(status_code=400, detail="Provide file or file_url")
    if file and file_url:
        raise HTTPException(status_code=400, detail="Provide either file OR file_url, not both")

    if not await _ensure_local_ready():
        raise HTTPException(status_code=502, detail="Local OCR service unreachable after WOL attempt")

    data = {k: v for k, v in (extra or {}).items() if v is not None}
    files = None
    if file:
        content = await file.read()
        files = [("file", (file.filename or "upload.bin", content, file.content_type or "application/octet-stream"))]
    if file_url:
        data["file_url"] = file_url

    url = f"{LOCAL_BASE}{path}"
    try:
        async with httpx.AsyncClient(timeout=OCR_TIMEOUT) as client:
            r = await client.post(url, data=data or None, files=files, headers=_llm_headers())
            ct = (r.headers.get("content-type") or "").lower()

            if r.status_code >= 400:
                try:
                    return JSONResponse(status_code=r.status_code, content=r.json())
                except Exception:
                    return JSONResponse(status_code=r.status_code, content={"error": r.text})

            if "application/" in ct or "text/csv" in ct or "application/pdf" in ct:
                disp = r.headers.get("content-disposition", "inline")
                return Response(content=r.content, media_type=ct or "application/octet-stream",
                                headers={"Content-Disposition": disp})

            return Response(content=r.content, media_type=ct or "application/json")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OCR upstream error: {e}")


@app.post("/ocr/convert")
async def ocr_convert(
    format: str = Form(...),
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    authorization: Annotated[str | None, Header()] = None,
):
    _check_adapter_auth(authorization)
    allowed = {"text", "csv", "docx", "searchable_pdf"}
    if format not in allowed:
        raise HTTPException(status_code=400, detail="Invalid format")

    files = []
    data = {"format": format}
    if file:
        content = await file.read()
        files.append(("file", (file.filename or "upload.bin", content, file.content_type or "application/octet-stream")))
    if file_url:
        data["file_url"] = file_url

    url = f"{LOCAL_BASE}{LOCAL_OCR_PATH}"
    try:
        async with httpx.AsyncClient(timeout=OCR_TIMEOUT) as client:
            r = await client.post(url, data=data, files=files or None, headers=_llm_headers())
            ct = r.headers.get("content-type", "")
            if "application/" in ct or "text/csv" in ct:
                return Response(content=r.content, media_type=ct,
                                headers={"Content-Disposition": r.headers.get("content-disposition", "inline")})
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OCR upstream error: {e}")

# -----------------------------------------------------------------------------
# Transcriptions
#------------------------------------------------------------------------------

@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    authorization: t.Annotated[str | None, Header()] = None,
    # multipart
    file: UploadFile | None = File(default=None),
    model: str | None = Form(default=None),      # compat OpenAI, non utilisé
    language: str | None = Form(default=None),
    timestamps: bool | None = Form(default=None),
    diarize: bool | None = Form(default=None),
    chunk: int | None = Form(default=None),
    stride: int | None = Form(default=None),
    summarize: bool | None = Form(default=False),
    summary_prompt: str | None = Form(default=None),
    return_html: bool | None = Form(default=False),
    llm_model: str | None = Form(default="annoter_rag_vecteur"),
    file_url_form: str | None = Form(default=None),
    metadata_form: str | None = Form(default=None),
    speakers_form: int | None = Form(default=None),
    min_speaker_duration_form: float | None = Form(default=None),
    collar_form: float | None = Form(default=None),
    allow_overlap_form: bool | None = Form(default=None),
    # JSON fallback
    json_body: TranscribeReq | None = Body(default=None),
):
    _check_adapter_auth(authorization)
    if not await _ensure_local_ready():
        raise HTTPException(502, "Local service unreachable after WOL")

    # Unifier params (multipart vs JSON)
    if json_body:
        language = json_body.language or language
        timestamps = json_body.timestamps if json_body.timestamps is not None else timestamps
        diarize = json_body.diarize if json_body.diarize is not None else diarize
        chunk = json_body.chunk or chunk
        stride = json_body.stride or stride
        summarize = json_body.summarize if json_body.summarize is not None else summarize
        summary_prompt = json_body.summary_prompt or summary_prompt
        return_html = json_body.return_html if json_body.return_html is not None else return_html
        llm_model = json_body.llm_model or llm_model
        file_url = json_body.file_url or file_url_form
        meta = dict(json_body.metadata or {})
        speakers = json_body.speakers if json_body.speakers is not None else speakers_form
        min_speaker_duration = json_body.min_speaker_duration if json_body.min_speaker_duration is not None else min_speaker_duration_form
        collar = json_body.collar if json_body.collar is not None else collar_form
        allow_overlap = json_body.allow_overlap if json_body.allow_overlap is not None else allow_overlap_form
    else:
        file_url = file_url_form
        meta = {}
        if metadata_form:
            try:
                meta.update(json.loads(metadata_form))
            except Exception:
                pass
        speakers = speakers_form
        min_speaker_duration = min_speaker_duration_form
        collar = collar_form
        allow_overlap = allow_overlap_form
    # 1) Transcription via /asr_voxtral
    headers = _llm_headers()
    asr_url = f"{LOCAL_BASE}/asr_voxtral"
    asr_params = {
        **({"lang": language} if language else {}),
        **({"timestamps": timestamps} if timestamps is not None else {}),
        **({"diarize": diarize} if diarize is not None else {}),
        **({"chunk": chunk} if chunk else {}),
        **({"stride": stride} if stride else {}),
    }
    # diar_options AVANT l'appel réseau
    diar_opts = {}
    if diarize:
        if speakers is not None:
            diar_opts["max_speakers"] = int(speakers)
        if min_speaker_duration is not None:
            diar_opts["min_speaker_duration"] = float(min_speaker_duration)
        if collar is not None:
            diar_opts["collar"] = float(collar)
        if allow_overlap is not None:
            diar_opts["allow_overlap"] = bool(allow_overlap)
    if diar_opts:
        asr_params["diar_options"] = diar_opts

    async with httpx.AsyncClient(timeout=TIMEOUT_LOCAL) as c:
        if file is not None:
            blob = await file.read()
            mt, _ = mimetypes.guess_type(file.filename or "")
            files = {"file": (file.filename or "audio.bin", blob, mt or "application/octet-stream")}
            r = await c.post(asr_url, params=asr_params, files=files, headers=headers)
        else:
            # par URL distante (attention à ALLOWED_WEB_DOMAINS si tu filtres)
            if file_url and hasattr(globals(), "_is_allowed_url") and not _is_allowed_url(file_url):
                raise HTTPException(403, f"URL non autorisée : {file_url}")
            data = {"file_url": file_url} if file_url else {}
            r = await c.post(asr_url, params=asr_params, data=data, headers=headers)

        r.raise_for_status()
        asr_json = r.json()

    transcript = asr_json.get("text") or asr_json.get("transcript") or ""
    if not transcript.strip():
        raise HTTPException(502, "ASR returned empty transcript")


    # Si pas de résumé demandé → on renvoie comme OpenAI
    if not summarize:
        return {
            "task": "transcribe",
            "language": language or asr_json.get("language", "auto"),
            "duration": asr_json.get("duration"),
            "text": transcript,
            "chunks": asr_json.get("chunks"),
            "srt_path": asr_json.get("srt_path"),
            "vtt_path": asr_json.get("vtt_path"),
        }

    # 2) Résumé via LLM local (par défaut annoter_rag_vecteur)
    # prompt par défaut sobre et efficace
    default_summary_prompt = (
        "Résume clairement la transcription suivante en points clés et propose 3 actions concrètes. "
        "Structure: TL;DR, Points clés (•), Actions (1..3)."
    )
    user_prompt = summary_prompt or default_summary_prompt
    final_prompt = f"{user_prompt}\n\nTRANSCRIPTION:\n{transcript}"

    # on peut pousser return_html comme méta (compat annoter_web/annoter)
    meta_llm = meta.copy()
    if return_html:
        meta_llm["return_html"] = True

    out = await _local_chat(final_prompt, route_hint=llm_model or "annoter_rag_vecteur", temperature=None, meta=meta_llm)

    return {
        "task": "transcribe+summarize",
        "text": transcript,
        "summary": out,
        "model": llm_model or "annoter_rag_vecteur",
        "created": int(time.time()),
    }



# -----------------------------------------------------------------------------
# OpenAI Images → pont ComfyUI
#------------------------------------------------------------------------------

@app.post("/v1/images/generations")
async def images_generations(req: ImageGenReq, authorization: t.Annotated[str | None, Header()] = None):
    _check_adapter_auth(authorization)
    if not await _ensure_local_ready():
        raise HTTPException(status_code=502, detail="Local image service unreachable after WOL attempt")
    headers = _llm_headers()
    payload = {
        "prompt": req.prompt,
        "n": req.n or 1,
        "size": req.size or "1024x1024",
        **(req.metadata or {}),
    }
    async with httpx.AsyncClient(timeout=TIMEOUT_LOCAL) as c:
        r = await c.post(f"{LOCAL_BASE}{COMFY_PREFIX}/prompt", json=payload, headers=headers)
        r.raise_for_status()
        j = r.json()
    # normalisation OpenAI (urls/base64)
    urls = []
    if isinstance(j, dict):
        urls = j.get("urls") or j.get("images") or []
    if not isinstance(urls, list):
        urls = [urls]
    data = [{"url": u} for u in urls][: (req.n or 1)]
    return {"created": int(time.time()), "data": data}


# -----------------------------------------------------------------------------
# Files proxy (vers Flask)
# -----------------------------------------------------------------------------
@app.post("/files")
async def upload_file(
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    authorization: Annotated[str | None, Header()] = None,
):
    _check_adapter_auth(authorization)
    if not await _ensure_local_ready():
        raise HTTPException(502, "Local file API unreachable after WOL attempt")
    if not file and not file_url:
        raise HTTPException(400, "Provide file or file_url")
    if file and file_url:
        raise HTTPException(400, "Provide either file OR file_url, not both")

    llm_path = "/files"
    url = f"{LOCAL_BASE}{llm_path}"

    data = {}
    files = None
    if file:
        content = await file.read()
        files = [("file", (file.filename or "upload.bin", content, file.content_type or "application/octet-stream"))]
    else:
        data["file_url"] = file_url

    try:
        async with httpx.AsyncClient(timeout=OCR_TIMEOUT) as client:
            r = await client.post(url, data=data or None, files=files, headers=_llm_headers())
            if r.status_code >= 400:
                return JSONResponse(status_code=r.status_code, content={"error": r.text})
            ct = r.headers.get("content-type", "application/json")
            return Response(content=r.content, media_type=ct)
    except Exception as e:
        raise HTTPException(502, f"Upload upstream error: {e}")

@app.get("/files/{file_id}")
async def get_file(
    file_id: str,
    authorization: Annotated[str | None, Header()] = None,
):
    _check_adapter_auth(authorization)
    if not await _ensure_local_ready():
        raise HTTPException(502, "Local file API unreachable after WOL attempt")
    llm_path = f"/files/{file_id}"
    url = f"{LOCAL_BASE}{llm_path}"
    try:
        async with httpx.AsyncClient(timeout=OCR_TIMEOUT) as client:
            r = await client.get(url, headers=_llm_headers(), timeout=OCR_TIMEOUT)
            if r.status_code >= 400:
                return JSONResponse(status_code=r.status_code, content={"error": r.text})
            ct = r.headers.get("content-type", "application/octet-stream")
            disp = r.headers.get("content-disposition", "attachment")
            return StreamingResponse(_stream_bytes(r), media_type=ct, headers={"Content-Disposition": disp})
    except Exception as e:
        raise HTTPException(502, f"Download upstream error: {e}")

async def _stream_bytes(resp: httpx.Response, chunk_size: int = 1024 * 256):
    async for chunk in resp.aiter_bytes(chunk_size=chunk_size):
        yield chunk



#==============================================================
# Fonctions bases de données devenues inutiles
#==============================================================

async def _rag_context(query: str) -> str:
    vec = (await _embed([query]))[0]
    # Qdrant prioritaire si défini, sinon Chroma
    if QDRANT_URL and QDRANT_COLLECTION:
        headers = {}
        if QDRANT_API_KEY:
            headers["api-key"] = QDRANT_API_KEY
        payload = {"vector": vec, "limit": RAG_TOP_K, "with_payload": True}
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search",
                                  json=payload, headers=headers, timeout=TIMEOUT_LOCAL)
            r.raise_for_status()
            j = r.json()
        chunks = []
        for p in j.get("result", []):
            pl = p.get("payload", {})
            txt = pl.get("text") or pl.get("content") or ""
            if txt:
                chunks.append(txt)
        return "\n\n".join(chunks)
    if CHROMA_URL and CHROMA_COLLECTION:
        # Chroma: query via HTTP API
        qvecs = [vec]
        payload = {
            "collection": CHROMA_COLLECTION,
            "query_embeddings": qvecs,
            "n_results": RAG_TOP_K,
        }
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{CHROMA_URL}/api/v1/query", json=payload, timeout=TIMEOUT_LOCAL)
            r.raise_for_status()
            j = r.json()
        docs = j.get("documents") or []
        if docs and len(docs) and len(docs[0]):
            return "\n\n".join(docs[0])
    return ""

def _want_rag(meta: dict | None) -> bool:
    if RAG_MODE == "always":
        return True
    if RAG_MODE == "on_tool":
        return bool(meta and meta.get("rag"))
    return False

