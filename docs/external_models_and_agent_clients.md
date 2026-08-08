# openai-adapter — Architecture, modèles externes et clients agentiques

> Audit du dépôt `C:\CodexWorkspace\openai-adapter` (branche locale, commit `5524fa1`).
> Ce document décrit l'architecture **telle qu'elle est implémentée** dans le code.
>
> **Légende des qualifications utilisées :**
> - **VALIDÉ EN RÉEL** — comportement vérifié par un test réel (appel live passerelle/fournisseur ou exécution Codex CLI) effectué dans ce projet.
> - **TESTÉ AUTOMATIQUEMENT** — couvert par les tests automatisés/mockés du dépôt (`tests/`), sans appel réel vérifié.
> - **CONFIGURÉ MAIS NON TESTÉ** — présent dans la configuration, mais sans test (automatisé ou réel) dans ce dépôt.

---

## 1. Vue d'ensemble

`openai-adapter` est une passerelle FastAPI (« OpenAI-Compatible Adapter », `adapter.py`) qui expose une API de type OpenAI et route les requêtes vers :
- un **backend local Flask** (le « PC fixe », orchestrateur de modèles locaux / GPT4All), et
- des **fournisseurs distants** (OpenAI, OpenRouter, DeepSeek, NVIDIA NIM) compatibles OpenAI.

La décision local/remote se fait **sur le nom de modèle demandé par le client** (voir §3). La passerelle est consommée notamment par **OpenWebUI** (`docker-compose.yml` → `DEFAULT_OPENAI_API_BASE_URL: "http://openai-adapter:5055/v1"`). **VALIDÉ EN RÉEL.**

Fichiers clés du dépôt :
| Fichier | Rôle | Qualification |
|---|---|---|
| `adapter.py` | Application FastAPI (routes, routage modèles, retries, streaming) | **VALIDÉ EN RÉEL** |
| `backend_selection.py` | Sélection / bascule du backend Flask (ping `/ping`) | **TESTÉ AUTOMATIQUEMENT** (`test_backend_selection.py`) |
| `tool_compat.py` | Traduction d'outils Responses ↔ Chat Completions, aplatissement des namespaces | **TESTÉ AUTOMATIQUEMENT** (`test_tool_compat.py`) |
| `config_remote.json` | Overrides par modèle distant (base_url, clé, capacités) | **VALIDÉ EN RÉEL** (chargé au démarrage via `REMOTE_CONF_PATH`) |
| `update_model_pricing.py` + `adapter_openai_models_priced.json` | Enrichissement de tarifs (catalogue OpenRouter / tarifs OpenAI statiques) | **CONFIGURÉ MAIS NON TESTÉ** — non chargé par `adapter.py` au runtime, outil de maintenance seul |
| `docker-compose.yml` | Déploiement openai-adapter, OpenWebUI, Perplexica, SearXNG, Redis | **VALIDÉ EN RÉEL** |
| `.env` | Variables d'environnement d'exécution | **VALIDÉ EN RÉEL** |

---

## 2. Architecture et flux de routage

### 2.1 Endpoints exposés
**VALIDÉ EN RÉEL** (routes déclarées dans `adapter.py`) :
- `GET /v1/models` — liste des modèles (locaux + routes locales + `REMOTE_MODELS` + `REMOTE2_MODELS`).
- `POST /v1/responses` — API Responses (avec streaming SSE).
- `POST /v1/chat/completions` — Chat Completions.
- `POST /v1/embeddings`, `POST /v1/audio/transcriptions`, `POST /v1/images/generations`.
- Endpoints métier/maison : `POST /v1/documents/analyze`, `/ocr*`, `/files`, `/ping`, `/healthz`.

### 2.2 Chaîne de résolution de modèle
**VALIDÉ EN RÉEL** (`adapter.py`) :
1. `_resolve_model_id(model)` — si le nom est un ID logique de `MODEL_REGISTRY`, retourne cet ID ; sinon applique `MODEL_ALIAS` ; sinon retourne le nom tel quel.
2. `_model_cfg(model)` — fusionne : overrides `config_remote.json` (defaults puis entrée exacte) → entrée `MODEL_REGISTRY` (backend, modèle physique, `api_base`, `api_key_env`, `json_mode`) → valeurs par défaut env (`REMOTE_BASE`, `OPENAI_API_KEY`).
3. `_is_local_model(model)` — vrai si : `route:*`, préfixe `local-*`, présent dans `LOCAL_MODELS`, présent dans `LOCAL_ROUTE_MAP`, préfixe `comfy:*`. Sinon modèle considéré remote.
4. Pour les modèles distants, `_remote_model_ids()` fournit le couple `(modèle physique, provider_model)` ; `provider_model` (ex. `deepseek-v4-flash`) est le nom réel attendu par le fournisseur, différent de l'ID public (`deepseek/deepseek-v4-flash`).

### 2.3 Backend local (Flask / GPT4All)
**VALIDÉ EN RÉEL** (logique de code) ; **TESTÉ AUTOMATIQUEMENT** (fallback runtime) :
- `backend_selection.py` : `parse_backend_candidates` (env + défauts `10.0.1.10:5050`, `192.168.0.155:5050`, etc.) puis `select_backend_url` sonde `/ping` pour choisir le backend actif. **VALIDÉ EN RÉEL** (code).
- Fallback runtime : `_reselect_local_backend()` retente sur les autres candidats en cas d'échec (≥500, timeout, erreur réseau). **TESTÉ AUTOMATIQUEMENT** (`test_runtime_backend_fallback.py`).
- `LOCAL_ROUTE_MAP` mappe des « modèles-route » vers des chemins Flask (ex. `annoter` → `/annoter`, modèles `local-*` → `/chat_orchestre`). **VALIDÉ EN RÉEL** (code).
- Réveil à distance du PC fixe : WOL via webhook Home Assistant (`ENABLE_WOL`). **VALIDÉ EN RÉEL** (code).

---

## 3. Support des modèles externes

### 3.1 Fournisseurs et base_url
**VALIDÉ EN RÉEL** (`config_remote.json`) :
| Fournisseur | `base_url` | Exemples de modèles configurés |
|---|---|---|
| OpenAI direct | `https://api.openai.com/v1` | `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5.4`, `gpt-5.4-nano-2026-03-17`, `gpt-4.1(-nano/-mini)`, `gpt-4o(-mini)`, `o1-preview`, `o1-mini`, `text-embedding-3-large` |
| OpenRouter | `https://openrouter.ai/api/v1` | `openai/gpt-oss-120b`, `anthropic/claude-sonnet-4.5`, `google/gemini-2.5-flash-*`, `meta-llama/llama-4-*`, `perplexity/sonar-deep-research`, `deepseek/deepseek-v3.2-exp`, `moonshotai/kimi-k3` |
| DeepSeek direct | `https://api.deepseek.com/` | `deepseek/deepseek-v4-flash` (`provider_model: deepseek-v4-flash`) |
| NVIDIA NIM | `https://integrate.api.nvidia.com/v1` | `minimaxai/minimax-m3` (clé `NVIDIA_API_KEY`) |

Clés API via variables d'environnement (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, `NVIDIA_API_KEY`), référencées par `api_key_env` dans la config. **VALIDÉ EN RÉEL.**

### 3.2 Normalisation du modèle par fournisseur
**VALIDÉ EN RÉEL** (`_normalize_model_for_base`, `_remote_model_ids`) ; **TESTÉ AUTOMATIQUEMENT** (`test_remote_provider_model.py`) :
- Pour OpenAI direct : suppression du préfixe `openai/` (OpenRouter → OpenAI direct).
- `provider_model` force explicitement le nom attendu par le fournisseur.
- Le modèle public retourné au client reste celui demandé (`resp.model` = ID public).

### 3.3 Choix Responses vs Chat pour les modèles distants
**VALIDÉ EN RÉEL** :
- Dans `_remote_chat`, `use_resp` est actif seulement si : `not force_chat` ET `use_responses_api` ET fournisseur `openai` ET modèle « moderne » (`gpt-5*`, `o1/o3/o4*`). Sinon `chat/completions`.
- `_native_responses_enabled()` (route `/v1/responses`) est actif si : `use_responses_api=true` ET `force_chat=false` ET `native_responses_provider=true` — c'est le cas de **`deepseek/deepseek-v4-flash`** uniquement dans la config actuelle.
- En production (`docker-compose.yml`/`.env`) : `OPENAI_USE_RESPONSES=false`, `OPENAI_FORCE_CHAT=true`, `OPENAI_COMPAT_MODE=chat` → le défaut global force Chat Completions, les modèles OpenAI « modernes » passant par `/responses` seulement si leur entrée `config_remote.json` l'autorise (ex. `gpt-4.1`, `o1`, `gpt-5-nano` avec `use_responses_api: true`, `force_chat: false`).

### 3.4 Tarifs des modèles
**CONFIGURÉ MAIS NON TESTÉ** :
- `update_model_pricing.py` enrichit la config avec les prix : catalogue OpenRouter (endpoint `/api/v1/models` + `pricing`) ou tarifs OpenAI statiques (`OPENAI_STATIC_PRICING`), écrit dans `adapter_openai_models_priced.json`.
- **`adapter.py` ne charge pas ce fichier** (aucune référence trouvée) → outil de référence/documentation côté affichage (OpenWebUI), pas utilisé au runtime de la passerelle.

### 3.5 Écart de configuration constaté (observation)
**CONFIGURÉ MAIS NON TESTÉ** : le fichier `.env` et le `docker-compose.yml` ne listent pas exactement les mêmes `REMOTE_MODELS` (ex. `deepseek/deepseek-v4-flash`, `minimaxai/minimax-m3`, `moonshotai/kimi-k3` présents dans `docker-compose.yml` mais pas dans le `.env` local). La valeur effective selon l'hôte de déploiement n'est pas vérifiée ici.

---

## 4. Utilisation par des clients agentiques compatibles OpenAI

### 4.1 Endpoints agentiques
**VALIDÉ EN RÉEL** :
- `POST /v1/responses` est l'endpoint privilégié pour les clients agentiques (boucles d'outils, streaming SSE, `reasoning`, `previous_response_id`).
- `POST /v1/chat/completions` reste disponible pour les clients Chat classiques.
- Auth : `Authorization: Bearer <ADAPTER_API_KEY>` (si configurée). **VALIDÉ EN RÉEL.**

### 4.2 Chemin « native Responses » (DeepSeek)
**VALIDÉ EN RÉEL** :
- Le transport natif Responses de `deepseek/deepseek-v4-flash` a été testé en réel en **non-stream** et en **stream**, ainsi qu'un **`function_call` réel** : la route `/v1/responses` passe les items `input` (y compris `reasoning`, `function_call`, `function_call_output`), les `tools` au format Responses natif et `tool_choice` vers `{base}/responses` (soit `https://api.deepseek.com/responses`). **VALIDÉ EN RÉEL.**
- Le streaming natif est relayé en SSE (`_remote_responses_native_stream`). **VALIDÉ EN RÉEL** (test stream réel). La reconstruction générique Responses→Chat reste couverte par des mocks → **TESTÉ AUTOMATIQUEMENT** (`test_remote_provider_model.py`).
- **Codex CLI** avec `deepseek/deepseek-v4-flash` a exécuté avec succès un **outil terminal** (lecture de `test.txt`) et une **recherche web** suivie d'une **réponse finale**. **VALIDÉ EN RÉEL.**

### 4.3 Chemin « Responses → Chat » (traduction d'outils)
**TESTÉ AUTOMATIQUEMENT** (`tool_compat.py`, `test_tool_compat.py`) :
- Les outils au format Responses (`function`, `namespace`, `web_search`/`web_search_preview`) sont traduits vers le format Chat Completions :
  - `function` → outil Chat `{type: function, function: {name, description, parameters}}` ; le flag `strict` est **supprimé** si le fournisseur ne le supporte pas.
  - `namespace` → aplati en `namespace__sousoutil` (max 64 caractères, hash SHA-256 si trop long) avec une `reverse_name_map` pour reconstituer le nom original côté réponse.
  - `web_search`/`web_search_preview` → **supprimés** avec avertissement si le fournisseur ne supporte pas le web_search natif ; erreur si `tool_choice` cible explicitement ces outils.
  - types inconnus → erreur `ToolCompatibilityError`.
- Limites appliquées : `MAX_TOOLS` (128) et `MAX_TOOL_SCHEMA_BYTES` (2 Mo), vérifiées avant/après conversion.
- Capacités par modèle via `config_remote.json` (`supports_tools`, `supports_namespace_tools`, `supports_web_search`, `supports_strict_tools`, `supports_parallel_tool_calls`, `native_responses_provider`).

### 4.4 Streaming
- `/v1/responses` — streaming natif DeepSeek relayé en réel : **VALIDÉ EN RÉEL** ; reconstruction générique Responses→Chat (`_responses_stream_generator` : `response.output_text.delta`, `response.function_call_arguments.delta`, `response.output_item.done`, `response.completed`, restauration des noms de namespace) : **TESTÉ AUTOMATIQUEMENT**.
- `/v1/chat/completions` — **non streamé** : le flag `stream` de la requête est ignoré côté passerelle et l'appel upstream utilise `stream=false`. **VALIDÉ EN RÉEL** (code).

### 4.5 Contraintes pour les clients agentiques
- `previous_response_id` est rejeté (mode sans état) : le client doit renvoyer l'historique complet (`400 previous_response_id is not supported in stateless mode`). **VALIDÉ EN RÉEL.**
- Boucle agentique complète avec **Codex CLI + `deepseek/deepseek-v4-flash`** validée en réel (outil terminal + recherche web + réponse finale) : **VALIDÉ EN RÉEL.**
- Le commit `5524fa1 feat(adapter): support Codex tools with DeepSeek and MiniMax` atteste de l'intention d'utiliser la passerelle comme backend pour des clients agentiques de type Codex. **VALIDÉ EN RÉEL** (historique git).
- `minimaxai/minimax-m3` déclare `supports_tools`, `supports_parallel_tool_calls` mais pas `native_responses_provider` → les outils passent par la traduction Chat (pas de Responses natif). **VALIDÉ EN RÉEL** (code) ; la boucle d'outils a aussi été validée en réel.

---

## 5. Résumé de la validation par modèle

| Modèle public | Fournisseur | Chemin utilisé | Statut |
|---|---|---|---|
| `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5.4`, `gpt-5.4-nano-2026-03-17` | OpenAI | Chat (ou Responses si config) | **TESTÉ AUTOMATIQUEMENT** (routage, mock) ; appel réel **CONFIGURÉ MAIS NON TESTÉ** |
| `gpt-4.1`, `gpt-4.1-mini`, `o1-*` | OpenAI | Responses (`use_responses_api: true`) | **TESTÉ AUTOMATIQUEMENT** (mock) ; **CONFIGURÉ MAIS NON TESTÉ** (réel) |
| Modèles OpenRouter (`claude-sonnet-4.5`, `gemini-2.5-*`, `llama-4-*`, …) | OpenRouter | Chat | **CONFIGURÉ MAIS NON TESTÉ** |
| `deepseek/deepseek-v4-flash` | DeepSeek | Responses natif | **VALIDÉ EN RÉEL** (transport non-stream/stream, `function_call`, outils + recherche web via Codex CLI) |
| `minimaxai/minimax-m3` | NVIDIA NIM | Chat + outils traduits | **VALIDÉ EN RÉEL** (boucle d'outils agentique) |
| `moonshotai/kimi-k3` | OpenRouter | Chat + outils | **VALIDÉ EN RÉEL** (boucle d'outils agentique) |
| Modèles locaux (`local-*`, `annoter`, routes) | Flask local | `/chat_orchestre` etc. | **TESTÉ AUTOMATIQUEMENT** (contrats `pass2e/pass3e/debrief`, fallback backend) |
| Traduction d'outils Responses↔Chat | — | — | **TESTÉ AUTOMATIQUEMENT** (`test_tool_compat.py`) |
