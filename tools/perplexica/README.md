# Perplexica Tools

Small Python tools for Perplexica 1.12.1. They do not require secrets and only use
Perplexica HTTP endpoints needed for chat creation and reading.

## Existing Chat Export

```bash
python perplexica_chat_export.py CHAT_ID --base-url http://perplexica-host:3000
python perplexica_chat_export.py CHAT_ID --output resultat.json
```

The base URL is resolved in this order:

1. `--base-url`
2. `PERPLEXICA_URL`

The exporter only performs:

```text
GET /api/chats/<chatId>
```

## Minimal Job Runner

```bash
python perplexica_job.py --prompt "Quelle est la capitale de la France ?" --base-url http://perplexica-host:3000
python perplexica_job.py --prompt-file prompts/test.txt
```

The job runner performs:

```text
GET /api/providers
POST /api/chat
GET /api/chats/<chatId>
```

It saves a JSON file in `tools/output/` and prints a short summary containing
the chat id, message id, source count, cited source count, and output path.

Optional model overrides are available when automatic provider selection is not
enough:

```bash
python perplexica_job.py --prompt "test" \
  --chat-model-provider-id PROVIDER_ID --chat-model-key MODEL_KEY \
  --embedding-model-provider-id PROVIDER_ID --embedding-model-key MODEL_KEY
```

## Output

The canonical JSON includes message ids, original Markdown answer text, all
flattened sources, cited sources, citation numbers, citation counts, unresolved
citation numbers, status, and timestamps.
