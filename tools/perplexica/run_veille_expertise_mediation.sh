#!/bin/sh

# POSIX wrapper for the Perplexica expertise/mediation watch job.
# No secret is stored here; runtime configuration stays in environment/.env files.

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" 2>/dev/null && pwd)
if [ -z "$SCRIPT_DIR" ]; then
    echo "Cannot determine script directory" >&2
    exit 1
fi

cd "$SCRIPT_DIR" || exit 1

PYTHON_BIN=${PERPLEXICA_PYTHON:-${PYTHON:-}}

if [ -z "$PYTHON_BIN" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=$(command -v python3)
    else
        echo "python3 not found; set PERPLEXICA_PYTHON or PYTHON" >&2
        exit 127
    fi
fi

if [ ! -x "$PYTHON_BIN" ]; then
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        PYTHON_BIN=$(command -v "$PYTHON_BIN")
    else
        echo "Configured Python is not executable: $PYTHON_BIN" >&2
        exit 127
    fi
fi

"$PYTHON_BIN" run_perplexica_mail_job.py --job jobs/veille_expertise_mediation.json "$@"
exit $?
