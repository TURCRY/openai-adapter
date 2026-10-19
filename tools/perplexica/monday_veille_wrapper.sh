#!/bin/sh
# Monday veille wrapper: never launches the heavy Perplexica/Gemma job more
# than once per day. Each cron slot is independent; a daily marker plus flock
# guard the heavy job, and a light SearXNG check decides whether to launch it.
set -u

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" 2>/dev/null && pwd)
if [ -z "$SCRIPT_DIR" ]; then
    echo "monday_veille_wrapper: cannot resolve script directory" >&2
    exit 1
fi

# --- config (overridable via environment) ---
PYTHON_BIN=${PERPLEXICA_PYTHON:-${PYTHON:-}}
if [ -z "$PYTHON_BIN" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=$(command -v python3)
    else
        echo "monday_veille_wrapper: python3 not found" >&2
        exit 127
    fi
fi

HELPER="$SCRIPT_DIR/monday_veille_light.py"
JOB_WRAPPER="$SCRIPT_DIR/run_veille_expertise_mediation.sh"
JOB_OUTPUT_ROOT="$SCRIPT_DIR/output/jobs/veille_expertise_mediation"
WRAPPER_OUTPUT="$SCRIPT_DIR/output/veille_wrapper"
MARKER_DIR="$WRAPPER_OUTPUT/markers"
LOG_DIR="$WRAPPER_OUTPUT/logs"
LOCK_FILE="$WRAPPER_OUTPUT/.lock"
MAIL_ENV="$SCRIPT_DIR/.env"
SEARXNG_URL="${VEILLE_SEARXNG_URL:-http://127.0.0.1:18081/search?q=expertise%20judiciaire&format=json}"
SEARXNG_TIMEOUT="${VEILLE_SEARXNG_TIMEOUT:-25}"
LAST_SLOT="${VEILLE_LAST_SLOT:-07:45}"
RECOVERY_ONLY="${VEILLE_RECOVERY_ONLY:-0}"
FAILURE_SUBJECT="${VEILLE_FAILURE_SUBJECT:-Veille hebdomadaire — échec (SearXNG indisponible)}"
FAILURE_BODY="${VEILLE_FAILURE_BODY:-Veille non exécutée : SearXNG indisponible / aucun moteur productif.}"

TODAY=${VEILLE_TODAY:-$(date +%F)}
WEEKDAY=${VEILLE_WEEKDAY:-$(date +%u)}
HOUR=${VEILLE_HOUR:-$(date +%H)}
SLOT=${VEILLE_SLOT:-$(date +%H:%M)}
LOG_FILE="$LOG_DIR/$TODAY.log"

mkdir -p "$MARKER_DIR" "$LOG_DIR" "$WRAPPER_OUTPUT/mails"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*" | tee -a "$LOG_FILE"
}

# flock: never run two wrapper instances concurrently
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    log "lock: another wrapper instance is running; exit."
    exit 0
fi

CONTEXT=$("$PYTHON_BIN" "$HELPER" business-context \
    --today "$TODAY" --weekday "$WEEKDAY" --hour "$HOUR" 2>&1)
CONTEXT_RC=$?
if [ "$CONTEXT_RC" -ne 0 ]; then
    log "business-context failed rc=$CONTEXT_RC: $CONTEXT; fail-closed, no run."
    exit 0
fi
MODE=$(printf '%s\n' "$CONTEXT" | sed -n 's/^mode=//p' | head -n 1)
DATE=$(printf '%s\n' "$CONTEXT" | sed -n 's/^business_date=//p' | head -n 1)
case "$MODE" in
    normal|recovery|recovery_closed) ;;
    *)
        log "business-context returned invalid mode '$MODE'; fail-closed, no run."
        exit 0
        ;;
esac
if [ -z "$DATE" ]; then
    log "business-context returned no business_date; fail-closed, no run."
    exit 0
fi
if [ "$MODE" = "recovery_closed" ]; then
    log "start today=$TODAY weekday=$WEEKDAY hour=$HOUR slot=$SLOT mode=recovery_closed business_date=$DATE; outside recovery window, no run."
    exit 0
fi
if [ "$RECOVERY_ONLY" = "1" ] && [ "$MODE" = "normal" ]; then
    MODE=recovery
fi

log "start today=$TODAY weekday=$WEEKDAY hour=$HOUR slot=$SLOT mode=$MODE business_date=$DATE"

# 1) daily marker guard
MARKER=$("$PYTHON_BIN" "$HELPER" marker-status "$MARKER_DIR" "$DATE" 2>/dev/null || echo none)
case "$MARKER" in
    ok)
        log "marker: $DATE.ok present (veille already done today); exit."
        exit 0
        ;;
    partial)
        log "marker: $DATE.partial present; checking editorial resume."
        CANDIDATE=$("$PYTHON_BIN" "$HELPER" resume-candidate "$MARKER_DIR" "$JOB_OUTPUT_ROOT" "$DATE" 2>&1)
        CANDIDATE_RC=$?
        log "resume-candidate rc=$CANDIDATE_RC: $CANDIDATE"
        ACTION=$(printf '%s\n' "$CANDIDATE" | sed -n 's/^action=//p' | head -n 1)
        RUN_ID=$(printf '%s\n' "$CANDIDATE" | sed -n 's/^run_id=//p' | head -n 1)
        RUN_DIR=$(printf '%s\n' "$CANDIDATE" | sed -n 's/^run_dir=//p' | head -n 1)
        if [ "$CANDIDATE_RC" -ne 0 ]; then
            log "resume-candidate failed; keeping .partial for next slot."
            exit 0
        fi
        case "$ACTION" in
            resume)
                if [ -z "$RUN_DIR" ]; then
                    log "resume-candidate returned resume without run_dir; keeping .partial."
                    exit 0
                fi
                RESUME_LOG="$LOG_DIR/resume_${TODAY}_${SLOT}.log"
                (
                    cd "$SCRIPT_DIR" || exit 1
                    sh "$JOB_WRAPPER" --resume-editorial-run "$RUN_DIR"
                ) >"$RESUME_LOG" 2>&1
                RESUME_RC=$?
                log "editorial resume rc=$RESUME_RC run_id=$RUN_ID log=$RESUME_LOG"
                if [ -f "$MARKER_DIR/$DATE.ok" ]; then
                    log "editorial resume completed: $MARKER_DIR/$DATE.ok present."
                else
                    log "editorial resume incomplete; .partial kept for next slot."
                fi
                exit 0
                ;;
            already_sent)
                log "editorial already sent for run ${RUN_ID:-unknown}; no resend."
                exit 0
                ;;
            smtp_outcome_unknown)
                log "editorial SMTP outcome unknown for run ${RUN_ID:-unknown}; manual verification required; no resend."
                exit 0
                ;;
            *)
                log "partial not resumable (${ACTION:-unknown}); no action."
                exit 0
                ;;
        esac
        ;;
    failed)
        log "marker: $DATE.failed present (failure notice already sent); exit."
        exit 0
        ;;
    none)
        if [ "$MODE" = "recovery" ]; then
            log "marker: none for business_date=$DATE; recovery-only mode, no heavy run."
            exit 0
        fi
        log "marker: none"
        ;;
    *)
        log "marker-status returned unexpected value '$MARKER'; treated as none."
        ;;
esac

if [ "$MODE" = "recovery" ]; then
    log "recovery mode: no heavy run for marker '$MARKER' (business_date=$DATE)."
    exit 0
fi

# 2) light SearXNG check
SEARXNG_OUT=$("$PYTHON_BIN" "$HELPER" searxng-check "$SEARXNG_URL" --timeout "$SEARXNG_TIMEOUT" 2>&1)
SEARXNG_RC=$?
log "searxng: rc=$SEARXNG_RC $SEARXNG_OUT"

if [ "$SEARXNG_RC" -ne 0 ]; then
    log "searxng: no productive engine (0 result or unreachable); heavy job NOT launched."
    if [ "$SLOT" = "$LAST_SLOT" ]; then
        log "last slot ($LAST_SLOT): sending light failure mail."
        MAIL_JSON="$WRAPPER_OUTPUT/mails/failure_${DATE}.json"
        if "$PYTHON_BIN" "$HELPER" failure-mail "$MAIL_JSON" --subject "$FAILURE_SUBJECT" --body "$FAILURE_BODY" >>"$LOG_FILE" 2>&1; then
            "$PYTHON_BIN" "$SCRIPT_DIR/mail_sender.py" "$MAIL_JSON" --env-file "$MAIL_ENV" >>"$LOG_FILE" 2>&1
            MAIL_RC=$?
            log "failure mail send rc=$MAIL_RC"
            if [ "$MAIL_RC" -eq 0 ]; then
                "$PYTHON_BIN" "$HELPER" write-marker "$MARKER_DIR" "$DATE" failed \
                    --slot "$SLOT" --run-id "" --run-status no_results \
                    --source-count 0 --cited-source-count 0 >>"$LOG_FILE" 2>&1
                log "failure marker written: $MARKER_DIR/$DATE.failed"
            else
                log "failure mail NOT confirmed; failure marker not written."
            fi
        else
            log "failure-mail build failed; no mail sent."
        fi
    else
        log "not the last slot; waiting for the next cron slot."
    fi
    exit 0
fi

# 3) launch the real job exactly once
log "searxng ok; launching heavy job once."
JOB_LOG="$LOG_DIR/job_${TODAY}_${SLOT}.log"
(
    cd "$SCRIPT_DIR" || exit 1
    sh "$JOB_WRAPPER"
) >"$JOB_LOG" 2>&1
JOB_RC=$?

RUN_DIR=$(sed -n 's/^run_dir: //p' "$JOB_LOG" | head -n 1)
if [ -z "$RUN_DIR" ]; then
    RUN_DIR=$(ls -dt "$JOB_OUTPUT_ROOT"/*/ 2>/dev/null | head -n 1)
    RUN_DIR=${RUN_DIR%/}
fi
log "job rc=$JOB_RC run_dir=${RUN_DIR:-unknown}"

if [ -z "$RUN_DIR" ] || [ ! -f "$RUN_DIR/run.json" ]; then
    log "run.json missing; no marker written (a later slot may retry)."
    exit 0
fi

# 4) summary + marker decision
SUMMARY=$("$PYTHON_BIN" "$HELPER" run-summary "$RUN_DIR" 2>&1)
SUMMARY_RC=$?
log "summary rc=$SUMMARY_RC: $SUMMARY"
if [ "$SUMMARY_RC" -ne 0 ]; then
    log "run-summary failed; no marker written (a later slot may retry)."
    exit 0
fi

STATUS=$(printf '%s\n' "$SUMMARY" | sed -n 's/^status=//p' | head -n 1)
RAW_SENT=$(printf '%s\n' "$SUMMARY" | sed -n 's/^raw_mail_sent=//p' | head -n 1)
EDIT_SENT=$(printf '%s\n' "$SUMMARY" | sed -n 's/^editorial_mail_sent=//p' | head -n 1)
MAIL_SENT=$(printf '%s\n' "$SUMMARY" | sed -n 's/^mail_sent=//p' | head -n 1)
SOURCES=$(printf '%s\n' "$SUMMARY" | sed -n 's/^source_count=//p' | head -n 1)
CITED=$(printf '%s\n' "$SUMMARY" | sed -n 's/^cited_source_count=//p' | head -n 1)
RUN_ID=$(printf '%s\n' "$SUMMARY" | sed -n 's/^run_id=//p' | head -n 1)
[ -z "$RUN_ID" ] && RUN_ID=$(basename "$RUN_DIR")

if [ "$STATUS" = "completed" ] && [ "$MAIL_SENT" = "True" ] && [ "$RAW_SENT" = "True" ] && [ "$EDIT_SENT" = "True" ]; then
    log "success: status=$STATUS mails sent (raw=$RAW_SENT editorial=$EDIT_SENT)."
    "$PYTHON_BIN" "$HELPER" write-marker "$MARKER_DIR" "$DATE" ok \
        --slot "$SLOT" --run-id "$RUN_ID" --run-status "$STATUS" \
        --raw-mail-sent true --editorial-mail-sent true --mail-sent true \
        --source-count "$SOURCES" --cited-source-count "$CITED" >>"$LOG_FILE" 2>&1
    log "success marker written: $MARKER_DIR/$DATE.ok"
    exit 0
fi

if [ "$RAW_SENT" = "True" ] || [ "$EDIT_SENT" = "True" ]; then
    log "partial failure: status=$STATUS raw=$RAW_SENT editorial=$EDIT_SENT; blocking marker written."
    "$PYTHON_BIN" "$HELPER" write-marker "$MARKER_DIR" "$DATE" partial \
        --slot "$SLOT" --run-id "$RUN_ID" --run-status "$STATUS" \
        --raw-mail-sent "$RAW_SENT" --editorial-mail-sent "$EDIT_SENT" \
        --mail-sent "$MAIL_SENT" --source-count "$SOURCES" --cited-source-count "$CITED" >>"$LOG_FILE" 2>&1
    log "partial marker written: $MARKER_DIR/$DATE.partial"
    exit 0
fi

log "hard failure: status=$STATUS no mail sent; no marker (a later slot may retry)."
exit 0
