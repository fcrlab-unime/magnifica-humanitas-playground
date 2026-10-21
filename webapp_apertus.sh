#!/usr/bin/env bash
LLM_ROOT="${LLM_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
# ────────────────────────────────────────────────────────────────────────────
#  Apertus Playground — script di gestione dell'app (web_app_apertus.py,
#  porta 5002; il modello si sceglie dalla pagina)
#  Usage:
#    ./webapp_apertus.sh start    avvia il server Flask in background
#    ./webapp_apertus.sh stop     ferma il server (kill del PID salvato)
#    ./webapp_apertus.sh restart  stop + start
#    ./webapp_apertus.sh status   verifica se il server è attivo
#    ./webapp_apertus.sh logs     tail -F del log
# ────────────────────────────────────────────────────────────────────────────
set -u

# Path assoluto di questo script, così `restart` può re-invocarsi a prescindere
# da come è stato lanciato (./webapp_apertus.sh, bash webapp_apertus.sh, path assoluto…).
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

BASE=${LLM_ROOT}
APP="$BASE/web_app_apertus.py"
VENV="$BASE/venv/bin/activate"
LOG="$BASE/webapp_apertus.log"
PIDFILE="$BASE/webapp_apertus.pid"
HOST="0.0.0.0"
PORT=5002

cmd="${1:-status}"

is_running() {
    if [[ -f "$PIDFILE" ]]; then
        local pid; pid=$(cat "$PIDFILE")
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        fi
    fi
    # Fallback: cerca un processo che esegua web_app_apertus.py.
    # ATTENZIONE: `pgrep -f web_app_apertus.py` matcha anche shell la cui
    # command-line contiene la stringa "web_app_apertus.py" → falsi positivi
    # pericolosi. Filtriamo quindi ai soli processi il cui eseguibile è
    # davvero python3.
    local p
    for p in $(pgrep -f "web_app_apertus.py" 2>/dev/null); do
        case "$(ps -o comm= -p "$p" 2>/dev/null)" in
            python3|python) echo "$p"; return 0 ;;
        esac
    done
    return 1
}

case "$cmd" in
    start)
        if pid=$(is_running); then
            echo "⚠ already running (PID $pid)"
            exit 0
        fi
        echo "▶ Starting Apertus Playground …"
        cd "$BASE"
        # shellcheck disable=SC1090
        source "$VENV"
        # Librerie CUDA di llama.cpp: quelle che torch installa nella venv
        # (nvidia/*/lib) bastano; LLM_CUDA_LIB, se impostata, ha la precedenza.
        NV=""
        if compgen -G "$BASE/venv/lib/python*/site-packages/nvidia/*/lib" >/dev/null; then
            NV="$(echo "$BASE"/venv/lib/python*/site-packages/nvidia/*/lib | tr ' ' ':')"
        fi
        export LD_LIBRARY_PATH="${LLM_CUDA_LIB:+$LLM_CUDA_LIB:}${NV}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export PYTHONUNBUFFERED=1
        nohup python3 "$APP" >"$LOG" 2>&1 &
        PID=$!
        echo "$PID" > "$PIDFILE"
        sleep 2
        if kill -0 "$PID" 2>/dev/null; then
            echo "✓ started (PID $PID)"
            echo "  Local URL:   http://localhost:$PORT"
            # Elenco TUTTE le interfacce IPv4 non-loopback
            ip -4 -o addr show 2>/dev/null \
              | awk -v port="$PORT" '$2!="lo" {split($4,a,"/"); printf("  URL %-8s http://%s:%s\n", $2":", a[1], port)}'
            echo "  Log:        $LOG"
        else
            echo "✗ start failed — check $LOG"
            tail -20 "$LOG" 2>/dev/null
            rm -f "$PIDFILE"
            exit 1
        fi
        ;;

    stop)
        if ! pid=$(is_running); then
            echo "▪ not running"
            rm -f "$PIDFILE"
            exit 0
        fi
        echo "■ Stopping the server (PID $pid) …"
        kill "$pid" 2>/dev/null
        for i in {1..10}; do
            sleep 0.5
            kill -0 "$pid" 2>/dev/null || break
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "  forcing kill -9 …"
            kill -9 "$pid" 2>/dev/null
            sleep 0.5
        fi
        rm -f "$PIDFILE"
        echo "✓ stopped"
        ;;

    restart)
        bash "$SELF" stop
        sleep 1
        bash "$SELF" start
        ;;

    status)
        if pid=$(is_running); then
            uptime=$(ps -o etime= -p "$pid" 2>/dev/null | xargs)
            echo "● RUNNING  (PID $pid · uptime $uptime)"
            echo "  Local URL:   http://localhost:$PORT"
            # Elenco TUTTE le interfacce IPv4 non-loopback
            ip -4 -o addr show 2>/dev/null \
              | awk -v port="$PORT" '$2!="lo" {split($4,a,"/"); printf("  URL %-8s http://%s:%s\n", $2":", a[1], port)}'
            echo "  Log:        $LOG"
        else
            echo "○ STOPPED"
            rm -f "$PIDFILE"
            exit 1
        fi
        ;;

    logs)
        if [[ ! -f "$LOG" ]]; then
            echo "▪ no log yet"
            exit 0
        fi
        echo "▸ tail -F $LOG (Ctrl-C to exit)"
        tail -F "$LOG"
        ;;

    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
