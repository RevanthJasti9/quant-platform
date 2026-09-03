#!/bin/bash
# Double-click this to open the AI Quant dashboard. Starts the local server
# first if it isn't already running.
cd "/Users/revanthjasti/quant-platform" || exit 1

if ! curl -s -o /dev/null "http://127.0.0.1:8000/dashboard"; then
    echo "Starting AI Quant server..."
    source .venv/bin/activate
    nohup uvicorn app.main:app --host 127.0.0.1 --port 8000 > /tmp/quant-platform.log 2>&1 &
    disown
    for i in 1 2 3 4 5 6 7 8 9 10; do
        sleep 1
        if curl -s -o /dev/null "http://127.0.0.1:8000/dashboard"; then
            break
        fi
    done
fi

open "http://127.0.0.1:8000/dashboard"
sleep 1
