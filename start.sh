#!/bin/bash
# Starts both the FastAPI backend and the Gradio frontend in a single
# container, so `docker run <image>` alone brings up the full application.
set -e

echo "Starting FastAPI on :8000 ..."
uvicorn main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Small buffer so the API is listening before Gradio's first request,
# though gradio_app.py itself only calls the API on user interaction.
sleep 2

echo "Starting Gradio on :7860 ..."
python gradio_app.py &
GRADIO_PID=$!

# Forward container stop signals to both child processes for a clean shutdown.
trap 'echo "Shutting down..."; kill -TERM "$API_PID" "$GRADIO_PID" 2>/dev/null; wait' SIGTERM SIGINT

# If either process dies unexpectedly, bring the whole container down too --
# this makes crashes visible in `docker ps` / restart policies instead of
# silently leaving a half-working container running.
wait -n "$API_PID" "$GRADIO_PID"
EXIT_CODE=$?
kill -TERM "$API_PID" "$GRADIO_PID" 2>/dev/null
exit $EXIT_CODE