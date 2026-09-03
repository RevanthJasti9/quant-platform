#!/bin/bash
# Reclaims RAM on this 8GB Mac by:
#   1. Telling Ollama to drop the local LLM from memory right now (~2.5GB),
#      instead of waiting out its ~5 min idle timeout.
# The dashboard now handles normal idle cleanup itself. This remains a
# manual, model-only escape hatch and intentionally does not kill a pipeline:
# stopping an ingest or database write midway is not safe.
#
# Safe to run anytime: it never touches the live dashboard server (uvicorn)
# or anything outside this project. Ollama reloads the model automatically
# the next time it's needed, and any pipeline step you stop here can just
# be re-run -- nothing here deletes data.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "Freeing RAM..."

MODEL=$("$PROJECT_DIR/.venv/bin/python" -c "
import yaml
print(yaml.safe_load(open('$PROJECT_DIR/config/settings.yaml'))['llm']['model'])
" 2>/dev/null)
MODEL="${MODEL:-llama3.2:3b}"

if curl -s -o /dev/null -X POST http://localhost:11434/api/generate \
    -d "{\"model\": \"${MODEL}\", \"keep_alive\": 0}"; then
    echo "- Unloaded Ollama model ($MODEL)"
else
    echo "- Ollama not reachable (already stopped, or nothing to unload)"
fi

echo "Done."
