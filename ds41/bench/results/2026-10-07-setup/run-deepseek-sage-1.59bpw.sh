#!/bin/sh
cd "/workspace/strata-release"
exec "/workspace/strata-release/.venv/bin/python" "/workspace/strata-release/serve/server.py" "--engine" "strata" "--config" "/workspace/strata-release/strata-deepseek-sage-1.59bpw.json" "--port" "8080"
