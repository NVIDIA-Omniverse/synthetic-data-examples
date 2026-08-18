#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

ENV_FILE=".env"
TOKEN_NAMES=(
    DATA_GENERATION_JUPYTER_TOKEN
    TRAINING_JUPYTER_TOKEN
    DEPLOYMENT_JUPYTER_TOKEN
)

if ! command -v openssl >/dev/null 2>&1; then
    echo "openssl is required to generate Jupyter tokens." >&2
    exit 1
fi

umask 077
touch "$ENV_FILE"
chmod 600 "$ENV_FILE"

for token_name in "${TOKEN_NAMES[@]}"; do
    if ! grep -q "^${token_name}=" "$ENV_FILE"; then
        printf "%s=%s\n" "$token_name" "$(openssl rand -hex 32)" >>"$ENV_FILE"
    fi
done

get_token() {
    grep "^$1=" "$ENV_FILE" | cut -d= -f2-
}

echo "Access the labs:"
echo "Part 1: http://127.0.0.1:8882/lab?token=$(get_token DATA_GENERATION_JUPYTER_TOKEN)"
echo "Part 2: http://127.0.0.1:8883/lab?token=$(get_token TRAINING_JUPYTER_TOKEN)"
echo "Part 3: http://127.0.0.1:8884/lab?token=$(get_token DEPLOYMENT_JUPYTER_TOKEN)"

HOST_UID="$(id -u)" HOST_GID="$(id -g)" docker-compose up
