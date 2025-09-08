#!/usr/bin/env bash

set -e
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

fail_under=${FAIL_UNDER:-"100"}
uv run pytest \
    --cov-config=.coveragerc \
    --cov=rigid/src/scholar_agent \
    --cov-report=term \
    --cov-report=html \
    --cov-fail-under=$fail_under \
    $warn_arg "$@"
