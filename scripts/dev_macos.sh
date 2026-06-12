#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This command is intended for macOS."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FRONTEND_DIR="${ROOT_DIR}/frontend"
VITE_PORT="5173"
VITE_URL="http://localhost:${VITE_PORT}"
VITE_CLI_RELATIVE="frontend/node_modules/vite/dist/node/cli.js"
VITE_CLI="${ROOT_DIR}/${VITE_CLI_RELATIVE}"
VITE_PID=""
APP_PID=""

log() {
  printf '[lity-dev] %s\n' "$*"
}

fail() {
  printf '[lity-dev] ERROR: %s\n' "$*" >&2
  exit 1
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  terminate_process_tree "${APP_PID}" "Lity"
  terminate_process_tree "${VITE_PID}" "Vite"
  if [[ -n "${VITE_PID}" ]]; then
    free_vite_port >/dev/null 2>&1 || true
  fi
  exit "${status}"
}

trap cleanup EXIT INT TERM

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  command -v curl >/dev/null 2>&1 || fail "uv is missing and curl is unavailable."

  log "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
  command -v uv >/dev/null 2>&1 || fail "uv installation finished, but uv is not on PATH."
}

ensure_npm() {
  if command -v npm >/dev/null 2>&1; then
    return
  fi

  if command -v brew >/dev/null 2>&1; then
    log "Installing Node.js with Homebrew..."
    brew install node
    command -v npm >/dev/null 2>&1 && return
  fi

  fail "npm is missing. Install Node.js LTS, then run this command again."
}

frontend_needs_install() {
  [[ ! -d "${FRONTEND_DIR}/node_modules" ]] && return 0
  [[ ! -f "${VITE_CLI}" ]] && return 0
  [[ ! -x "${FRONTEND_DIR}/node_modules/.bin/vite" ]] && return 0
  [[ "${FRONTEND_DIR}/package-lock.json" -nt "${FRONTEND_DIR}/node_modules/.package-lock.json" ]] && return 0
  return 1
}

install_frontend_deps() {
  if frontend_needs_install; then
    log "Installing frontend dependencies..."
    rm -rf "${FRONTEND_DIR}/node_modules"
    (cd "${FRONTEND_DIR}" && npm ci)
  else
    log "Frontend dependencies already installed."
  fi

  [[ -f "${VITE_CLI}" ]] || fail "Vite is still incomplete after npm ci: ${VITE_CLI_RELATIVE}"
}

sync_python_deps() {
  log "Syncing Python dependencies..."
  cd "${ROOT_DIR}"
  uv sync --extra desktop --extra web --extra dev --extra packaging
}

process_tree_pids() {
  local root_pid="$1"
  local children child
  children="$(pgrep -P "${root_pid}" 2>/dev/null || true)"
  while IFS= read -r child; do
    [[ -n "${child}" ]] || continue
    process_tree_pids "${child}"
    printf '%s\n' "${child}"
  done <<< "${children}"
}

terminate_process_tree() {
  local root_pid="$1"
  local label="$2"
  local pids pid alive

  [[ -n "${root_pid}" ]] || return 0
  kill -0 "${root_pid}" >/dev/null 2>&1 || return 0

  log "Stopping ${label}..."
  pids="$(process_tree_pids "${root_pid}"; printf '%s\n' "${root_pid}")"
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    kill -TERM "${pid}" >/dev/null 2>&1 || true
  done <<< "${pids}"

  for _ in {1..20}; do
    alive=""
    while IFS= read -r pid; do
      [[ -n "${pid}" ]] || continue
      if kill -0 "${pid}" >/dev/null 2>&1; then
        alive=1
        break
      fi
    done <<< "${pids}"
    [[ -z "${alive}" ]] && return 0
    sleep 0.25
  done

  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    kill -KILL "${pid}" >/dev/null 2>&1 || true
  done <<< "${pids}"
}

vite_port_pids() {
  lsof -nP -iTCP:"${VITE_PORT}" -sTCP:LISTEN -t 2>/dev/null | sort -u || true
}

wait_for_vite_port_to_close() {
  for _ in {1..40}; do
    if [[ -z "$(vite_port_pids)" ]]; then
      return 0
    fi
    sleep 0.25
  done
  return 1
}

free_vite_port() {
  local pids pid command
  pids="$(vite_port_pids)"
  if [[ -z "${pids}" ]]; then
    log "Port ${VITE_PORT} is free."
    return 0
  fi

  log "Port ${VITE_PORT} is already in use; closing listener(s)."
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    command="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
    log "Stopping PID ${pid}${command:+: ${command}}"
    kill -TERM "${pid}" >/dev/null 2>&1 || true
  done <<< "${pids}"

  if wait_for_vite_port_to_close; then
    return 0
  fi

  log "Port ${VITE_PORT} is still busy; force-stopping listener(s)."
  pids="$(vite_port_pids)"
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    kill -KILL "${pid}" >/dev/null 2>&1 || true
  done <<< "${pids}"

  wait_for_vite_port_to_close
}

start_vite() {
  log "Starting Vite on ${VITE_URL}..."
  (
    trap '' INT
    cd "${FRONTEND_DIR}"
    exec npm run dev -- --host 127.0.0.1 --port "${VITE_PORT}"
  ) &
  VITE_PID=$!
}

wait_for_vite() {
  log "Waiting for Vite..."
  for _ in {1..80}; do
    if ! kill -0 "${VITE_PID}" >/dev/null 2>&1; then
      fail "Vite stopped before it became ready."
    fi
    if curl -fsS "${VITE_URL}" >/dev/null 2>&1; then
      log "Vite is ready."
      return
    fi
    sleep 0.25
  done
  fail "Vite did not answer on ${VITE_URL}."
}

run_desktop_app() {
  log "Starting Lity in web dev mode..."
  (
    trap '' INT
    cd "${ROOT_DIR}"
    exec uv run lity --ui web --dev "$@"
  ) &
  APP_PID=$!
}

wait_for_desktop_app() {
  [[ -n "${APP_PID}" ]] || fail "Lity did not start."
  wait "${APP_PID}"
}

main() {
  ensure_uv
  ensure_npm
  sync_python_deps
  install_frontend_deps
  free_vite_port
  start_vite
  wait_for_vite
  run_desktop_app "$@"
  wait_for_desktop_app
}

main "$@"
