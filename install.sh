#!/usr/bin/env bash
# itsbob installer — one command from a clone to a working assistant.
#
#   ./install.sh
#
# Creates a virtualenv, installs itsbob and its optional extras, and runs the
# setup wizard. Safe to re-run: it upgrades in place and never touches your keys.
set -euo pipefail

BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; RED=$'\033[31m'; OFF=$'\033[0m'
say()  { printf '%s\n' "$*"; }
ok()   { printf '  %s✓%s %s\n' "$GREEN" "$OFF" "$*"; }
die()  { printf '  %s✗ %s%s\n' "$RED" "$*" "$OFF" >&2; exit 1; }

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

say ""
say "  ${BOLD}itsbob${OFF}"
say "  ${DIM}a personal assistant that runs on your own laptop${OFF}"
say ""

# --- Python ---------------------------------------------------------------
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)'; then
      PYTHON="$candidate"; break
    fi
  fi
done
[ -n "$PYTHON" ] || die "Python 3.10+ is required. Install it from https://python.org/downloads"
ok "using $($PYTHON --version)"

# --- virtualenv -----------------------------------------------------------
VENV="${ITSBOB_VENV:-$HERE/.venv}"
if [ ! -d "$VENV" ]; then
  "$PYTHON" -m venv "$VENV" || die "could not create a virtualenv at $VENV
    On Debian/Ubuntu you may need:  sudo apt install python3-venv"
  ok "created virtualenv at ${VENV/#$HOME/\~}"
else
  ok "reusing virtualenv at ${VENV/#$HOME/\~}"
fi

PIP="$VENV/bin/pip"
[ -x "$PIP" ] || PIP="$VENV/Scripts/pip.exe"       # Git Bash on Windows
[ -x "$PIP" ] || die "virtualenv looks broken — delete $VENV and re-run"

# --- install --------------------------------------------------------------
say "  ${DIM}installing (this takes a minute)…${OFF}"
"$PIP" install --quiet --upgrade pip >/dev/null 2>&1 || true
if ! "$PIP" install --quiet --upgrade -e ".[all]"; then
  say "  ${DIM}full install failed; falling back to the core package…${OFF}"
  "$PIP" install --quiet --upgrade -e "." || die "installation failed — re-run with:
    $PIP install -e '.[all]'"
  ok "installed itsbob (core only — no GUI; add it later with: $PIP install -e '.[gui]')"
else
  ok "installed itsbob with the browser interface and the fast recall path"
fi

BIN="$VENV/bin/itsbob"; [ -x "$BIN" ] || BIN="$VENV/Scripts/itsbob.exe"

# --- put it on PATH -------------------------------------------------------
LINKED=""
for dir in "$HOME/.local/bin" "$HOME/bin"; do
  if [ -d "$dir" ] && case ":$PATH:" in *":$dir:"*) true;; *) false;; esac; then
    ln -sf "$BIN" "$dir/itsbob" && LINKED="$dir/itsbob" && break
  fi
done
if [ -n "$LINKED" ]; then
  ok "linked ${LINKED/#$HOME/\~} — run it as: itsbob"
else
  say "  ${DIM}·${OFF} not on PATH. Either run it as ${BOLD}$BIN${OFF},"
  say "    or add this to your shell profile:"
  say "      ${DIM}export PATH=\"$VENV/bin:\$PATH\"${OFF}"
fi

# --- setup ----------------------------------------------------------------
exec "$BIN" setup
