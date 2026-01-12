check_conda_hook() {
  section "4) Conda hook check (eval \"\$(conda shell.bash hook)\" + base availability)"

  local out rc
  out="$(
    bash -l 2>&1 <<'BASH'
set -u

echo "[INFO] Shell: $0"
echo "[INFO] Starting PATH: $PATH"

ok()   { echo "✅ $*"; }
warn() { echo "⚠️  $*"; }
fail() { echo "❌ $*"; exit 1; }

# 4a) Ensure "conda" executable exists
if ! command -v conda >/dev/null 2>&1; then
  fail "conda executable not found in PATH.
Fix: add Miniconda/Anaconda bin to PATH or source conda.sh, e.g.
  source <miniconda>/etc/profile.d/conda.sh
or ensure your ~/.bashrc conda init block runs in login shells."
fi
ok "Found conda executable: $(command -v conda)"

# 4b) Try to deactivate (may be a no-op)
echo "[INFO] Attempting conda deactivate (may be a no-op)..."
conda deactivate >/dev/null 2>&1 || true

echo "[INFO] Before hook: $(type -t conda 2>/dev/null || echo "<missing>")"

# 4c) Run the hook
echo "[INFO] Running: eval \"\$(conda shell.bash hook)\""
HOOK_OUT="$(conda shell.bash hook 2>&1)" || {
  echo "$HOOK_OUT"
  fail "conda shell.bash hook failed."
}
eval "$HOOK_OUT" || fail "eval of conda hook output failed."

echo "[INFO] After hook: $(type -t conda 2>/dev/null || echo "<missing>")"

# 4d) Verify base directory
BASE_DIR="$(conda info --base 2>/dev/null || true)"
if [[ -z "$BASE_DIR" ]]; then
  fail "conda appears present, but 'conda info --base' returned nothing.
This suggests partial/broken initialization."
fi
ok "conda base directory: $BASE_DIR"

# 4e) Activate base
echo "[INFO] Attempting: conda activate base"
ACT_OUT="$(conda activate base 2>&1)" || {
  echo "$ACT_OUT"
  fail "conda activate base failed."
}

if [[ "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
  ok "Successfully activated base environment (CONDA_DEFAULT_ENV=base)."
else
  warn "conda activate base did not set CONDA_DEFAULT_ENV=base (got: ${CONDA_DEFAULT_ENV:-<unset>})."
fi

if [[ "$(type -t conda)" == "function" ]]; then
  ok "conda is initialized as a shell function (good)."
else
  warn "conda is not a shell function after hook (type -t conda = $(type -t conda))."
fi

echo "[INFO] Conda hook test complete."
BASH
  )"
  rc=$?

  printf '%s\n' "$out"

  if [[ $rc -eq 0 ]]; then
    ok "Conda hook + base activation check passed."
    return 0
  else
    fail "Conda hook + base activation check FAILED (see output above)."
    return 1
  fi
}
