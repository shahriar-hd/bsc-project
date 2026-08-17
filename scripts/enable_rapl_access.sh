#!/usr/bin/env bash
#
# enable_rapl_access.sh — grant read access to Intel RAPL energy counters.
#
# The CPU power figures in PowerMonitor come from the powercap sysfs interface:
#   /sys/class/powercap/intel-rapl:<pkg>/energy_uj
#
# Since Linux 5.10 those files are mode 0400 root-only (CVE-2020-8694: the
# counters leak enough timing signal to mount a side-channel attack on AES
# keys). So an unprivileged training run reads 0.0 W for CPU until the mode
# is relaxed.
#
# Usage:
#   sudo bash scripts/enable_rapl_access.sh            # until next reboot
#   sudo bash scripts/enable_rapl_access.sh --persist  # survives reboot (udev)
#   sudo bash scripts/enable_rapl_access.sh --revert   # restore 0400
#   bash scripts/enable_rapl_access.sh --check         # read-only, no sudo
#
# Security note: this makes the energy counters world-readable, which is the
# side-channel the CVE describes. Fine on a single-user workstation; do not do
# this on a shared or multi-tenant machine.

set -euo pipefail

RAPL_ROOT="/sys/class/powercap"
MODE="apply"

for arg in "$@"; do
  case "$arg" in
    --persist) MODE="persist" ;;
    --revert)  MODE="revert"  ;;
    --check)   MODE="check"   ;;
    -h|--help)
      sed -n '3,22p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

# ── does this machine even expose RAPL? ──────────────────────────────────────
if [[ ! -d "$RAPL_ROOT/intel-rapl" ]]; then
  echo "[!] No intel-rapl domain under $RAPL_ROOT"
  echo "    Possible causes:"
  echo "      - AMD CPU        → use amd_energy / k10temp instead"
  echo "      - VM / container → RAPL is usually not passed through"
  echo "      - module unloaded → try: sudo modprobe intel_rapl_common"
  echo "    PowerMonitor will report cpu_w = 0.0 and estimate the rest."
  exit 1
fi

# Glob the sysfs tree rather than using `find`: the domain directories are
# root-only, so an unprivileged `find` cannot descend into them and would
# report zero counters even when they exist.
ENERGY_FILES=()
for dom in "$RAPL_ROOT"/intel-rapl/intel-rapl:*; do
  [[ -e "$dom/energy_uj" ]] && ENERGY_FILES+=("$dom/energy_uj")
  for sub in "$dom"/intel-rapl:*; do
    [[ -e "$sub/energy_uj" ]] && ENERGY_FILES+=("$sub/energy_uj")
  done
done

if [[ ${#ENERGY_FILES[@]} -eq 0 ]]; then
  echo "[!] intel-rapl present but no energy_uj files found." >&2
  exit 1
fi

# ── report current state ─────────────────────────────────────────────────────
show_state() {
  echo "  RAPL domains found: ${#ENERGY_FILES[@]}"
  for f in "${ENERGY_FILES[@]}"; do
    local dom name perms
    dom="$(dirname "$f")"
    name="$(cat "$dom/name" 2>/dev/null || echo '?')"
    perms="$(stat -c '%a %U:%G' "$f")"
    if [[ -r "$f" ]]; then
      printf '    %-28s %-12s %-14s readable ✓\n' "$(basename "$dom")" "$name" "$perms"
    else
      printf '    %-28s %-12s %-14s NOT readable ✗\n' "$(basename "$dom")" "$name" "$perms"
    fi
  done
}

if [[ "$MODE" == "check" ]]; then
  echo "── RAPL access check ──────────────────────────────────────────"
  show_state
  if [[ -r "${ENERGY_FILES[0]}" ]]; then
    echo "  → CPU power monitoring should work."
  else
    echo "  → CPU power unavailable. Run: sudo bash $0"
  fi
  exit 0
fi

# ── everything past here needs root ──────────────────────────────────────────
if [[ "$EUID" -ne 0 ]]; then
  echo "[!] This mode needs root. Re-run with: sudo bash $0 ${*:-}" >&2
  exit 1
fi

UDEV_RULE="/etc/udev/rules.d/99-intel-rapl-readable.rules"

case "$MODE" in
  apply|persist)
    echo "── Granting read access to RAPL counters ──────────────────────"
    chmod -R a+r "$RAPL_ROOT/intel-rapl" 2>/dev/null || true
    # chmod -R on sysfs can skip files behind symlinks; set each one directly.
    for f in "${ENERGY_FILES[@]}"; do
      chmod a+r "$f" 2>/dev/null || echo "    [warn] could not chmod $f"
    done
    show_state

    if [[ "$MODE" == "persist" ]]; then
      cat > "$UDEV_RULE" <<'EOF'
# Make Intel RAPL energy counters world-readable so unprivileged processes
# can measure CPU power (see scripts/enable_rapl_access.sh).
# Trade-off: re-opens the CVE-2020-8694 side channel. Single-user hosts only.
SUBSYSTEM=="powercap", KERNEL=="intel-rapl*", RUN+="/bin/chmod -R a+r /sys%p"
EOF
      udevadm control --reload-rules 2>/dev/null || true
      udevadm trigger --subsystem-match=powercap 2>/dev/null || true
      echo "  Installed udev rule → $UDEV_RULE (survives reboot)"
    else
      echo "  Note: resets on reboot. Use --persist to make it permanent."
    fi
    ;;

  revert)
    echo "── Restoring root-only access (0400) ──────────────────────────"
    for f in "${ENERGY_FILES[@]}"; do
      chmod 400 "$f" 2>/dev/null || true
    done
    [[ -f "$UDEV_RULE" ]] && rm -f "$UDEV_RULE" && \
      udevadm control --reload-rules 2>/dev/null || true
    show_state
    echo "  Reverted. CPU power will read 0.0 W again."
    ;;
esac

echo
echo "  Verify from Python:"
echo "    python -c \"from src.config import get_config; \\"
echo "from src.utils.power_utils import power_monitor_from_config as f; \\"
echo "m=f(get_config()); print('cpu_w =', m.read_cpu_power())\""
