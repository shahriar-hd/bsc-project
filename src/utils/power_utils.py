"""
Power & energy monitoring shared by train.py and preprocessing.py.

Measures CPU (Intel RAPL sysfs), GPU (pynvml, falling back to nvidia-smi),
and estimates RAM/SSD/board draw, then integrates power over time into kWh
and a CO2 estimate.

Design notes
------------
* CPU energy is read as a *cumulative counter delta* over the whole poll
  interval rather than a 0.1 s spot sample, which removes the extra sleep
  from the polling loop and averages over the real interval.
* RAPL counters wrap at `max_energy_range_uj`; the wrap is handled explicitly.
* Every backend degrades to 0.0 W instead of raising, so monitoring never
  takes down a training run. `psutil` and `pynvml` are optional.

Usage
-----
    monitor = power_monitor_from_config(cfg)
    monitor.start()
    ...
    avg_w = monitor.stop()          # returns mean total power (W)
    monitor.log_summary(logger)

Or scoped:

    with power_monitor_from_config(cfg) as monitor:
        ...
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Optional dependencies — absent in a minimal install.
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import pynvml
    _HAS_PYNVML = True
except ImportError:
    _HAS_PYNVML = False


@dataclass
class GPUSample:
    """Single power sample for one GPU device."""
    gpu_id: int
    power_w: float
    timestamp: float


class PowerMonitor:
    """
    Background thread that polls CPU (RAPL), RAM, and one or more CUDA GPUs.

    Args:
        poll_interval: Seconds between samples.
        rapl_path: Sysfs path prefix for Intel RAPL.
        ram_coeff: W per GB of used system RAM (estimate).
        ssd_coeff: W for SSD activity (fixed estimate).
        other_coeff: W for misc board components (fixed estimate).
        gpu_ids: CUDA device indices to monitor; empty/None = all visible.
        per_gpu_log: If True, keep a per-GPU breakdown alongside aggregates.
        co2_kg_per_kwh: Grid emission factor.
        overhead_multiplier: Scales measured power to account for PSU losses.
        csv_path: If set, append one row per sample for offline analysis.
    """

    def __init__(
        self,
        poll_interval: float = 5.0,
        rapl_path: str = "/sys/class/powercap/intel-rapl",
        ram_coeff: float = 0.375,
        ssd_coeff: float = 2.0,
        other_coeff: float = 5.0,
        gpu_ids: Optional[List[int]] = None,
        per_gpu_log: bool = True,
        co2_kg_per_kwh: float = 0.494,
        overhead_multiplier: float = 1.15,
        csv_path: Optional[str] = None,
    ) -> None:
        self.poll_interval = max(0.5, float(poll_interval))
        self.rapl_path = rapl_path
        self.ram_coeff = ram_coeff
        self.ssd_coeff = ssd_coeff
        self.other_coeff = other_coeff
        self.per_gpu_log = per_gpu_log
        self.co2_kg_per_kwh = co2_kg_per_kwh
        self.overhead_multiplier = overhead_multiplier
        self.csv_path = csv_path

        self._nvml_ready = self._init_nvml()
        self._gpu_ids: List[int] = self._resolve_gpu_ids(gpu_ids)
        self._rapl_files: List[Path] = self._discover_rapl_files()

        # Samples (W)
        self._samples: List[float] = []
        self._cpu_samples: List[float] = []
        self._gpu_samples: Dict[int, List[float]] = {g: [] for g in self._gpu_ids}

        # RAPL cumulative-counter state
        self._last_rapl_uj: Optional[int] = None
        self._last_rapl_t: Optional[float] = None

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t_start: Optional[float] = None
        self._elapsed_s: float = 0.0

        # Energy counters (kWh)
        self.total_energy_kwh: float = 0.0
        self.gpu_energy_kwh: Dict[int, float] = {g: 0.0 for g in self._gpu_ids}
        self.cpu_energy_kwh: float = 0.0

        self._csv_initialised = False

    # ── backwards-compatible alias ───────────────────────────────────────────

    @property
    def readings(self) -> List[float]:
        """Total-power samples collected so far (thread-safe copy)."""
        with self._lock:
            return list(self._samples)

    # ── backend discovery ────────────────────────────────────────────────────

    @staticmethod
    def _init_nvml() -> bool:
        """Initialise NVML once; return False if unavailable."""
        if not _HAS_PYNVML:
            return False
        try:
            pynvml.nvmlInit()
            return True
        except Exception:
            return False

    def _resolve_gpu_ids(self, requested: Optional[List[int]]) -> List[int]:
        """Return validated GPU indices, defaulting to all visible devices."""
        available: List[int] = []

        if self._nvml_ready:
            try:
                available = list(range(pynvml.nvmlDeviceGetCount()))
            except Exception:
                available = []

        if not available:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                ).stdout
                available = [
                    int(x.strip()) for x in out.strip().splitlines() if x.strip()
                ]
            except Exception:
                available = []

        if not available:
            return []
        if not requested:
            return available
        return [g for g in requested if g in available]

    def _discover_rapl_files(self) -> List[Path]:
        """
        Find RAPL package-domain energy counters.

        Only top-level `intel-rapl:N` package domains are used; sub-domains
        (core/uncore/dram) are nested inside them and would double-count.
        """
        root = Path(self.rapl_path)
        if not root.exists():
            return []

        files: List[Path] = []
        try:
            for entry in sorted(root.iterdir()):
                if not entry.name.startswith("intel-rapl:"):
                    continue
                if entry.name.count(":") != 1:      # skip intel-rapl:0:0 etc.
                    continue
                name_file = entry / "name"
                energy_file = entry / "energy_uj"
                if not (name_file.exists() and energy_file.exists()):
                    continue
                try:
                    if "package" not in name_file.read_text().strip():
                        continue
                    energy_file.read_text()          # permission probe
                except (OSError, PermissionError):
                    continue
                files.append(energy_file)
        except OSError:
            return []
        return files
    # ── sampling backends ────────────────────────────────────────────────────

    def _read_rapl_uj(self) -> Optional[int]:
        """Sum the raw RAPL package counters (microjoules)."""
        if not self._rapl_files:
            return None
        total = 0
        ok = False
        for f in self._rapl_files:
            try:
                total += int(f.read_text().strip())
                ok = True
            except (OSError, ValueError):
                continue
        return total if ok else None

    def _read_cpu_watts(self) -> float:
        """
        CPU package power from the RAPL counter delta since the last poll.

        Returns 0.0 on the first call (no baseline yet) or if RAPL is
        unreadable — run
        `sudo chmod -R a+r /sys/class/powercap/intel-rapl` to enable it.
        """
        now = time.time()
        current = self._read_rapl_uj()
        if current is None:
            return 0.0

        if self._last_rapl_uj is None or self._last_rapl_t is None:
            self._last_rapl_uj = current
            self._last_rapl_t = now
            return 0.0

        delta_uj = current - self._last_rapl_uj
        delta_t = now - self._last_rapl_t

        self._last_rapl_uj = current
        self._last_rapl_t = now

        if delta_uj < 0:
            # Counter wrapped; skip this interval rather than emit a spike.
            return 0.0

        # Guard against a degenerate window (two reads back to back). Over a
        # full poll interval delta_uj == 0 means a genuinely idle package, so
        # it is reported as-is rather than spot-measured.
        if delta_t < 0.05:
            return self._spot_cpu_watts()

        return max(0.0, (delta_uj / 1e6) / delta_t)   # J / s = W

    def _spot_cpu_watts(self, window: float = 0.1) -> float:
        """Blocking CPU power measurement over a short fixed window."""
        first = self._read_rapl_uj()
        if first is None:
            return 0.0
        time.sleep(window)
        second = self._read_rapl_uj()
        if second is None or second < first:
            return 0.0
        self._last_rapl_uj = second
        self._last_rapl_t = time.time()
        return max(0.0, ((second - first) / 1e6) / window)

    def _read_gpu_watts(self) -> Dict[int, float]:
        """Instantaneous power draw per monitored GPU, in watts."""
        result: Dict[int, float] = {g: 0.0 for g in self._gpu_ids}
        if not self._gpu_ids:
            return result

        if self._nvml_ready:
            for gid in self._gpu_ids:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gid)
                    result[gid] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except Exception:
                    result[gid] = 0.0
            if any(v > 0.0 for v in result.values()):
                return result

        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                try:
                    gid = int(parts[0])
                    if gid in result:
                        result[gid] = float(parts[1])
                except ValueError:
                    continue
        except Exception:
            pass
        return result

    def _estimate_system_overhead(self) -> float:
        """Estimate non-CPU, non-GPU power (RAM + SSD + board)."""
        if _HAS_PSUTIL:
            try:
                ram_gb = psutil.virtual_memory().used / (1024 ** 3)
            except Exception:
                ram_gb = 8.0
        else:
            ram_gb = 8.0
        return ram_gb * self.ram_coeff + self.ssd_coeff + self.other_coeff

    # ── polling loop ─────────────────────────────────────────────────────────

    def _poll(self) -> None:
        """
        Background polling loop — runs in a daemon thread.

        Waits *before* each sample so that every RAPL counter delta spans a
        full poll interval. Combined with the baseline primed in start(),
        this means no sample is measured over a degenerate window.
        """
        while not self._stop_event.is_set():
            self._stop_event.wait(self.poll_interval)
            if self._stop_event.is_set():
                break
            self._sample_once()

    def _sample_once(self) -> float:
        """Take one power sample, accumulate energy, and return total watts."""
        t_sample = time.time()
        cpu_w = self._read_cpu_watts()
        gpu_w = self._read_gpu_watts()
        overhead_w = self._estimate_system_overhead()

        total_gpu_w = sum(gpu_w.values())
        total_w = (cpu_w + total_gpu_w + overhead_w) * self.overhead_multiplier
        hours = self.poll_interval / 3600.0

        with self._lock:
            self._samples.append(total_w)
            self._cpu_samples.append(cpu_w)
            for gid, w in gpu_w.items():
                self._gpu_samples[gid].append(w)

            self.total_energy_kwh += total_w * hours / 1000.0
            self.cpu_energy_kwh += cpu_w * hours / 1000.0
            for gid, w in gpu_w.items():
                self.gpu_energy_kwh[gid] += w * hours / 1000.0

        if self.csv_path:
            self._append_csv(t_sample, cpu_w, total_gpu_w, overhead_w, total_w)
        return total_w

    def _append_csv(
        self,
        timestamp: float,
        cpu_w: float,
        gpu_w: float,
        overhead_w: float,
        total_w: float,
    ) -> None:
        """Append one sample row; never raises."""
        try:
            path = Path(self.csv_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not self._csv_initialised and not path.exists()
            with open(path, "a", encoding="utf-8") as f:
                if write_header:
                    f.write("timestamp,cpu_w,gpu_w,overhead_w,total_w,energy_kwh\n")
                f.write(
                    f"{timestamp:.3f},{cpu_w:.2f},{gpu_w:.2f},"
                    f"{overhead_w:.2f},{total_w:.2f},{self.total_energy_kwh:.8f}\n"
                )
            self._csv_initialised = True
        except OSError:
            pass

    # ── public API ───────────────────────────────────────────────────────────

    def start(self) -> "PowerMonitor":
        """Start the background monitoring thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._t_start = time.time()
        # Prime the RAPL counter baseline so the first sample reports real CPU
        # power instead of 0.0 W (which would bias the mean downward).
        self._read_cpu_watts()
        self._thread = threading.Thread(
            target=self._poll, daemon=True, name="PowerMonitor"
        )
        self._thread.start()
        return self

    def stop(self) -> float:
        """
        Stop monitoring and return mean total system power in watts.

        The return value is what train.py logs as `avg_power`.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_interval + 2)
            self._thread = None
        if self._t_start is not None:
            self._elapsed_s = time.time() - self._t_start
        return self.mean_power_w()

    def mean_power_w(self) -> float:
        with self._lock:
            samples = list(self._samples)
        return sum(samples) / len(samples) if samples else 0.0

    def read_cpu_power(self, settle: float = 1.0) -> float:
        """
        One-shot CPU power reading in watts, for diagnostics.

        Takes a baseline, waits `settle` seconds, then measures the counter
        delta. Returns 0.0 if RAPL is unreadable — see
        scripts/enable_rapl_access.sh.
        """
        self._read_cpu_watts()
        time.sleep(max(0.05, settle))
        return self._read_cpu_watts()

    def rapl_available(self) -> bool:
        """True if at least one RAPL package counter is readable."""
        return bool(self._rapl_files)

    def __enter__(self) -> "PowerMonitor":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def summary(self) -> Dict:
        """
        Aggregate statistics for logging and thesis tables.

        Keys: mean_power_w, mean_cpu_w, mean_gpu_w, total_energy_kwh,
        cpu_energy_kwh, gpu_energy_kwh, total_co2_kg, elapsed_s,
        num_samples, num_gpus_monitored, gpu_ids, per_gpu.
        """
        with self._lock:
            samples = list(self._samples)
            cpu_samples = list(self._cpu_samples)
            gpu_samples = {g: list(v) for g, v in self._gpu_samples.items()}
            total_energy = self.total_energy_kwh
            cpu_energy = self.cpu_energy_kwh
            gpu_energy = dict(self.gpu_energy_kwh)

        def mean(xs: List[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        elapsed = self._elapsed_s
        if not elapsed and self._t_start is not None:
            elapsed = time.time() - self._t_start

        per_gpu: Dict[int, Dict] = {}
        if self.per_gpu_log:
            for gid in self._gpu_ids:
                e = gpu_energy.get(gid, 0.0)
                per_gpu[gid] = {
                    "mean_power_w": mean(gpu_samples.get(gid, [])),
                    "energy_kwh": e,
                    "co2_kg": e * self.co2_kg_per_kwh,
                }

        return {
            "mean_power_w": mean(samples),
            "mean_cpu_w": mean(cpu_samples),
            "mean_gpu_w": sum(mean(v) for v in gpu_samples.values()),
            "total_energy_kwh": total_energy,
            "cpu_energy_kwh": cpu_energy,
            "gpu_energy_kwh": sum(gpu_energy.values()),
            "total_co2_kg": total_energy * self.co2_kg_per_kwh,
            "elapsed_s": elapsed,
            "num_samples": len(samples),
            "num_gpus_monitored": len(self._gpu_ids),
            "gpu_ids": list(self._gpu_ids),
            "per_gpu": per_gpu,
        }

    def log_summary(self, logger=None) -> None:
        """Print (or log) a human-readable power summary."""
        s = self.summary()
        sep = "─" * 52
        lines = [
            sep,
            "  Power & Energy Summary",
            f"  Duration           : {s['elapsed_s'] / 60:.1f} min "
            f"({s['num_samples']} samples)",
            f"  Mean system power  : {s['mean_power_w']:.1f} W",
            f"  Mean CPU power     : {s['mean_cpu_w']:.1f} W",
            f"  Mean GPU power     : {s['mean_gpu_w']:.1f} W",
            f"  Total energy       : {s['total_energy_kwh']:.4f} kWh",
            f"  Estimated CO2      : {s['total_co2_kg']:.4f} kg",
            f"  GPUs monitored     : {s['num_gpus_monitored']} {s['gpu_ids']}",
        ]
        for gid, info in s["per_gpu"].items():
            lines.append(
                f"    GPU {gid}: {info['mean_power_w']:.1f} W avg, "
                f"{info['energy_kwh']:.4f} kWh, {info['co2_kg']:.4f} kg CO2"
            )
        if s["mean_cpu_w"] == 0.0:
            lines.append(
                "  [!] CPU power unavailable — run: "
                "sudo chmod -R a+r /sys/class/powercap/intel-rapl"
            )
        lines.append(sep)

        msg = "\n".join(lines)
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)


def power_monitor_from_config(cfg, csv_path: Optional[str] = None) -> PowerMonitor:
    """
    Build a PowerMonitor from a Config object.

    Args:
        cfg: Full Config (uses `cfg.power`).
        csv_path: Optional per-sample CSV log path.
    """
    pc = cfg.power
    gpu_ids = list(pc.gpu_ids) if pc.gpu_ids else None
    return PowerMonitor(
        poll_interval=pc.poll_interval_sec,
        rapl_path=pc.rapl_path,
        ram_coeff=pc.ram_coeff,
        ssd_coeff=pc.ssd_coeff,
        other_coeff=pc.other_coeff,
        gpu_ids=gpu_ids if pc.monitor_all_gpus else [],
        per_gpu_log=pc.per_gpu_log,
        co2_kg_per_kwh=pc.co2_kg_per_kwh,
        overhead_multiplier=pc.overhead_multiplier,
        csv_path=csv_path,
    )
