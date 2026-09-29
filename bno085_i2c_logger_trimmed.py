#!/usr/bin/env python3
"""
bno085_i2c_logger_trimmed.py – log a single sensor (accel, gyro or mag) over I²C

Features
--------
* Raspberry‑Pi hardware I²C (SDA = GPIO2, SCL = GPIO3) at 400 kHz.
* User selects ONE measurement (accelerometer, gyroscope or magnetometer) via
  a command‑line flag.  The script enables **only that report** and sets the
  report period to the maximum rate allowed by the datasheet:
      – Accel  : 500 Hz  (2 ms → 2 000 µs)
      – Gyro   : 400 Hz  (2.5 ms → 2 500 µs)
      – Mag    : 100 Hz  (10 ms → 10 000 µs)
* Auto‑saves to CSV *or* TSV (toggle with the USE_CSV constant at the top).
* Optional live Matplotlib plot – **enabled only with `--plot`**.
* Optional `--duration <seconds>` argument to stop automatically.
* Log files are named `<measurement>_YYYYMMDD_HHMMSS.<ext>` (e.g.
  `bno085_i2c_accel_20260928_145408.csv`).
* Every 400 ms the script can print the measured sample‑rate (Hz);
  disable it with `--no‑rate`.
* Graceful cleanup on Ctrl‑C, SIGTERM, or timer expiry.

Author: <your‑name>
Date  : 2026‑09‑29
"""

# --------------------------------------------------------------
# 0️⃣  Imports & command‑line parsing
# --------------------------------------------------------------
import sys, time, datetime, csv, signal, argparse, os, collections
from pathlib import Path

import board, busio, adafruit_bno08x.i2c
import matplotlib.pyplot as plt, matplotlib.animation as animation

# --------------------------------------------------------------
# 1️⃣  USER SETTINGS (edit before you run)
# --------------------------------------------------------------
I2C_FREQUENCY = 400_000            # 400 kHz – Pi fast‑mode (max supported)
MAX_SECONDS   = 30                 # seconds shown on the live plot (if enabled)
LOG_ROOT      = Path.cwd() / "logs"

# Choose ONE of the two output formats (comment the one you don’t want)
USE_CSV = True   # CSV → .csv (Excel‑friendly)
#USE_CSV = False # TSV → .txt (a little smaller)

# --------------------------------------------------------------
# 2️⃣  Command‑line arguments
# --------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Log ONE BNO085 measurement (accel, gyro or mag) over I2C. "
                "Optional flags: --duration, --plot, --no-rate."
)

# ---- measurement selection – mutually exclusive & required -------------
meas_group = parser.add_mutually_exclusive_group(required=True)
meas_group.add_argument("--accel", action="store_true", help="Log accelerometer only (max 500 Hz).")
meas_group.add_argument("--gyro",  action="store_true", help="Log gyroscope only (max 400 Hz).")
meas_group.add_argument("--mag",   action="store_true", help="Log magnetometer only (max 100 Hz).")

# ---- other optional flags -------------------------------------------
parser.add_argument(
    "--duration",
    type=float,
    help="Run time in seconds (e.g. 1800 for 30 min). Omit for infinite run."
)
parser.add_argument(
    "--plot",
    action="store_true",
    help="Enable live Matplotlib plot (disabled by default for max throughput)."
)
parser.add_argument(
    "--no-rate",
    action="store_true",
    help="Do NOT print the sample‑rate to the console (enabled by default)."
)

args = parser.parse_args()
RUN_FOR_SECONDS = args.duration               # None → run forever
USE_PLOT        = args.plot                    # Default: False (max‑throughput)
PRINT_RATE      = not args.no_rate            # Default: True

# --------------------------------------------------------------
# 3️⃣  Determine which measurement we are logging
# --------------------------------------------------------------
if args.accel:
    MEAS_NAME   = "accel"
    REPORT_TYPE = adafruit_bno08x.BNO_REPORT_ACCELEROMETER
    MAX_HZ      = 500
    HEADER_FIELDS = ["accel_x", "accel_y", "accel_z"]
elif args.gyro:
    MEAS_NAME   = "gyro"
    REPORT_TYPE = adafruit_bno08x.BNO_REPORT_GYROSCOPE
    MAX_HZ      = 400
    HEADER_FIELDS = ["gyro_x", "gyro_y", "gyro_z"]
else:  # args.mag
    MEAS_NAME   = "mag"
    REPORT_TYPE = adafruit_bno08x.BNO_REPORT_MAGNETOMETER
    MAX_HZ      = 100
    HEADER_FIELDS = ["mag_x", "mag_y", "mag_z"]

# Compute the report period in micro‑seconds (µs)
REPORT_PERIOD_US = int(1_000_000 / MAX_HZ)   # e.g. 500 Hz → 2000 µs

# --------------------------------------------------------------
# 4️⃣  CSV / TSV header (timestamp + chosen measurement fields)
# --------------------------------------------------------------
CSV_HEADER = ["timestamp"] + HEADER_FIELDS

# --------------------------------------------------------------
# 5️⃣  Helper functions
# --------------------------------------------------------------
def _make_log_path() -> Path:
    """Create a unique file name like bno085_i2c_<meas>_YYYYMMDD_HHMMSS.<ext>."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "csv" if USE_CSV else "txt"
    return LOG_ROOT / f"bno085_i2c_{MEAS_NAME}_{stamp}.{suffix}"

def _open_log_file(path: Path):
    """Open the log file line‑buffered; return (handle, csv.writer|None)."""
    if USE_CSV:
        f = open(path, "w", newline="", buffering=1)
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        return f, w
    else:
        f = open(path, "w", buffering=1)
        f.write("\t".join(CSV_HEADER) + "\n")
        return f, None

def _write_row_csv(writer, row):
    writer.writerow(row)

def _write_row_txt(f, row):
    f.write("\t".join(str(v) for v in row) + "\n")

# --------------------------------------------------------------
# 6️⃣  I²C bus & sensor initialization (enable only the chosen report)
# --------------------------------------------------------------
print("[INFO] Initialising I²C …")
i2c = busio.I2C(board.SCL, board.SDA, frequency=I2C_FREQUENCY)

print("[INFO] Creating BNO085 driver (I2C) …")
bno = adafruit_bno08x.i2c.BNO08X_I2C(i2c)

# Enable the single report the user asked for and set its maximum period
bno.enable_feature(REPORT_TYPE)
bno.set_report_period(REPORT_TYPE, REPORT_PERIOD_US)

# --------------------------------------------------------------
# 7️⃣  Open a fresh log file (auto‑named by start time & measurement)
# --------------------------------------------------------------
LOG_ROOT.mkdir(parents=True, exist_ok=True)
log_path = _make_log_path()
log_file, log_writer = _open_log_file(log_path)
print(f"[INFO] Logging {MEAS_NAME.upper()} data to {log_path}")

# --------------------------------------------------------------
# 8️⃣  Sample‑rate statistics (updated every 0.4 s)
# --------------------------------------------------------------
sample_times = collections.deque(maxlen=2000)   # store recent timestamps
LAST_RATE_PRINT = time.time()
RATE_PRINT_INTERVAL = 0.4                     # seconds (twice per second)

def _maybe_print_rate():
    """Print current sample rate (Hz) if interval elapsed and printing enabled."""
    global LAST_RATE_PRINT
    if not PRINT_RATE:
        return

    now = time.time()
    if now - LAST_RATE_PRINT >= RATE_PRINT_INTERVAL:
        # keep only timestamps from the last second for a smoother estimate
        while sample_times and (now - sample_times[0] > 1.0):
            sample_times.popleft()

        if sample_times:
            elapsed = now - sample_times[0]
            hz = len(sample_times) / elapsed if elapsed > 0 else 0.0
            print(f"[INFO] Current {MEAS_NAME} sample rate ≈ {hz:.1f} Hz")
        LAST_RATE_PRINT = now

# --------------------------------------------------------------
# 9️⃣  Plot set‑up (only if USE_PLOT is True)
# --------------------------------------------------------------
if USE_PLOT:
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"Live {MEAS_NAME.capitalize()} (3 axes)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{MEAS_NAME.capitalize()}")

    # Choose sensible Y‑limits per sensor type
    if MEAS_NAME == "accel":
        ax.set_ylim(-20, 20)          # m/s²
    elif MEAS_NAME == "gyro":
        ax.set_ylim(-500, 500)        # rad/s (wide enough for most rotations)
    else:  # mag
        ax.set_ylim(-2000, 2000)      # µT (typical earth field range)

    ax.grid(True)

    time_vals = []
    v1_vals, v2_vals, v3_vals = [], [], []   # three axes

    line1, = ax.plot([], [], label=f"{MEAS_NAME[0].upper()}1", color="#ff5555")
    line2, = ax.plot([], [], label=f"{MEAS_NAME[0].upper()}2", color="#55ff55")
    line3, = ax.plot([], [], label=f"{MEAS_NAME[0].upper()}3", color="#5555ff")
    ax.legend(loc="upper right")
else:
    # We still need a reference start‑time for timestamps.
    time_vals = None

start_time = time.time()           # for timestamps (and plot X‑axis if used)
run_start   = start_time           # for the optional duration timer

# --------------------------------------------------------------
# 🔟  Graceful cleanup helpers
# --------------------------------------------------------------
def _cleanup_and_exit():
    """Flush/close the log file, close the plot (if any), and exit."""
    try:
        log_file.flush()
        os.fsync(log_file.fileno())
        log_file.close()
    except Exception:               # pragma: no cover
        pass
    if USE_PLOT:
        plt.close(fig)
    sys.exit(0)

def _signal_handler(sig, frame):
    print("\n[INFO] Signal received – cleaning up and exiting …")
    _cleanup_and_exit()

signal.signal(signal.SIGINT,  _signal_handler)   # Ctrl‑C
signal.signal(signal.SIGTERM, _signal_handler)   # systemd stop

# --------------------------------------------------------------
# 🔄  Main acquisition loop – works with or without plotting
# --------------------------------------------------------------
while True:
    # ---- 1️⃣  Pull the selected raw report (non‑blocking) --------------------
    if MEAS_NAME == "accel":
        data = bno.acceleration   # (x, y, z) – m/s²
    elif MEAS_NAME == "gyro":
        data = bno.gyro           # (x, y, z) – rad/s
    else:
        data = bno.magnetic       # (x, y, z) – µT

    # If no fresh packet, just loop (still allow rate printing)
    if data is None:
        _maybe_print_rate()
        continue

    # ---- 2️⃣  Assemble timestamped CSV/TSV row -----------------------------
    now = time.time() - start_time
    ts  = datetime.datetime.now().isoformat()

    row = [ts, data[0], data[1], data[2]]

    if USE_CSV:
        _write_row_csv(log_writer, row)
    else:
        _write_row_txt(log_file, row)

    # ---- 3️⃣  Record sample time for rate calculation ----------------------
    sample_times.append(time.time())

    # ---- 4️⃣  Plot update (if enabled) ------------------------------------
    if USE_PLOT:
        time_vals.append(now)
        v1_vals.append(data[0])
        v2_vals.append(data[1])
        v3_vals.append(data[2])

        # Trim buffers to the last MAX_SECONDS seconds
        while time_vals and (now - time_vals[0] > MAX_SECONDS):
            time_vals.pop(0)
            v1_vals.pop(0)
            v2_vals.pop(0)
            v3_vals.pop(0)

        line1.set_data(time_vals, v1_vals)
        line2.set_data(time_vals, v2_vals)
        line3.set_data(time_vals, v3_vals)
        ax.set_xlim(max(0, now - MAX_SECONDS), now + 0.5)

        # Minimal pause – keeps GUI responsive with virtually no overhead.
        plt.pause(0.001)

    # ---- 5️⃣  Print sample rate (every 0.4 s) -------------------------------
    _maybe_print_rate()

    # ---- 6️⃣  Duration timer check -----------------------------------------
    if RUN_FOR_SECONDS is not None:
        if (time.time() - run_start) >= RUN_FOR_SECONDS:
            print(f"\n[INFO] Run‑time limit of {RUN_FOR_SECONDS:.1f}s reached – stopping.")
            _cleanup_and_exit()
