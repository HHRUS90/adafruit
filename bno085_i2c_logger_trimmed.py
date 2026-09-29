#!/usr/bin/env python3
"""
bno085_i2c_logger_trimmed.py – log accel/gyro/mag over I²C
with optional live plot, auto‑named files, optional rate printing,
and a flag to enable plotting (default: no plot, max‑throughput mode).

Features
--------
* Raspberry‑Pi hardware I²C (SDA = GPIO2, SCL = GPIO3) at 400 kHz.
* Records accelerometer, gyroscope and magnetometer + a timestamp.
* Auto‑saves to CSV *or* TSV (toggle with USE_CSV flag at the top).
* Optional live Matplotlib plot – **enabled only with `--plot`**.
* Optional `--duration <seconds>` argument to stop automatically.
* Each run creates a file named 20260929‑T1428.csv (or .txt) – the
  date‑time when the script starts.
* Every 400 ms the script can print the measured sample‑rate (Hz);
  disable it with the `--no-rate` flag.
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
    description="Log BNO085 accel/gyro/mag over I2C. "
                "Optional flags: --duration, --plot, --no-rate."
)
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
USE_PLOT        = args.plot                    # **default is False**
PRINT_RATE      = not args.no_rate            # default True, can be disabled

# --------------------------------------------------------------
# 3️⃣  CSV / TSV header (only the three raw streams)
# --------------------------------------------------------------
CSV_HEADER = [
    "timestamp",
    "accel_x", "accel_y", "accel_z",
    "gyro_x",  "gyro_y",  "gyro_z",
    "mag_x",   "mag_y",   "mag_z",
]

# --------------------------------------------------------------
# 4️⃣  Helper functions
# --------------------------------------------------------------
def _make_log_path() -> Path:
    """Create a unique file name like 20260929‑T1428.csv based on start time."""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%TH%M")
    suffix = "csv" if USE_CSV else "txt"
    return LOG_ROOT / f"bno085_i2c_{stamp}.{suffix}"

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
# 5️⃣  I²C bus & sensor initialization (enable raw streams)
# --------------------------------------------------------------
print("[INFO] Initialising I²C …")
i2c = busio.I2C(board.SCL, board.SDA, frequency=I2C_FREQUENCY)

print("[INFO] Creating BNO085 driver (I2C) …")
bno = adafruit_bno08x.i2c.BNO08X_I2C(i2c)

# Enable the three raw reports we care about
bno.enable_feature(adafruit_bno08x.BNO_REPORT_ACCELEROMETER)
bno.enable_feature(adafruit_bno08x.BNO_REPORT_GYROSCOPE)
bno.enable_feature(adafruit_bno08x.BNO_REPORT_MAGNETOMETER)

# --------------------------------------------------------------
# 6️⃣  Open a fresh log file (auto‑named by start time)
# --------------------------------------------------------------
LOG_ROOT.mkdir(parents=True, exist_ok=True)
log_path = _make_log_path()
log_file, log_writer = _open_log_file(log_path)
print(f"[INFO] Logging to {log_path}")

# --------------------------------------------------------------
# 7️⃣  Sample‑rate statistics (updated every 0.4 s)
# --------------------------------------------------------------
sample_times = collections.deque(maxlen=2000)   # store recent timestamps
LAST_RATE_PRINT = time.time()
RATE_PRINT_INTERVAL = 0.4                     # seconds (now prints twice per sec)

def _maybe_print_rate():
    """Print current sample rate (Hz) if the interval has elapsed and printing is enabled."""
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
            print(f"[INFO] Current sample rate ≈ {hz:.1f} Hz")

        LAST_RATE_PRINT = now

# --------------------------------------------------------------
# 8️⃣  Plot set‑up (only if USE_PLOT is True)
# --------------------------------------------------------------
if USE_PLOT:
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Live Accelerometer (m/s²)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accel (m/s²)")
    ax.set_ylim(-20, 20)               # adjust if you expect higher g‑forces
    ax.grid(True)

    time_vals, ax_vals, ay_vals, az_vals = [], [], [], []
    line_ax, = ax.plot([], [], label="Ax", color="#ff5555")
    line_ay, = ax.plot([], [], label="Ay", color="#55ff55")
    line_az, = ax.plot([], [], label="Az", color="#5555ff")
    ax.legend(loc="upper right")
else:
    # Even without a plot we still need a reference for timestamps.
    time_vals = None

start_time = time.time()           # for timestamps and (optional) plot X‑axis
run_start   = start_time           # for the optional duration timer

# --------------------------------------------------------------
# 9️⃣  Graceful cleanup helpers
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
    # ---- 1️⃣  Pull the three raw reports (non‑blocking) --------------------
    accel = bno.acceleration   # (x, y, z) – m/s²
    gyro  = bno.gyro           # (x, y, z) – rad/s
    mag   = bno.magnetic       # (x, y, z) – µT

    # If no fresh packet, just continue (still allow rate printing)
    if accel is None:
        _maybe_print_rate()
        continue

    # ---- 2️⃣  Assemble timestamped CSV/TSV row ---------------------------
    now = time.time() - start_time
    ts  = datetime.datetime.now().isoformat()

    row = [
        ts,
        accel[0], accel[1], accel[2],
        gyro[0],  gyro[1],  gyro[2],
        mag[0],   mag[1],   mag[2],
    ]

    if USE_CSV:
        _write_row_csv(log_writer, row)
    else:
        _write_row_txt(log_file, row)

    # ---- 3️⃣  Record sample time for rate calculation --------------------
    sample_times.append(time.time())

    # ---- 4️⃣  Plot update (if enabled) -----------------------------------
    if USE_PLOT:
        time_vals.append(now)
        ax_vals.append(accel[0])
        ay_vals.append(accel[1])
        az_vals.append(accel[2])

        # Trim buffers to keep only the last MAX_SECONDS seconds
        while time_vals and (now - time_vals[0] > MAX_SECONDS):
            time_vals.pop(0)
            ax_vals.pop(0)
            ay_vals.pop(0)
            az_vals.pop(0)

        line_ax.set_data(time_vals, ax_vals)
        line_ay.set_data(time_vals, ay_vals)
        line_az.set_data(time_vals, az_vals)
        ax.set_xlim(max(0, now - MAX_SECONDS), now + 0.5)

        # Minimal pause to keep the GUI responsive
        plt.pause(0.001)

    # ---- 5️⃣  Print sample rate (every 0.4 s) ----------------------------
    _maybe_print_rate()

    # ---- 6️⃣  Duration timer check ---------------------------------------
    if RUN_FOR_SECONDS is not None:
        if (time.time() - run_start) >= RUN_FOR_SECONDS:
            print(f"\n[INFO] Run‑time limit of {RUN_FOR_SECONDS:.1f}s reached – stopping.")
            _cleanup_and_exit()
