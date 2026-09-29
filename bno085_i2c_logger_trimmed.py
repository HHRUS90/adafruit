#!/usr/bin/env python3
"""
bno085_i2c_logger_trimmed.py – log only accel, gyro, mag over I2C

Features
--------
* Uses the Raspberry‑Pi’s hardware I²C (SDA = GPIO2, SCL = GPIO3).
* Records accelerometer, gyroscope and magnetometer data + a timestamp.
* Auto‑saves to CSV *or* TSV (choose with the USE_CSV flag at the top).
* Can be started manually or as a systemd service on power‑up.
* Optional --duration <seconds> argument – stops automatically after the
  given time; omit the flag to run until you press Ctrl‑C or stop the service.
* Live Matplotlib plot of the three accelerometer axes (you can replace
  with gyro or mag if you prefer).

Author: <your‑name>
Date  : 2026‑09‑28
"""

# --------------------------------------------------------------
# 0️⃣  Imports & command‑line parsing
# --------------------------------------------------------------
import sys, time, datetime, csv, signal, argparse, os
from pathlib import Path

import board, busio, adafruit_bno08x.i2c
import matplotlib.pyplot as plt, matplotlib.animation as animation

# --------------------------------------------------------------
# 1️⃣  USER SETTINGS (tweak before you run)
# --------------------------------------------------------------
I2C_FREQUENCY = 400_000           # 400 kHz – fast‑mode (max the Pi supports)
MAX_SECONDS   = 30                 # seconds shown on the live plot
LOG_ROOT      = Path.cwd() / "logs"

# Choose ONE of the two output formats (comment the one you don’t want)
USE_CSV = True      # CSV → .csv  (Excel‑friendly)
#USE_CSV = False    # TSV → .txt  (a little smaller)

# --------------------------------------------------------------
# 2️⃣  Optional run‑time limit (seconds)
# --------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Log BNO085 accel/gyro/mag over I2C.  Use --duration to auto‑stop."
)
parser.add_argument(
    "--duration",
    type=float,
    help="Run time in seconds (e.g. 1800 for 30 min).  Omit for infinite run.",
)
args = parser.parse_args()
RUN_FOR_SECONDS = args.duration   # None → run forever

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
    """Create a unique file name based on the current date‑time."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "csv" if USE_CSV else "txt"
    return LOG_ROOT / f"bno085_i2c_{stamp}.{suffix}"

def _open_log_file(path: Path):
    """Open log file line‑buffered. Returns (handle, csv.writer|None)."""
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
# 5️⃣  I²C bus & sensor initialisation (enable only accel/gyro/mag)
# --------------------------------------------------------------
print("[INFO] Initialising I²C …")
i2c = busio.I2C(board.SCL, board.SDA, frequency=I2C_FREQUENCY)

print("[INFO] Creating BNO085 driver (I2C) …")
bno = adafruit_bno08x.i2c.BNO08X_I2C(i2c)

# Enable the three raw reports we care about
bno.enable_feature(adafruit_bno08x.ACCELEROMETER)
bno.enable_feature(adafruit_bno08x.GYROSCOPE)
bno.enable_feature(adafruit_bno08x.MAGNETOMETER)

# --------------------------------------------------------------
# 6️⃣  Open a fresh log file (one per power‑up / run)
# --------------------------------------------------------------
LOG_ROOT.mkdir(parents=True, exist_ok=True)
log_path = _make_log_path()
log_file, log_writer = _open_log_file(log_path)
print(f"[INFO] Logging to {log_path}")

# --------------------------------------------------------------
# 7️⃣  Live plot set‑up (accelerometer X/Y/Z)
# --------------------------------------------------------------
plt.style.use("seaborn-darkgrid")
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Live Accelerometer (m/s²)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Accel (m/s²)")
ax.set_ylim(-20, 20)               # tweak if you expect higher g‑forces
ax.grid(True)

time_vals, ax_vals, ay_vals, az_vals = [], [], [], []
line_ax, = ax.plot([], [], label="Ax", color="#ff5555")
line_ay, = ax.plot([], [], label="Ay", color="#55ff55")
line_az, = ax.plot([], [], label="Az", color="#5555ff")
ax.legend(loc="upper right")

start_time = time.time()           # for the live‑plot X‑axis
run_start   = start_time           # for the optional timer

def init_plot():
    line_ax.set_data([], [])
    line_ay.set_data([], [])
    line_az.set_data([], [])
    return line_ax, line_ay, line_az

def update_plot(frame):
    """
    1️⃣ Grab latest accel/gyro/mag packets (non‑blocking)
    2️⃣ Write a timestamped row to CSV / TSV
    3️⃣ Update the rolling plot buffers
    4️⃣ Stop automatically if a --duration limit was supplied
    """
    # -------- 1️⃣  Read the three reports ------------------------------------
    accel = bno.acceleration   # (x, y, z) – m/s²
    gyro  = bno.gyro           # (x, y, z) – rad/s
    mag   = bno.magnetic       # (x, y, z) – µT

    # If nothing arrived yet, just keep the old plot
    if accel is None:
        return line_ax, line_ay, line_az

    # -------- 2️⃣  Assemble timestamped row ------------------------------
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

    # -------- 3️⃣  Update the plot buffers ---------------------------------
    time_vals.append(now)
    ax_vals.append(accel[0])
    ay_vals.append(accel[1])
    az_vals.append(accel[2])

    # Keep only the last MAX_SECONDS seconds in memory
    while time_vals and (now - time_vals[0] > MAX_SECONDS):
        time_vals.pop(0)
        ax_vals.pop(0)
        ay_vals.pop(0)
        az_vals.pop(0)

    line_ax.set_data(time_vals, ax_vals)
    line_ay.set_data(time_vals, ay_vals)
    line_az.set_data(time_vals, az_vals)
    ax.set_xlim(max(0, now - MAX_SECONDS), now + 0.5)

    # -------- 4️⃣  Timer check (if a duration was given) ------------------
    if RUN_FOR_SECONDS is not None:
        elapsed = time.time() - run_start
        if elapsed >= RUN_FOR_SECONDS:
            print(f"\n[INFO] Run‑time limit of {RUN_FOR_SECONDS:.1f}s reached – stopping.")
            _cleanup_and_exit()

    return line_ax, line_ay, line_az

# --------------------------------------------------------------
# 8️⃣  Graceful cleanup helpers
# --------------------------------------------------------------
def _cleanup_and_exit():
    """Flush/close the log file, close the plot, and exit."""
    try:
        log_file.flush()
        os.fsync(log_file.fileno())   # force OS to write to the SD card
        log_file.close()
    except Exception:               # pragma: no cover
        pass
    plt.close(fig)
    sys.exit(0)

def _signal_handler(sig, frame):
    print("\n[INFO] Signal received – cleaning up and exiting …")
    _cleanup_and_exit()

signal.signal(signal.SIGINT,  _signal_handler)   # Ctrl‑C
signal.signal(signal.SIGTERM, _signal_handler)   # systemd stop

# --------------------------------------------------------------
# 9️⃣  Start the Matplotlib animation (the main loop)
# --------------------------------------------------------------
ani = animation.FuncAnimation(
    fig,
    update_plot,
    init_func=init_plot,
    interval=5,            # ms → ~200 Hz update (sensor limited to 200 Hz on I2C)
    blit=True,
)

print("[INFO] Recording started …")
if RUN_FOR_SECONDS is not None:
    print(f"[INFO] Will auto‑stop after {RUN_FOR_SECONDS:.1f} seconds.")
else:
    print("[INFO] Running until you press Ctrl‑C or stop the service.")

plt.show()
