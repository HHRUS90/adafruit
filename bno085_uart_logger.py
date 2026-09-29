#!/usr/bin/env python3
"""
bno085_uart_logger.py – full‑report logger with optional run‑time limit

Features
--------
* All six sensor streams (Euler, Quaternion, Accel, Gyro, Mag)
* Timestamped rows
* Auto‑save to CSV *or* TSV (choose with USE_CSV flag)
* Starts recording on launch (perfect for boot‑time start‑up)
* Can be given a `--duration <seconds>` argument:
      – If supplied, the script stops automatically after that many seconds.
      – If omitted, it runs indefinitely until you press Ctrl‑C or
        the service is stopped.
* Graceful cleanup on SIGINT, SIGTERM or on timer expiry.
"""

# --------------------------------------------------------------
# 0️⃣  Imports & command‑line parsing
# --------------------------------------------------------------
import sys, time, datetime, csv, signal, argparse
from pathlib import Path

import board, busio, adafruit_bno08x.uart
import matplotlib.pyplot as plt, matplotlib.animation as animation

# --------------------------------------------------------------
# 1️⃣  USER SETTINGS (tweak before you run)
# --------------------------------------------------------------
UART_DEVICE      = "/dev/serial0"
BAUDRATE         = 921_600          # 921 600 baud → 400 Hz max
MAX_SECONDS      = 30              # seconds shown on the live plot
LOG_ROOT         = Path.cwd() / "logs"

# Choose ONE of the two output formats (comment the one you don’t want)
USE_CSV = True      # CSV → .csv  (Excel‑friendly)
#USE_CSV = False    # TSV → .txt  (slightly smaller)

# --------------------------------------------------------------
# 2️⃣  Parse optional “--duration” argument
# --------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Log all BNO085 reports.  "
                "If --duration is omitted the program runs until you stop it."
)
parser.add_argument(
    "--duration",
    type=float,
    help="Run time in seconds (e.g. 1800 for 30 min).  Omit for infinite run.",
)
args = parser.parse_args()
RUN_FOR_SECONDS = args.duration   # None → run forever

# --------------------------------------------------------------
# 3️⃣  CSV / TSV header (same layout for both formats)
# --------------------------------------------------------------
CSV_HEADER = [
    "timestamp",
    "roll_deg", "pitch_deg", "yaw_deg",
    "quat_i", "quat_j", "quat_k", "quat_real",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
]

# --------------------------------------------------------------
# 4️⃣  Helper functions (file handling, degree conversion, etc.)
# --------------------------------------------------------------
def deg_from_rad(rad: float) -> float:
    return rad * 180.0 / 3.141592653589793

def _make_log_path() -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "csv" if USE_CSV else "txt"
    return LOG_ROOT / f"bno085_{stamp}.{suffix}"

def _open_log_file(path: Path):
    """Open file for *append* (line‑buffered). Returns (handle, csv_writer|None)."""
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
# 5️⃣  UART / sensor start‑up
# --------------------------------------------------------------
print("[INFO] Initialising UART …")
uart = busio.UART(board.TX, board.RX, baudrate=BAUDRATE)
print("[INFO] Creating BNO085 driver …")
bno = adafruit_bno08x.uart.BNO08X_UART(uart)

# Enable every report we intend to log
bno.enable_feature(adafruit_bno08x.SHORT_EULER)
bno.enable_feature(adafruit_bno08x.QUATERNION)
bno.enable_feature(adafruit_bno08x.ACCELEROMETER)
bno.enable_feature(adafruit_bno08x.GYROSCOPE)
bno.enable_feature(adafruit_bno08x.MAGNETOMETER)

# --------------------------------------------------------------
# 6️⃣  Open a fresh log file (one file per power‑up / run)
# --------------------------------------------------------------
LOG_ROOT.mkdir(parents=True, exist_ok=True)
log_path = _make_log_path()
log_file, log_writer = _open_log_file(log_path)
print(f"[INFO] Logging to {log_path}")

# --------------------------------------------------------------
# 7️⃣  Plot set‑up (unchanged from the original version)
# --------------------------------------------------------------
plt.style.use("seaborn-darkgrid")
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Live BNO085 Euler Angles (degrees)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Angle (°)")
ax.set_ylim(-180, 180)
ax.grid(True)

time_vals, roll_vals, pitch_vals, yaw_vals = [], [], [], []
line_roll,  = ax.plot([], [], label="Roll",  color="#ff5555")
line_pitch, = ax.plot([], [], label="Pitch", color="#55ff55")
line_yaw,   = ax.plot([], [], label="Yaw",   color="#5555ff")
ax.legend(loc="upper right")

start_time = time.time()          # for the live‑plot X‑axis
run_start   = start_time          # for the optional timer

def init_plot():
    line_roll.set_data([], [])
    line_pitch.set_data([], [])
    line_yaw.set_data([], [])
    return line_roll, line_pitch, line_yaw

def update_plot(frame):
    """
    Called repeatedly by Matplotlib’s animation loop.
    * Pulls the latest sensor packet (non‑blocking)
    * Writes a timestamped row to the chosen log format
    * Updates the live plot buffers
    """
    # ---- 1️⃣  Grab the newest packet (may be None if nothing arrived) ----
    euler = bno.euler
    quat   = bno.quaternion
    accel  = bno.acceleration
    gyro   = bno.gyro
    mag    = bno.magnetic

    if euler is None:
        # No fresh data – just keep the plot as‑is
        return line_roll, line_pitch, line_yaw

    # ---- 2️⃣  Assemble the row -------------------------------------------------
    now = time.time() - start_time
    ts  = datetime.datetime.now().isoformat()

    roll_deg  = deg_from_rad(euler[0])
    pitch_deg = deg_from_rad(euler[1])
    yaw_deg   = deg_from_rad(euler[2])

    row = [
        ts,
        roll_deg, pitch_deg, yaw_deg,
        quat[0], quat[1], quat[2], quat[3],
        accel[0], accel[1], accel[2],
        gyro[0], gyro[1], gyro[2],
        mag[0], mag[1], mag[2],
    ]

    # ---- 3️⃣  Write to disk (CSV or TSV) ---------------------------------------
    if USE_CSV:
        _write_row_csv(log_writer, row)
    else:
        _write_row_txt(log_file, row)

    # ---- 4️⃣  Update the rolling plot buffers -----------------------------------
    time_vals.append(now)
    roll_vals.append(roll_deg)
    pitch_vals.append(pitch_deg)
    yaw_vals.append(yaw_deg)

    # keep only the last MAX_SECONDS seconds in memory
    while time_vals and (now - time_vals[0] > MAX_SECONDS):
        time_vals.pop(0)
        roll_vals.pop(0)
        pitch_vals.pop(0)
        yaw_vals.pop(0)

    # ---- 5️⃣  Refresh Matplotlib objects ---------------------------------------
    line_roll.set_data(time_vals, roll_vals)
    line_pitch.set_data(time_vals, pitch_vals)
    line_yaw.set_data(time_vals, yaw_vals)
    ax.set_xlim(max(0, now - MAX_SECONDS), now + 0.5)

    # ---- 6️⃣  **Timer check** – stop if we’ve reached the user‑defined limit --- 
    if RUN_FOR_SECONDS is not None:
        elapsed = time.time() - run_start
        if elapsed >= RUN_FOR_SECONDS:
            print(f"\n[INFO] Run‑time limit of {RUN_FOR_SECONDS:.1f}s reached – stopping.")
            _cleanup_and_exit()

    return line_roll, line_pitch, line_yaw

# --------------------------------------------------------------
# 8️⃣  Graceful shutdown helpers
# --------------------------------------------------------------
def _cleanup_and_exit():
    """Close files, the plot, and exit the process."""
    try:
        log_file.close()
    except Exception:  # pragma: no cover
        pass
    plt.close(fig)
    sys.exit(0)

def _signal_handler(sig, frame):
    """Catch Ctrl‑C (SIGINT) or SIGTERM and shut down cleanly."""
    print("\n[INFO] Signal received – cleaning up and exiting …")
    _cleanup_and_exit()

signal.signal(signal.SIGINT,  _signal_handler)   # Ctrl‑C from terminal
signal.signal(signal.SIGTERM, _signal_handler)   # systemd stop command

# --------------------------------------------------------------
# 9️⃣  Start the Matplotlib animation (the main loop)
# --------------------------------------------------------------
ani = animation.FuncAnimation(
    fig,
    update_plot,
    init_func=init_plot,
    interval=5,            # ms → ~200 Hz update (sensor dictates max)
    blit=True,
)

print("[INFO] Recording started …")
if RUN_FOR_SECONDS is not None:
    print(f"[INFO] Will auto‑stop after {RUN_FOR_SECONDS:.1f} seconds.")
else:
    print("[INFO] Running until you press Ctrl‑C or stop the service.")

plt.show()
