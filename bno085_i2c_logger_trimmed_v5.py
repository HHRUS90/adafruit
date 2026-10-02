#!/usr/bin/env python3
"""
bno085_i2c_logger_trimmed.py – log a single sensor (accel, gyro or mag) over I²C

Features
--------
* Raspberry‑Pi hardware I²C (SDA = GPIO2, SCL = GPIO3) at 400 kHz.
* User selects ONE measurement (accelerometer, gyroscope or magnetometer) via
  a command‑line flag.
* Auto‑saves to CSV *or* TSV.
* Live Matplotlib plot – fixed to render correctly using FuncAnimation.
* Graceful cleanup on Ctrl‑C, SIGTERM, or timer expiry.
"""

import sys, time, datetime, csv, signal, argparse, os, collections
from pathlib import Path

import board, busio, adafruit_bno08x.i2c

# --------------------------------------------------------------
# 🔄 FORCE LOCAL MONITOR DISPLAY OVERRIDE
# --------------------------------------------------------------
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0.0"  # Forces Python to look at your local Pi screen

import matplotlib
matplotlib.use('TkAgg')  # Force the interactive window engine

import matplotlib.pyplot as plt
import matplotlib.animation as animation



# --------------------------------------------------------------
# 1️⃣  USER SETTINGS
# --------------------------------------------------------------
I2C_FREQUENCY = 400_000            # 400 kHz – Pi fast‑mode (max supported)
MAX_SECONDS   = 30                 # seconds shown on the live plot (if enabled)
LOG_ROOT      = Path.cwd() / "logs"
USE_CSV       = True               # CSV → .csv, False → TSV

# --------------------------------------------------------------
# 2️⃣  Command‑line arguments
# --------------------------------------------------------------
parser = argparse.ArgumentParser(description="Log ONE BNO085 measurement over I2C.")
meas_group = parser.add_mutually_exclusive_group(required=True)
meas_group.add_argument("--accel", action="store_true", help="Log accelerometer only (max 500 Hz).")
meas_group.add_argument("--gyro",  action="store_true", help="Log gyroscope only (max 400 Hz).")
meas_group.add_argument("--mag",   action="store_true", help="Log magnetometer only (max 100 Hz).")

parser.add_argument("--duration", type=float, help="Run time in seconds. Omit for infinite run.")
parser.add_argument("--plot", action="store_true", help="Enable live Matplotlib plot.")
parser.add_argument("--no-rate", action="store_true", help="Do NOT print the sample‑rate.")

args = parser.parse_args()
RUN_FOR_SECONDS = args.duration
USE_PLOT        = args.plot
PRINT_RATE      = not args.no_rate

# --------------------------------------------------------------
# 3️⃣  Determine measurement configuration
# --------------------------------------------------------------
if args.accel:
    MEAS_NAME, REPORT_TYPE, MAX_HZ = "accel", adafruit_bno08x.BNO_REPORT_ACCELEROMETER, 500
    HEADER_FIELDS = ["accel_x", "accel_y", "accel_z"]
elif args.gyro:
    MEAS_NAME, REPORT_TYPE, MAX_HZ = "gyro", adafruit_bno08x.BNO_REPORT_GYROSCOPE, 400
    HEADER_FIELDS = ["gyro_x", "gyro_y", "gyro_z"]
else:
    MEAS_NAME, REPORT_TYPE, MAX_HZ = "mag", adafruit_bno08x.BNO_REPORT_MAGNETOMETER, 100
    HEADER_FIELDS = ["mag_x", "mag_y", "mag_z"]

REPORT_PERIOD_US = int(1_000_000 / MAX_HZ)
CSV_HEADER = ["timestamp"] + HEADER_FIELDS

# --------------------------------------------------------------
# 4️⃣  Helper functions
# --------------------------------------------------------------
def _make_log_path() -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "csv" if USE_CSV else "txt"
    return LOG_ROOT / f"bno085_i2c_{MEAS_NAME}_{stamp}.{suffix}"

def _open_log_file(path: Path):
    if USE_CSV:
        f = open(path, "w", newline="", buffering=1)
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        return f, w
    else:
        f = open(path, "w", buffering=1)
        f.write("\t".join(CSV_HEADER) + "\n")
        return f, None

# --------------------------------------------------------------
# 5️⃣  Initialization
# --------------------------------------------------------------
print("[INFO] Initialising I²C …")
i2c = busio.I2C(board.SCL, board.SDA, frequency=I2C_FREQUENCY)
bno = adafruit_bno08x.i2c.BNO08X_I2C(i2c)
bno.enable_feature(REPORT_TYPE, REPORT_PERIOD_US)

LOG_ROOT.mkdir(parents=True, exist_ok=True)
log_path = _make_log_path()
log_file, log_writer = _open_log_file(log_path)
print(f"[INFO] Logging {MEAS_NAME.upper()} data to {log_path}")

sample_times = collections.deque(maxlen=2000)
LAST_RATE_PRINT = time.time()
RATE_PRINT_INTERVAL = 0.4

def _maybe_print_rate():
    global LAST_RATE_PRINT
    if not PRINT_RATE:
        return
    now = time.time()
    if now - LAST_RATE_PRINT >= RATE_PRINT_INTERVAL:
        while sample_times and (now - sample_times[0] > 1.0):
            sample_times.popleft()
        if sample_times:
            elapsed = now - sample_times[0]
            hz = len(sample_times) / elapsed if elapsed > 0 else 0.0
            print(f"[INFO] Current {MEAS_NAME} sample rate ≈ {hz:.1f} Hz")
        LAST_RATE_PRINT = now

# Tracking data metrics
start_time = time.time()
run_start = start_time
time_vals, v1_vals, v2_vals, v3_vals = [], [], [], []

# --------------------------------------------------------------
# 6️⃣  Cleanup and Exit Handler
# --------------------------------------------------------------
def _cleanup_and_exit():
    try:
        log_file.flush()
        os.fsync(log_file.fileno())
        log_file.close()
    except Exception:
        pass
    print("\n[INFO] Cleanup finished. Exiting.")
    sys.exit(0)

def _signal_handler(sig, frame):
    print("\n[INFO] Signal received – cleaning up …")
    _cleanup_and_exit()

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# --------------------------------------------------------------
# 7️⃣  Core Data Harvesting Logic
# --------------------------------------------------------------
def get_sensor_data():
    """Reads sensor data, logs it, and calculates metrics."""
    if MEAS_NAME == "accel":
        data = bno.acceleration
    elif MEAS_NAME == "gyro":
        data = bno.gyro
    else:
        data = bno.magnetic

    if data is None:
        _maybe_print_rate()
        return None

    now = time.time() - start_time
    ts = datetime.datetime.now().isoformat()
    row = [ts, data[0], data[1], data[2]]

    if USE_CSV:
        log_writer.writerow(row)
    else:
        log_file.write("\t".join(str(v) for v in row) + "\n")

    sample_times.append(time.time())
    _maybe_print_rate()

    # Duration check
    if RUN_FOR_SECONDS is not None and (time.time() - run_start) >= RUN_FOR_SECONDS:
        print(f"\n[INFO] Run‑time limit of {RUN_FOR_SECONDS:.1f}s reached.")
        _cleanup_and_exit()

    return now, data

# --------------------------------------------------------------
# 8️⃣  Execution Block (Plotting vs High-Throughput Headless)
# --------------------------------------------------------------
if USE_PLOT:
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"Live {MEAS_NAME.capitalize()} (3 axes)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{MEAS_NAME.capitalize()}")

    if MEAS_NAME == "accel":
        ax.set_ylim(-20, 20)
    elif MEAS_NAME == "gyro":
        ax.set_ylim(-500, 500)
    else:
        ax.set_ylim(-2000, 2000)

    ax.grid(True)

    line1, = ax.plot([], [], label=f"{HEADER_FIELDS[0]}", color="#ff5555")
    line2, = ax.plot([], [], label=f"{HEADER_FIELDS[1]}", color="#55ff55")
    line3, = ax.plot([], [], label=f"{HEADER_FIELDS[2]}", color="#5555ff")
    ax.legend(loc="upper right")

    def update_plot(frame):
        # Read up to 5 samples per frame to clear the I2C cache without trapping the UI thread
        for _ in range(5):
            result = get_sensor_data()
            if result is None:
                break
            
            now, data = result
            time_vals.append(now)
            v1_vals.append(data[0])
            v2_vals.append(data[1])
            v3_vals.append(data[2])

        # Trim old values to maintain historical window
        if time_vals:
            current_now = time_vals[-1]
            while time_vals and (current_now - time_vals[0] > MAX_SECONDS):
                time_vals.pop(0)
                v1_vals.pop(0)
                v2_vals.pop(0)
                v3_vals.pop(0)

            # Update line structures and adjust view bounds dynamically
            line1.set_data(time_vals, v1_vals)
            line2.set_data(time_vals, v2_vals)
            line3.set_data(time_vals, v3_vals)
            ax.set_xlim(max(0, current_now - MAX_SECONDS), current_now + 0.5)

        return line1, line2, line3


    # FuncAnimation natively handles the window updates and window close events
    anim = animation.FuncAnimation(fig, update_plot, interval=20, cache_frame_data=False)
    plt.show()

else:
    # Headless loop optimized for maximum data capture speeds
    print("[INFO] Running in headless mode. Press Ctrl+C to stop.")
    while True:
        get_sensor_data()
        # Minor rest to lower heavy CPU load when no data is waiting on I2C
        time.sleep(0.001)
