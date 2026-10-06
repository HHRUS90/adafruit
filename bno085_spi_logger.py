#!/usr/bin/env python3
"""
bno085_spi_logger.py – Log and plot 1‑2 BNO085 sensors over SPI.

Features
--------
* Detects up to two sensors on separate CS pins (default: D5 and D6).
* Configurable to log accel, gyro, mag or any combination.
* Head‑less by default – live Matplotlib plot only when ``--plot`` is given.
* Same command‑line interface as the original I²C version.
"""

import sys
import time
import datetime
import csv
import signal
import argparse
import os
import collections
from pathlib import Path

import board
import busio
import adafruit_bno08x
from adafruit_bno08x.spi import BNO08X_SPI   # <-- SPI class

# --------------------------------------------------------------
# 🔄 FORCE LOCAL MONITOR DISPLAY OVERRIDE
# --------------------------------------------------------------
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0.0"      # Forces Python to look at your local Pi screen

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --------------------------------------------------------------
# 1️⃣ USER & HARDWARE SETTINGS
# --------------------------------------------------------------
SPI_FREQUENCY = 10_000_000          # 10 MHz – safe, fast, and well within Pi limits
MAX_SECONDS   = 30                  # seconds shown on the live plot (if enabled)
LOG_ROOT      = Path.cwd() / "logs"
USE_CSV       = True                # CSV → .csv, False → TSV

# Two possible CS pins – change if your wiring uses different GPIOs
POSSIBLE_CS_PINS = [
    board.CE0,   # hardware CE0  → GPIO 8  (pin 24)
    board.CE1,   # hardware CE1  → GPIO 7  (pin 26)
]

ALL_FEATURES = {
    "accel": (adafruit_bno08x.BNO_REPORT_ACCELEROMETER,
              500,
              ["accel_x", "accel_y", "accel_z"],
              (-20, 20)),
    "gyro":  (adafruit_bno08x.BNO_REPORT_GYROSCOPE,
              400,
              ["gyro_x", "gyro_y", "gyro_z"],
              (-500, 500)),
    "mag":   (adafruit_bno08x.BNO_REPORT_MAGNETOMETER,
              100,
              ["mag_x", "mag_y", "mag_z"],
              (-2000, 2000)),
}

# --------------------------------------------------------------
# 2️⃣ Command‑line arguments  (unchanged from the I²C version)
# --------------------------------------------------------------
parser = argparse.ArgumentParser(description="Log multiple BNO085 sensors dynamically over SPI.")
meas_group = parser.add_mutually_exclusive_group(required=True)
meas_group.add_argument("--accel", action="store_true", help="Log accelerometer only (max 500 Hz).")
meas_group.add_argument("--gyro",  action="store_true", help="Log gyroscope only (max 400 Hz).")
meas_group.add_argument("--mag",   action="store_true", help="Log magnetometer only (max 100 Hz).")
meas_group.add_argument("--all",   action="store_true", help="Log ALL three reports simultaneously.")
parser.add_argument("--duration", type=float, help="Run time in seconds. Omit for infinite run.")
parser.add_argument("--plot", action="store_true", help="Enable live Matplotlib plot.")
parser.add_argument("--no-rate", action="store_true", help="Do NOT print the sample‑rate updates.")
args = parser.parse_args()

RUN_FOR_SECONDS = args.duration
USE_PLOT        = args.plot
PRINT_RATE      = not args.no_rate

# --------------------------------------------------------------
# 3️⃣ Determine which feature set(s) are active
# --------------------------------------------------------------
ACTIVE_FEATURES = {}
if args.all:
    ACTIVE_FEATURES = ALL_FEATURES
else:
    for key in ["accel", "gyro", "mag"]:
        if getattr(args, key):
            ACTIVE_FEATURES[key] = ALL_FEATURES[key]

# --------------------------------------------------------------
# 4️⃣ SPI bus initialisation + sensor discovery
# --------------------------------------------------------------
print("[INFO] Initialising SPI bus at", f"{SPI_FREQUENCY // 1_000_000} MHz …")
spi = busio.SPI(board.SCK, board.MOSI, board.MISO, frequency=SPI_FREQUENCY)

sensors = {}          # key = "S1"/"S2", value = BNO08X_SPI instance
CSV_HEADER = ["timestamp"]

for idx, cs_pin in enumerate(POSSIBLE_CS_PINS, start=1):
    sensor_id = f"S{idx}"
    try:
        print(f"[INFO] Trying to talk to a BNO085 on CS pin {cs_pin} ({sensor_id}) …")
        bno = BNO08X_SPI(spi, cs_pin)      # <-- instantiate the SPI version
        # A quick read forces the driver to talk to the chip; if it fails an exception is raised
        _ = bno.product_id
        sensors[sensor_id] = bno
        print(f"[SUCCESS] Found BNO085 on {sensor_id} (CS pin {cs_pin}).")
        # Build CSV header entries for each active measurement of this sensor
        for feat_name, (_, _, subfields, _) in ACTIVE_FEATURES.items():
            for field in subfields:
                CSV_HEADER.append(f"{sensor_id}_{field}")
    except Exception as e:
        print(f"[INFO] No sensor detected on CS pin {cs_pin} ({sensor_id}). ({e})")

if not sensors:
    print("[ERROR] No BNO085 sensors detected on the SPI bus. Check wiring / CS pins.")
    sys.exit(1)

# --------------------------------------------------------------
# 5️⃣ Enable the requested feature set on every discovered device
# --------------------------------------------------------------
for sensor_id, bno in sensors.items():
    print(f"[INFO] Configuring features for {sensor_id}:")
    for name, (report_type, max_hz, _, _) in ACTIVE_FEATURES.items():
        period_us = int(1_000_000 / max_hz)           # SH‑2 wants the period in µs
        bno.enable_feature(report_type, period_us)   # fire the Set Feature command
        print(f"       -> Enabled {name.upper()} at {max_hz} Hz (period {period_us} µs)")

# --------------------------------------------------------------
# 6️⃣ File‑generation setup (same as before)
# --------------------------------------------------------------
def _make_log_path() -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "csv" if USE_CSV else "txt"
    feat_label = "all" if args.all else "_".join(list(ACTIVE_FEATURES.keys()))
    return LOG_ROOT / f"bno085_spi_{feat_label}_{stamp}.{suffix}"

LOG_ROOT.mkdir(parents=True, exist_ok=True)
log_path = _make_log_path()

if USE_CSV:
    log_file = open(log_path, "w", newline="", buffering=1)
    log_writer = csv.writer(log_file)
    log_writer.writerow(CSV_HEADER)
else:
    log_file = open(log_path, "w", buffering=1)
    log_file.write("\t".join(CSV_HEADER) + "\n")
    log_writer = None

print(f"[INFO] Logging data to {log_path}")

sample_times = collections.deque(maxlen=2000)
LAST_RATE_PRINT = time.time()
RATE_PRINT_INTERVAL = 0.4
run_start = time.time()
start_time = run_start

# --------------------------------------------------------------
# 7️⃣ Helper: periodic sample‑rate printing
# --------------------------------------------------------------
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
            print(f"[INFO] Current polling rate ≈ {hz:.1f} Hz")
        LAST_RATE_PRINT = now

# --------------------------------------------------------------
# 8️⃣ Cleanup & signal handling (unchanged)
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
# 9️⃣ Core data‑harvesting routine
# --------------------------------------------------------------
def get_sensor_data():
    """Read all enabled measurements from every detected sensor, write a row, and
    return a tuple (elapsed‑since‑start, {sensor_id: [float,…]}) for the plot."""
    ts = datetime.datetime.now().isoformat()
    row = [ts]
    data_captured = False
    all_devices_frame = {}

    for sensor_id, bno in sensors.items():
        dev_data = []
        try:
            if "accel" in ACTIVE_FEATURES:
                a = bno.acceleration
                dev_data.extend(a if a is not None else (None, None, None))
            if "gyro" in ACTIVE_FEATURES:
                g = bno.gyro
                dev_data.extend(g if g is not None else (None, None, None))
            if "mag" in ACTIVE_FEATURES:
                m = bno.magnetic
                dev_data.extend(m if m is not None else (None, None, None))

            # If at least one non‑None value arrived, we consider the frame valid
            if any(v is not None for v in dev_data):
                data_captured = True

            # Replace any ``None`` with ``0.0`` so the CSV file stays rectangular
            clean_data = [v if v is not None else 0.0 for v in dev_data]
            row.extend(clean_data)
            all_devices_frame[sensor_id] = clean_data

        except Exception:
            # If the SPI transaction fails, pad with zeroes (same width as active features)
            pad_len = len(ACTIVE_FEATURES) * 3
            row.extend([0.0] * pad_len)
            all_devices_frame[sensor_id] = [0.0] * pad_len

    if not data_captured:
        _maybe_print_rate()
        return None

    # ---- Write to file -------------------------------------------------
    if USE_CSV:
        log_writer.writerow(row)
    else:
        log_file.write("\t".join(str(v) for v in row) + "\n")

    sample_times.append(time.time())
    _maybe_print_rate()

    # ---- Enforce run‑time limit ----------------------------------------
    if RUN_FOR_SECONDS is not None and (time.time() - run_start) >= RUN_FOR_SECONDS:
        print(f"\n[INFO] Run‑time limit of {RUN_FOR_SECONDS:.1f}s reached.")
        _cleanup_and_exit()

    return time.time() - start_time, all_devices_frame

# --------------------------------------------------------------
# 10️⃣ Live plot vs headless operation
# --------------------------------------------------------------
PLOT_WINDOW_OPEN = True

if USE_PLOT:
    plt.style.use("seaborn-v0_8-darkgrid")
    num_plots = len(ACTIVE_FEATURES)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]

    # Mapping from feature → list of (sensor_id, line_handle) that belong to that subplot
    plot_mappings = {}
    colors = ["#ff5555", "#55ff55", "#5555ff", "#ffa500", "#8a2be2", "#00ffff"]

    # Historical data containers (one per sensor, one list per axis)
    plot_data_history = {sid: collections.defaultdict(list) for sid in sensors}
    time_history = {sid: [] for sid in sensors}

    feature_keys = list(ACTIVE_FEATURES.keys())
    for ax_idx, feat_name in enumerate(feature_keys):
        _, _, _, y_bounds = ACTIVE_FEATURES[feat_name]
        axes[ax_idx].set_title(f"Live {feat_name.capitalize()} Streams")
        axes[ax_idx].set_ylabel(feat_name.capitalize())
        axes[ax_idx].set_ylim(y_bounds)
        axes[ax_idx].grid(True)

        plot_mappings[feat_name] = []
        c_idx = 0
        for sid in sensors:
            for axis_lbl in ["X", "Y", "Z"]:
                line, = axes[ax_idx].plot([], [],
                                          label=f"{sid}_{axis_lbl}",
                                          color=colors[c_idx % len(colors)])
                plot_mappings[feat_name].append((sid, line))
                c_idx += 1
        axes[ax_idx].legend(loc="upper right", ncol=2)

    axes[-1].set_xlabel("Time (s)")

    def on_close(event):
        global PLOT_WINDOW_OPEN
        PLOT_WINDOW_OPEN = False
        print("\n[INFO] Plot window closed – switching to headless logging…")

    fig.canvas.mpl_connect("close_event", on_close)

    def update_plot(frame):
        if not PLOT_WINDOW_OPEN:
            return []

        # Pull a few frames from the sensor(s) each animation tick. 4 is a good compromise.
        for _ in range(4):
            result = get_sensor_data()
            if result is None:
                break
            now, devices_frame = result
            for sid, data_list in devices_frame.items():
                time_history[sid].append(now)
                # ``data_list`` contains the three axes of each enabled feature in order.
                for idx, val in enumerate(data_list):
                    plot_data_history[sid][idx].append(val)

                # Trim old data so the rolling window never exceeds MAX_SECONDS
                while time_history[sid] and (now - time_history[sid][0] > MAX_SECONDS):
                    time_history[sid].pop(0)
                    for idx in plot_data_history[sid].keys():
                        plot_data_history[sid][idx].pop(0)

        # Update every line handle
        all_lines = []
        for ax_idx, feat_name in enumerate(feature_keys):
            sub_map = plot_mappings[feat_name]
            for line_idx, (sid, line) in enumerate(sub_map):
                # Axis offset inside the flat data list:
                #   accel → indices 0‑2, gyro → 3‑5, mag → 6‑8   (depending on which features are active)
                feat_offset = ax_idx * 3 + (line_idx % 3)
                line.set_data(time_history[sid], plot_data_history[sid][feat_offset])
                all_lines.append(line)

            # Keep X‑axis *window* centred on the newest sample
            latest_times = [time_history[s][-1] for s in sensors if time_history[s]]
            if latest_times:
                now = max(latest_times)
                axes[ax_idx].set_xlim(max(0, now - MAX_SECONDS), now + 0.5)

        return all_lines

    anim = animation.FuncAnimation(fig, update_plot, interval=30, cache_frame_data=False)
    plt.show()

# --------------------------------------------------------------
# 11️⃣ Head‑less fallback (runs when no plot or when the plot window is closed)
# --------------------------------------------------------------
if not USE_PLOT or not PLOT_WINDOW_OPEN:
    print("[INFO] Running in headless mode. Press Ctrl+C to stop.")
    while True:
        get_sensor_data()
        time.sleep(0.001)          # tiny sleep keeps CPU usage reasonable

