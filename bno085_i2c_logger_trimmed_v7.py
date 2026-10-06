#!/usr/bin/env python3
"""
bno085_i2c_logger_trimmed_v6.py – Log and plot 2 BNO085 sensors over I²C simultaneously.

Features
--------
* Automatically targets up to 2 BNO085 sensors at addresses 0x4A and 0x4B.
* Flexible configurations: log individual channels or all three (accel, gyro, mag) together.
* Headless mode by default – live Matplotlib plotting runs only if --plot is called.
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
from adafruit_bno08x.i2c import BNO08X_I2C

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
# 1️⃣  USER & HARDWARE SETTINGS
# --------------------------------------------------------------
I2C_FREQUENCY = 400_000            # 400 kHz – Pi fast‑mode (max supported)
MAX_SECONDS   = 30                 # seconds showed on the live plot (if enabled)
LOG_ROOT      = Path.cwd() / "logs"
USE_CSV       = True               # CSV → .csv, False → TSV

# Up to 2 sensors mapped to their hardware jumpers (DI pin open vs tied to 3V3)
POSSIBLE_I2C_ADDRESSES = [0x4A, 0x4B]

ALL_FEATURES = {
    "accel": (adafruit_bno08x.BNO_REPORT_ACCELEROMETER, 500, ["accel_x", "accel_y", "accel_z"], (-20, 20)),
    "gyro":  (adafruit_bno08x.BNO_REPORT_GYROSCOPE,  400, ["gyro_x", "gyro_y", "gyro_z"], (-500, 500)),
    "mag":   (adafruit_bno08x.BNO_REPORT_MAGNETOMETER, 100, ["mag_x", "mag_y", "mag_z"], (-2000, 2000)),
}

# --------------------------------------------------------------
# 2️⃣  Command‑line arguments
# --------------------------------------------------------------
parser = argparse.ArgumentParser(description="Log multiple BNO085 sensors dynamically over I2C.")
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
# 3️⃣  Determine active feature set configurations
# --------------------------------------------------------------
ACTIVE_FEATURES = {}
if args.all:
    ACTIVE_FEATURES = ALL_FEATURES
else:
    for key in ["accel", "gyro", "mag"]:
        if getattr(args, key):
            ACTIVE_FEATURES[key] = ALL_FEATURES[key]

# --------------------------------------------------------------
# 4️⃣  Initialization & Discovery
# --------------------------------------------------------------
print("[INFO] Initialising I²C Bus at 400kHz …")
i2c = busio.I2C(board.SCL, board.SDA, frequency=I2C_FREQUENCY)

sensors = {}
CSV_HEADER = ["timestamp"]

for addr in POSSIBLE_I2C_ADDRESSES:
    try:
        print(f"[INFO] Scanning for BNO085 at address: {hex(addr)}...")
        bno = BNO08X_I2C(i2c, address=addr)
        sensors[addr] = bno
        print(f"[SUCCESS] Found BNO085 at address: {hex(addr)}")
        
        # Build headers for active features for this sensor
        for feat_name, (_, _, subfields, _) in ACTIVE_FEATURES.items():
            for field in subfields:
                CSV_HEADER.append(f"bno_{hex(addr)}_{field}")
    except Exception:
        print(f"[INFO] No sensor responding at {hex(addr)}")

if not sensors:
    print("[ERROR] No BNO085 sensors detected on the I2C bus. Check wiring.")
    sys.exit(1)

# Enable the selected data channels on each discovered device
for addr, bno in sensors.items():
    print(f"[INFO] Configuring features for BNO085 ({hex(addr)}):")
    for name, (report_type, max_hz, _, _) in ACTIVE_FEATURES.items():
        period_us = int(1_000_000 / max_hz)
        bno.enable_feature(report_type, period_us)
        print(f"       -> Enabled {name.upper()} at {max_hz}Hz (Period: {period_us}µs)")

# --------------------------------------------------------------
# 5️⃣  File Generation Setup
# --------------------------------------------------------------
def _make_log_path() -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "csv" if USE_CSV else "txt"
    feat_label = "all" if args.all else "_".join(list(ACTIVE_FEATURES.keys()))
    return LOG_ROOT / f"bno085_multi_{feat_label}_{stamp}.{suffix}"

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

# Global switch flag to track visualization window state
PLOT_WINDOW_OPEN = True

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
    """Reads sensor data, logs it, and formats it for plotting."""
    ts = datetime.datetime.now().isoformat()
    row = [ts]
    data_captured = False
    all_devices_frame = {}

    for addr, bno in sensors.items():
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
            
            # If any real float data was read out
            if any(v is not None for v in dev_data):
                data_captured = True
            
            # Swap Nones with zeroes for file padding consistency
            clean_data = [v if v is not None else 0.0 for v in dev_data]
            row.extend(clean_data)
            all_devices_frame[addr] = clean_data
        except Exception:
            # Pad empty array values if an I2C transaction times out or slips
            pad_len = len(ACTIVE_FEATURES) * 3
            row.extend([0.0] * pad_len)
            all_devices_frame[addr] = [0.0] * pad_len

    if not data_captured:
        _maybe_print_rate()
        return None

    if USE_CSV:
        log_writer.writerow(row)
    else:
        log_file.write("\t".join(str(v) for v in row) + "\n")

    sample_times.append(time.time())
    _maybe_print_rate()

    # Duration execution checker
    if RUN_FOR_SECONDS is not None and (time.time() - run_start) >= RUN_FOR_SECONDS:
        print(f"\n[INFO] Run‑time limit of {RUN_FOR_SECONDS:.1f}s reached.")
        _cleanup_and_exit()

    return time.time() - start_time, all_devices_frame

# --------------------------------------------------------------
# 8️⃣  Execution Block (Plotting Engine vs Headless Log Engine)
# --------------------------------------------------------------
if USE_PLOT:
    plt.style.use("seaborn-v0_8-darkgrid")
    
    # Generate subplots dynamically based on how many sub-features are activated
    num_plots = len(ACTIVE_FEATURES)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]
        
    plot_mappings = {}
    colors = ["#ff5555", "#55ff55", "#5555ff", "#ffa500", "#8a2be2", "#00ffff"]
    
    # Store dynamic line allocations across multiple dimensions
    plot_data_history = {addr: collections.defaultdict(list) for addr in sensors}
    time_history = {addr: [] for addr in sensors}
    
    feature_keys = list(ACTIVE_FEATURES.keys())
    for ax_idx, feat_name in enumerate(feature_keys):
        _, _, _, y_bounds = ACTIVE_FEATURES[feat_name]
        axes[ax_idx].set_title(f"Live {feat_name.capitalize()} Streams")
        axes[ax_idx].set_ylabel(f"{feat_name.capitalize()}")
        axes[ax_idx].set_ylim(y_bounds)
        axes[ax_idx].grid(True)
        
        plot_mappings[feat_name] = []
        color_idx = 0
        for addr in sensors.keys():
            for axis_lbl in ["X", "Y", "Z"]:
                line, = axes[ax_idx].plot([], [], label=f"{hex(addr)}_{axis_lbl}", color=colors[color_idx % len(colors)])
                plot_mappings[feat_name].append((addr, line))
                color_idx += 1
        axes[ax_idx].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("Time (s)")

    def on_close(event):
        global PLOT_WINDOW_OPEN
        PLOT_WINDOW_OPEN = False
        print("\n[INFO] Plot window closed. Switching seamlessly to high-throughput headless logging...")

    fig.canvas.mpl_connect('close_event', on_close)

    def update_plot(frame):
        if not PLOT_WINDOW_OPEN:
            return []

        # Consume entries off the physical I2C register block
        for _ in range(4):
            result = get_sensor_data()
            if result is None:
                break
            
            now, devices_frame = result
            for addr, data_list in devices_frame.items():
                time_history[addr].append(now)
                # Parse tracking elements sequentially into historic records
                for idx, val in enumerate(data_list):
                    plot_data_history[addr][idx].append(val)
                
                # Trim historic rolling display values
                while time_history[addr] and (now - time_history[addr][0] > MAX_SECONDS):
                    time_history[addr].pop(0)
                    for idx in plot_data_history[addr].keys():
                        plot_data_history[addr][idx].pop(0)

        # Re-plot line streams across visual frameworks
        all_lines = []
        for ax_idx, feat_name in enumerate(feature_keys):
            sub_mappings = plot_mappings[feat_name]
            for line_idx, (addr, line) in enumerate(sub_mappings):
                feat_offset = ax_idx * 3 + (line_idx % 3)
                line.set_data(time_history[addr], plot_data_history[addr][feat_offset])
                all_lines.append(line)
            
            valid_times = [time_history[a][-1] for a in sensors if time_history[a]]
            if valid_times:
                max_now = max(valid_times)
                axes[ax_idx].set_xlim(max(0, max_now - MAX_SECONDS), max_now + 0.5)
                
        return all_lines
            
    anim = animation.FuncAnimation(fig, update_plot, interval=30, cache_frame_data=False)
    plt.show()
        
# --------------------------------------------------------------
# 9️⃣ Continuous Headless Logging (Default Fallback Loop)
# --------------------------------------------------------------
if not USE_PLOT or not PLOT_WINDOW_OPEN:
    print("[INFO] Running in headless mode. Press Ctrl+C to stop.")
    while True:
        get_sensor_data()
        time.sleep(0.001)
