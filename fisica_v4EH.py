#!/usr/bin/env python3
"""
bno085_i2c_logger_multi.py – Log single or combined sensor fields over I2C
"""

import sys
import time
import datetime
import csv
import signal
import argparse
import collections
from pathlib import Path

import board
import busio
import adafruit_bno08x.i2c
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --------------------------------------------------------------
# 1️⃣ USER SETTINGS
# --------------------------------------------------------------
I2C_FREQUENCY = 400_000            # 400 kHz Fast-Mode
MAX_SECONDS   = 10                 # Live chart horizontal depth window
LOG_ROOT      = Path.cwd() / "logs"
USE_CSV       = True               # True -> CSV, False -> TSV

# --------------------------------------------------------------
# 2️⃣ Command‑line arguments
# --------------------------------------------------------------
parser = argparse.ArgumentParser(description="Log BNO085 streams over I2C.")
meas_group = parser.add_mutually_exclusive_group(required=True)
meas_group.add_argument("--accel", action="store_true", help="Log accelerometer (max 500 Hz).")
meas_group.add_argument("--gyro",  action="store_true", help="Log gyroscope (max 400 Hz).")
meas_group.add_argument("--mag",   action="store_true", help="Log magnetometer (max 100 Hz).")
meas_group.add_argument("--all",   action="store_true", help="Log Accel, Gyro, and Mag simultaneously (100 Hz).")

parser.add_argument("--duration", type=float, help="Run time limit in seconds.")
parser.add_argument("--plot", action="store_true", help="Enable async multi-axes live plotting.")
parser.add_argument("--no-rate", action="store_true", help="Disable console rate tracking printout.")

args = parser.parse_args()
RUN_FOR_SECONDS = args.duration               
USE_PLOT        = args.plot                    
PRINT_RATE      = not args.no_rate            

# --------------------------------------------------------------
# 3️⃣ Structural Feature Mapping Configuration
# --------------------------------------------------------------
ACTIVE_REPORTS = {}

if args.accel:
    MEAS_NAME = "accel"
    ACTIVE_REPORTS[adafruit_bno08x.BNO_REPORT_ACCELEROMETER] = int(1_000_000 / 500)
    HEADER_FIELDS = ["accel_x", "accel_y", "accel_z"]
elif args.gyro:
    MEAS_NAME = "gyro"
    ACTIVE_REPORTS[adafruit_bno08x.BNO_REPORT_GYROSCOPE] = int(1_000_000 / 400)
    HEADER_FIELDS = ["gyro_x", "gyro_y", "gyro_z"]
elif args.mag:
    MEAS_NAME = "mag"
    ACTIVE_REPORTS[adafruit_bno08x.BNO_REPORT_MAGNETOMETER] = int(1_000_000 / 100)
    HEADER_FIELDS = ["mag_x", "mag_y", "mag_z"]
else: # args.all configuration profile
    MEAS_NAME = "all"
    
    # Assign each sensor channel its absolute maximum datasheet target rate
    ACTIVE_REPORTS[adafruit_bno08x.BNO_REPORT_ACCELEROMETER] = int(1_000_000 / 500) # 500 Hz (2ms period)
    ACTIVE_REPORTS[adafruit_bno08x.BNO_REPORT_GYROSCOPE]     = int(1_000_000 / 400) # 400 Hz (2.5ms period)
    ACTIVE_REPORTS[adafruit_bno08x.BNO_REPORT_MAGNETOMETER]  = int(1_000_000 / 100) # 100 Hz (10ms period)
    
    HEADER_FIELDS = [
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z",
        "mag_x", "mag_y", "mag_z"
    ]

CSV_HEADER = ["timestamp"] + HEADER_FIELDS

# --------------------------------------------------------------
# 4️⃣ File System Helper Definitions
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
# 5️⃣ I2C Hardware Interfacing & Feature Activation
# --------------------------------------------------------------
print("[INFO] Mounting I2C physical bus...")
i2c = busio.I2C(board.SCL, board.SDA, frequency=I2C_FREQUENCY)
print("[INFO] Spawning internal BNO085 microcontroller handle...")
bno = adafruit_bno08x.i2c.BNO08X_I2C(i2c)

for report_type, period_us in ACTIVE_REPORTS.items():
    bno.enable_feature(report_type, period_us)

LOG_ROOT.mkdir(parents=True, exist_ok=True)
log_path = _make_log_path()
log_file, log_writer = _open_log_file(log_path)
print(f"[INFO] Operational telemetry routed to: {log_path}")

# Pipeline Register Setup
sample_times = collections.deque(maxlen=2000)   
LAST_RATE_PRINT = time.time()
START_RUN_TIME = time.time()
last_logged_vector = None  # Cache deduplication filter matrix register

def _maybe_print_rate():
    global LAST_RATE_PRINT
    if not PRINT_RATE: return
    now = time.time()
    if now - LAST_RATE_PRINT >= 0.4:
        while sample_times and (now - sample_times[0] > 1.0):
            sample_times.popleft()
        if sample_times:
            elapsed = now - sample_times[0]
            hz = len(sample_times) / elapsed if elapsed > 0 else 0.0
            print(f"[INFO] Stream throughput calculation ≈ {hz:.1f} Hz")
        LAST_RATE_PRINT = now

def safe_shutdown(signum, frame):
    print("\n[INFO] Flashing buffer lines to disk space. Closing log file safely.")
    log_file.close()
    sys.exit(0)

signal.signal(signal.SIGINT, safe_shutdown)
signal.signal(signal.SIGTERM, safe_shutdown)

# --------------------------------------------------------------
# 6️⃣ Asynchronous Direct Matrix Channel Interceptor
# --------------------------------------------------------------
# Track individual historical registries independently to bypass blocking loops
last_a, last_g, last_m = None, None, None

def poll_bno_pipeline():
    """Polls hardware channels dynamically, catches individual asynchronous updates,

    and handles EMI glitches by logging zeros when sensors are unreadable.
    """
    global last_a, last_g, last_m
    
    # Initialize blank data values
    current_a, current_g, current_m = None, None, None
    glitch_a, glitch_g, glitch_m = False, False, False

    # ---- 1️⃣ Read Individual Sensors with Isolated Try/Except Blocks ----
    if MEAS_NAME in ("accel", "all"):
        try:
            current_a = bno.acceleration
            if current_a is None or any(v is None for v in current_a):
                glitch_a = True
        except Exception:
            glitch_a = True

    if MEAS_NAME in ("gyro", "all"):
        try:
            current_g = bno.gyro
            if current_g is None or any(v is None for v in current_g):
                glitch_g = True
        except Exception:
            glitch_g = True

    if MEAS_NAME in ("mag", "all"):
        try:
            current_m = bno.magnetic
            if current_m is None or any(v is None for v in current_m):
                glitch_m = True
        except Exception:
            glitch_m = True

    # ---- 2️⃣ Assign Values or Zero Fallbacks based on Glitch Status ----
    out_a = [0.0, 0.0, 0.0] if glitch_a else (list(current_a) if current_a else [0.0, 0.0, 0.0])
    out_g = [0.0, 0.0, 0.0] if glitch_g else (list(current_g) if current_g else [0.0, 0.0, 0.0])
    out_m = [0.0, 0.0, 0.0] if glitch_m else (list(current_m) if current_m else [0.0, 0.0, 0.0])

    # If any error occurred, we force a zero-row record down the pipeline immediately
    if (MEAS_NAME == "all" and (glitch_a or glitch_g or glitch_m)) or \
       (MEAS_NAME == "accel" and glitch_a) or \
       (MEAS_NAME == "gyro" and glitch_g) or \
       (MEAS_NAME == "mag" and glitch_m):
        
        if MEAS_NAME == "all": return out_a + out_g + out_m
        elif MEAS_NAME == "accel": return out_a
        elif MEAS_NAME == "gyro": return out_g
        else: return out_m

    # ---- 3️⃣ No Glitches: Validate Value Changes to Avoid Duplications ----
    has_changed = False
    
    if MEAS_NAME == "accel" and current_a != last_a:
        last_a = current_a
        has_changed = True
    elif MEAS_NAME == "gyro" and current_g != last_g:
        last_g = current_g
        has_changed = True
    elif MEAS_NAME == "mag" and current_m != last_m:
        last_m = current_m
        has_changed = True
    elif MEAS_NAME == "all" and (current_a != last_a or current_g != last_g or current_m != last_m):
        if current_a: last_a = current_a
        if current_g: last_g = current_g
        if current_m: last_m = current_m
        has_changed = True

    if has_changed:
        if MEAS_NAME == "all": return out_a + out_g + out_m
        elif MEAS_NAME == "accel": return out_a
        elif MEAS_NAME == "gyro": return out_g
        else: return out_m

    return None

# --------------------------------------------------------------
# 7️⃣ Runtime Strategy Fork (Live Visual Plotting vs Headless Logging)
# --------------------------------------------------------------
if USE_PLOT:
    max_pts = int(MAX_SECONDS * (100 if MEAS_NAME == "all" else 400))
    t_history = collections.deque(maxlen=max_pts)
    
    # Generate trace channels for every field column dynamically
    traces = {i: collections.deque(maxlen=max_pts) for i in range(len(HEADER_FIELDS))}
    
    if MEAS_NAME == "all":
        fig, subplots = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        ax_accel, ax_gyro, ax_mag = subplots
    else:
        fig, single_ax = plt.subplots(figsize=(10, 5))
        subplots = [single_ax]

    fig.suptitle(f"BNO085 Telemetry Node Monitor [{MEAS_NAME.upper()}]", fontsize=12, fontweight="bold")
    
    line_objects = []
    if MEAS_NAME == "all":
        # Group vectors cleanly onto independent grid subplots
        line_objects.append(ax_accel.plot([], [], color="crimson", label="Accel X")[0])
        line_objects.append(ax_accel.plot([], [], color="salmon", label="Accel Y")[0])
        line_objects.append(ax_accel.plot([], [], color="darkred", label="Accel Z")[0])
        
        line_objects.append(ax_gyro.plot([], [], color="teal", label="Gyro X")[0])
        line_objects.append(ax_gyro.plot([], [], color="turquoise", label="Gyro Y")[0])
        line_objects.append(ax_gyro.plot([], [], color="cadetblue", label="Gyro Z")[0])
        
        line_objects.append(ax_mag.plot([], [], color="goldenrod", label="Mag X")[0])
        line_objects.append(ax_mag.plot([], [], color="khaki", label="Mag Y")[0])
        line_objects.append(ax_mag.plot([], [], color="darkgoldenrod", label="Mag Z")[0])
        
        ax_accel.set_ylabel("Accel (m/s²)")
        ax_gyro.set_ylabel("Gyro (rad/s)")
        ax_mag.set_ylabel("Mag (µT)")
        
        ax_accel.set_ylim(-16, 16)
        ax_gyro.set_ylim(-6, 6)
        ax_mag.set_ylim(-75, 75)
    else:
        line_objects.append(single_ax.plot([], [], color="crimson", label="Axis X")[0])
        line_objects.append(single_ax.plot([], [], color="teal", label="Axis Y")[0])
        line_objects.append(single_ax.plot([], [], color="goldenrod", label="Axis Z")[0])
        single_ax.set_ylabel("Magnitude")
        
        if MEAS_NAME == "accel": single_ax.set_ylim(-16, 16)
        elif MEAS_NAME == "gyro": single_ax.set_ylim(-6, 6)
        else: single_ax.set_ylim(-75, 75)

    for ax in subplots:
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
    subplots[-1].set_xlabel("Time (s)")

    def process_plot_tick(frame):
        global last_logged_vector
        now = time.time()
        
        if RUN_FOR_SECONDS and (now - START_RUN_TIME >= RUN_FOR_SECONDS):
            plt.close(fig)
            return line_objects

        dataset = poll_bno_pipeline()
        # Clean structural list matching comparison now operates reliably
        if dataset is not None and dataset != last_logged_vector:
            last_logged_vector = dataset
            rel_t = now - START_RUN_TIME
            iso_stamp = datetime.datetime.now().isoformat()
            
            row = [iso_stamp] + dataset
            if USE_CSV: log_writer.writerow(row)
            else: log_file.write("\t".join(str(v) for v in row) + "\n")
            
            sample_times.append(now)
            t_history.append(rel_t)
            for idx, val in enumerate(dataset):
                traces[idx].append(val)

        _maybe_print_rate()

        if t_history:
            for idx, line in enumerate(line_objects):
                line.set_data(list(t_history), list(traces[idx]))
            subplots[-1].set_xlim(t_history[0], max(t_history[-1], t_history[0] + 0.1))
            
        return line_objects

    anim = animation.FuncAnimation(fig, process_plot_tick, interval=10, blit=True, cache_frame_data=False)
    plt.show()
    log_file.close()

else:
    print("[INFO] Maximum throughput streaming engine initialized.")
    print("[INFO] Target specs: Accel 500Hz, Gyro 400Hz, Mag 100Hz. Logging active...")
    
    try:
        while True:
            loop_tick = time.time()
            if RUN_FOR_SECONDS and (loop_tick - START_RUN_TIME >= RUN_FOR_SECONDS):
                break

            dataset = poll_bno_pipeline()
            if dataset is not None:
                iso_stamp = datetime.datetime.now().isoformat()
                row = [iso_stamp] + dataset
                
                if USE_CSV: log_writer.writerow(row)
                else: log_file.write("\t".join(str(v) for v in row) + "\n")
                
                sample_times.append(loop_tick)

            _maybe_print_rate()
            
            # Tiny Micro-delay to avoid Linux CPU locking, keeping maximum bus priority active
            time.sleep(0.0001)

    except KeyboardInterrupt:
        pass
    finally:
        log_file.close()
        print("[INFO] Data processing completed. Stream saved safely.")
